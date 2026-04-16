import time
import numpy as np
import random
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

CARPET_PTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}


class RatTracker:
    def __init__(self, T: np.ndarray):
        self.T = T
        self.num_cells = 64
        self.belief = np.zeros(self.num_cells)
        self.belief[0] = 1.0
        for _ in range(1000):
            self.belief = self.belief @ self.T

    def update(self, board_state: board.Board, noise_enum: enums.Noise, distance_estimate: int):
        self.belief = self.belief @ self.T
        likelihood = np.zeros(self.num_cells)
        worker_loc = board_state.player_worker.get_location()
        for i in range(self.num_cells):
            y, x = divmod(i, 8)
            cell_type = board_state.get_cell((x, y))
            prob_noise = self._get_noise_prob(cell_type, noise_enum)
            true_dist = abs(worker_loc[0] - x) + abs(worker_loc[1] - y)
            prob_dist = self._get_dist_prob(true_dist, distance_estimate)
            likelihood[i] = prob_noise * prob_dist
        self.belief = self.belief * likelihood
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            self.belief = np.ones(self.num_cells) / self.num_cells

    def _get_noise_prob(self, cell_type: enums.Cell, noise_enum: enums.Noise) -> float:
        if cell_type == enums.Cell.BLOCKED:
            probs = {enums.Noise.SQUEAK: 0.5, enums.Noise.SCRATCH: 0.3, enums.Noise.SQUEAL: 0.2}
        elif cell_type == enums.Cell.SPACE:
            probs = {enums.Noise.SQUEAK: 0.7, enums.Noise.SCRATCH: 0.15, enums.Noise.SQUEAL: 0.15}
        elif cell_type == enums.Cell.PRIMED:
            probs = {enums.Noise.SQUEAK: 0.1, enums.Noise.SCRATCH: 0.8, enums.Noise.SQUEAL: 0.1}
        elif cell_type == enums.Cell.CARPET:
            probs = {enums.Noise.SQUEAK: 0.1, enums.Noise.SCRATCH: 0.1, enums.Noise.SQUEAL: 0.8}
        else:
            probs = {enums.Noise.SQUEAK: 0.33, enums.Noise.SCRATCH: 0.33, enums.Noise.SQUEAL: 0.34}
        return probs.get(noise_enum, 0.0)

    def _get_dist_prob(self, true_dist: int, est_dist: int) -> float:
        diff = est_dist - true_dist
        if true_dist == 0 and est_dist == 0: return 0.82
        if diff == -1: return 0.12
        if diff == 0:  return 0.70
        if diff == 1:  return 0.12
        if diff == 2:  return 0.06
        return 0.0


class PlayerAgent:
    def __init__(self, board_state: board.Board, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix) if transition_matrix is not None else None
        self.tracker = RatTracker(self.T) if self.T is not None else None
        self.total_turns = enums.MAX_TURNS_PER_PLAYER
        self.turns_taken = 0

    def commentate(self):
        return "Running Iterative Deepening Minimax."

    def play(self, board_state: board.Board, sensor_data: Tuple, time_left: Callable):
        start_time = time.time()
        noise_val, dist_val = sensor_data

        last_search_loc, last_search_result = board_state.player_search
        if last_search_result and self.tracker:
            self.tracker = RatTracker(self.T)

        if self.tracker:
            self.tracker.update(board_state, noise_val, dist_val)

        moves = board_state.get_valid_moves(exclude_search=False)
        if not moves:
            return None

        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            # Lower threshold: EV positive when p > 0.333; use 0.40 so we actually search
            # when the HMM has reasonable confidence, rather than the original 0.85 that never fires
            if best_prob > 0.40:
                y, x = divmod(best_cell_idx, 8)
                search_move = move.Move.search((x, y))
                if search_move in moves:
                    self.turns_taken += 1
                    return search_move

        time_remaining = time_left()
        turns_left = max(1, self.total_turns - self.turns_taken)
        time_budget = (time_remaining / turns_left) * 0.9
        time_budget = min(time_budget, time_remaining - 0.5)

        best_move = random.choice([m for m in moves if m.move_type != enums.MoveType.SEARCH])
        depth = 1

        try:
            while depth < 10:
                eval_score, current_best_move = self.minimax(
                    board_state, depth, -float('inf'), float('inf'),
                    True, start_time, time_budget
                )
                if current_best_move:
                    best_move = current_best_move
                depth += 1
        except TimeoutError:
            pass

        self.turns_taken += 1
        return best_move

    def minimax(self, current_board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        if depth == 0 or current_board.is_game_over():
            return self.evaluate_board(current_board, is_maximizing), None

        moves = current_board.get_valid_moves(exclude_search=True)
        if not moves:
            return self.evaluate_board(current_board, is_maximizing), None

        moves = self.order_moves(moves)
        best_move = moves[0]

        if is_maximizing:
            max_eval = -float('inf')
            for m in moves:
                next_board = current_board.forecast_move(m, check_ok=False)
                if not next_board: continue
                next_board.reverse_perspective()
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, False, start_time, time_limit)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = m
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for m in moves:
                next_board = current_board.forecast_move(m, check_ok=False)
                if not next_board: continue
                next_board.reverse_perspective()
                eval_score, _ = self.minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = m
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def order_moves(self, moves):
        def move_priority(m):
            if m.move_type == enums.MoveType.CARPET:
                if m.roll_length == 1:
                    return -1
                return 3 + m.roll_length * 2
            if m.move_type == enums.MoveType.PRIME: return 2
            return 1
        return sorted(moves, key=move_priority, reverse=True)

    def _carpet_potential(self, b: board.Board, pos) -> float:
        """
        For each direction from pos, compute the value of the longest carpet
        we could EVENTUALLY roll: count contiguous primed squares already there,
        then contiguous space squares we could still prime.
        Return the sum of potential carpet values across all 4 directions.
        This incentivises positioning near long prime chains before rolling.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        total = 0.0
        for dx, dy in directions:
            primed = 0
            space = 0
            cx, cy = pos[0] + dx, pos[1] + dy
            # Count contiguous primed squares in this direction
            while b.is_valid_cell((cx, cy)) and b.get_cell((cx, cy)) == enums.Cell.PRIMED:
                primed += 1
                cx += dx
                cy += dy
            # Count contiguous space squares beyond (future primes)
            while b.is_valid_cell((cx, cy)) and b.get_cell((cx, cy)) == enums.Cell.SPACE:
                space += 1
                cx += dx
                cy += dy
            run = primed + space
            if run >= 2:
                # Value = points we'd score if we eventually built and rolled this whole run
                total += CARPET_PTS.get(min(run, 7), 21)
        return total

    def evaluate_board(self, b: board.Board, is_maximizing: bool) -> float:
        if is_maximizing:
            my_worker = b.player_worker
            opp_worker = b.opponent_worker
            my_moves = b.get_valid_moves(enemy=False, exclude_search=True)
            opp_moves = b.get_valid_moves(enemy=True, exclude_search=True)
        else:
            my_worker = b.opponent_worker
            opp_worker = b.player_worker
            my_moves = b.get_valid_moves(enemy=True, exclude_search=True)
            opp_moves = b.get_valid_moves(enemy=False, exclude_search=True)

        score = (my_worker.get_points() - opp_worker.get_points()) * 100

        # Current carpet opportunities (unchanged from original)
        my_carpet_value = sum(
            enums.CARPET_POINTS_TABLE[m.roll_length]
            for m in my_moves
            if m.move_type == enums.MoveType.CARPET and m.roll_length >= 2
        )
        score += my_carpet_value * 15

        opp_carpet_value = sum(
            enums.CARPET_POINTS_TABLE[m.roll_length]
            for m in opp_moves
            if m.move_type == enums.MoveType.CARPET and m.roll_length >= 2
        )
        score -= opp_carpet_value * 10

        score += len(my_moves) * 2
        score -= len(opp_moves) * 3

        # Rat proximity (unchanged from original)
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            if best_prob > 0.4:
                ry, rx = divmod(best_cell_idx, 8)
                my_loc = my_worker.get_location()
                dist = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
                score += (14 - dist) * 3

        my_pos = my_worker.get_location()
        opp_pos = opp_worker.get_location()

        # Open adjacent space squares (unchanged from original)
        open_adjacent = sum(
            1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if b.is_valid_cell((my_pos[0] + dx, my_pos[1] + dy))
            and b.get_cell((my_pos[0] + dx, my_pos[1] + dy)) == enums.Cell.SPACE
        )
        score += open_adjacent * 5

        # NEW: carpet potential — reward being near long buildable prime runs.
        # This is what Albert Lite does better: it positions to roll len=4+ carpets.
        # We value our potential more than opponent's to encourage long-chain planning.
        my_potential = self._carpet_potential(b, my_pos)
        opp_potential = self._carpet_potential(b, opp_pos)
        score += my_potential * 6
        score -= opp_potential * 4

        return score