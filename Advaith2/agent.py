import time
import numpy as np
import random
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

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
            cell_loc = (x, y)
            cell_type = board_state.get_cell(cell_loc)
            
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
            probs = {enums.Noise.SQUEAK: 0.3, enums.Noise.SCRATCH: 0.5, enums.Noise.SQUEAL: 0.2}
        elif cell_type == enums.Cell.SPACE:
            probs = {enums.Noise.SQUEAK: 0.15, enums.Noise.SCRATCH: 0.7, enums.Noise.SQUEAL: 0.15}
        elif cell_type == enums.Cell.PRIMED:
            probs = {enums.Noise.SQUEAK: 0.8, enums.Noise.SCRATCH: 0.1, enums.Noise.SQUEAL: 0.1}
        elif cell_type == enums.Cell.CARPET:
            probs = {enums.Noise.SQUEAK: 0.1, enums.Noise.SCRATCH: 0.1, enums.Noise.SQUEAL: 0.8}
        else:
            probs = {enums.Noise.SQUEAK: 0.33, enums.Noise.SCRATCH: 0.33, enums.Noise.SQUEAL: 0.34}
        return probs.get(noise_enum, 0.0)
        
    def _get_dist_prob(self, true_dist: int, est_dist: int) -> float:
        diff = est_dist - true_dist
        if true_dist == 0 and est_dist == 0: return 0.82 
        if diff == -1: return 0.12
        if diff == 0: return 0.70
        if diff == 1: return 0.12
        if diff == 2: return 0.06
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
            
        # 1. Handle Rat Searching independently of the Minimax tree
        moves = board_state.get_valid_moves(exclude_search=False)
        if not moves:
            return None
            
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            
            # The mathematical EV threshold for searching
            if best_prob > 0.60:
                y, x = divmod(best_cell_idx, 8)
                search_move = move.Move.search((x, y))
                if search_move in moves:
                    self.turns_taken += 1
                    return search_move

        # 2. Time Management for Iterative Deepening
        time_remaining = time_left()
        turns_left = max(1, self.total_turns - self.turns_taken)
        time_budget = (time_remaining / turns_left) * 0.9 # Leave a 10% safety buffer
        time_budget = min(time_budget, time_remaining - 0.5) # Hard minimum buffer
        
        best_move = random.choice([m for m in moves if m.move_type != enums.MoveType.SEARCH])
        depth = 1
        
        try:
            while depth < 10: # Arbitrary high max depth
                eval_score, current_best_move = self.minimax(
                    board_state, depth, -float('inf'), float('inf'), 
                    True, start_time, time_budget
                )
                if current_best_move:
                    best_move = current_best_move
                depth += 1
        except TimeoutError:
            # Time budget exhausted, fall back to the best move found at the previous completed depth
            pass

        self.turns_taken += 1
        return best_move

    def minimax(self, current_board, depth, alpha, beta, is_maximizing, start_time, time_limit):
        if time.time() - start_time > time_limit:
            raise TimeoutError()

        if depth == 0 or current_board.is_game_over():
            return self.evaluate_board(current_board, is_maximizing), None

        # Exclude searches from the tree to prevent hallucinated points
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
                    break # Beta cutoff
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
                    break # Alpha cutoff
            return min_eval, best_move

    def order_moves(self, moves):
        # Move ordering improves Alpha-Beta pruning efficiency
        def move_priority(m):
            if m.move_type == enums.MoveType.CARPET: return 3
            if m.move_type == enums.MoveType.PRIME: return 2
            return 1
        return sorted(moves, key=move_priority, reverse=True)

    def evaluate_board(self, b: board.Board, is_maximizing: bool) -> float:
        # We always want the evaluation to be from the root agent's perspective
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
        score += len(my_moves) * 2
        score -= len(opp_moves) * 2

        # Reward positioning near the rat even if we aren't searching yet
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            if best_prob > 0.4:
                ry, rx = divmod(best_cell_idx, 8)
                my_loc = my_worker.get_location()
                dist = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
                # Max distance on 8x8 is 14
                score += (14 - dist) * 5

        return score