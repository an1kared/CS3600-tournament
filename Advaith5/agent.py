import time
import numpy as np
import random
from collections.abc import Callable
from typing import Tuple

from game import board, move, enums

# The definitive scoring table
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
        return "Executing Targeted Expectiminimax with Contested Carpet Heuristic."

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

        # 1. Calculate Expected Value of the best possible search
        best_search_move = None
        search_ev = -float('inf')
        
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            p_catch = self.tracker.belief[best_cell_idx]
            
            # Mathematical EV: Catch is +4, Miss is -2
            search_ev = (p_catch * 4.0) + ((1.0 - p_catch) * -2.0)
            
            # Only consider searching if EV is strictly positive (p > 0.333)
            if search_ev > 0.1:  
                y, x = divmod(best_cell_idx, 8)
                best_search_move = move.Move.search((x, y))

        time_remaining = time_left()
        turns_left = max(1, self.total_turns - self.turns_taken)
        time_budget = (time_remaining / turns_left) * 1.2
        time_budget = min(time_budget, time_remaining - 0.2) 
        
        safe_moves = [m for m in moves if m.move_type != enums.MoveType.SEARCH]
        best_move = random.choice(safe_moves) if safe_moves else moves[0]
        depth = 1
        best_spatial_eval = -float('inf')
        
        try:
            while depth < 12: 
                eval_score, current_best_move = self.spatial_minimax(
                    board_state, depth, -float('inf'), float('inf'), 
                    True, start_time, time_budget
                )
                if current_best_move:
                    best_spatial_eval = eval_score
                    best_move = current_best_move
                depth += 1
        except TimeoutError:
            pass 

        self.turns_taken += 1
        
        # 2. Chance Node Resolution: Compare Search EV against Spatial Tree Eval
        # We scale the search EV by 100 to match the heuristic multiplier
        if best_search_move and (search_ev * 100) > best_spatial_eval:
            return best_search_move
            
        return best_move

    def spatial_minimax(self, current_board, depth, alpha, beta, is_maximizing, start_time, time_limit):
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
                eval_score, _ = self.spatial_minimax(next_board, depth - 1, alpha, beta, False, start_time, time_limit)
                
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
                eval_score, _ = self.spatial_minimax(next_board, depth - 1, alpha, beta, True, start_time, time_limit)
                
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
                    return -10
                return 100 + (m.roll_length * 5)
            if m.move_type == enums.MoveType.PRIME: return 50
            return 0
        return sorted(moves, key=move_priority, reverse=True)

    def _contested_carpet_potential(self, b: board.Board, my_pos, opp_pos) -> float:
        """
        Evaluates potential carpet lines, but factors in who is closer to the start of the line.
        If the opponent is closer to a line of primes than we are, it is a negative score.
        """
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        net_potential = 0.0
        
        for dx, dy in directions:
            primed = 0
            space = 0
            cx, cy = my_pos[0] + dx, my_pos[1] + dy
            
            while b.is_valid_cell((cx, cy)) and b.get_cell((cx, cy)) == enums.Cell.PRIMED:
                primed += 1
                cx += dx
                cy += dy
                
            while b.is_valid_cell((cx, cy)) and b.get_cell((cx, cy)) == enums.Cell.SPACE:
                space += 1
                cx += dx
                cy += dy
                
            run = primed + space
            if run >= 2:
                potential_points = CARPET_PTS.get(min(run, 7), 21)
                
                # Check contested distance to the start of this specific line
                start_x, start_y = my_pos[0] + dx, my_pos[1] + dy
                my_dist = abs(my_pos[0] - start_x) + abs(my_pos[1] - start_y)
                opp_dist = abs(opp_pos[0] - start_x) + abs(opp_pos[1] - start_y)
                
                if my_dist <= opp_dist:
                    net_potential += potential_points
                else:
                    # Penalty for setting up a line the opponent can steal
                    net_potential -= potential_points * 1.5 
                    
        return net_potential

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

        my_pos = my_worker.get_location()
        opp_pos = opp_worker.get_location()

        score = (my_worker.get_points() - opp_worker.get_points()) * 100
        
        # Immediate Carpet Opportunities
        my_carpet_value = sum(
            CARPET_PTS.get(m.roll_length, 0)
            for m in my_moves
            if m.move_type == enums.MoveType.CARPET and m.roll_length >= 2
        )
        score += my_carpet_value * 20

        opp_carpet_value = sum(
            CARPET_PTS.get(m.roll_length, 0)
            for m in opp_moves
            if m.move_type == enums.MoveType.CARPET and m.roll_length >= 2
        )
        # Block opponent carpets heavily
        score -= opp_carpet_value * 25 

        score += len(my_moves) * 2
        score -= len(opp_moves) * 3

        # Rat proximity based on HMM belief
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            if best_prob > 0.3:
                ry, rx = divmod(best_cell_idx, 8)
                dist = abs(my_pos[0] - rx) + abs(my_pos[1] - ry)
                score += (14 - dist) * 4

        # Defensive Carpet Potential
        my_potential = self._contested_carpet_potential(b, my_pos, opp_pos)
        opp_potential = self._contested_carpet_potential(b, opp_pos, my_pos)
        
        score += my_potential * 10
        score -= opp_potential * 12

        return score