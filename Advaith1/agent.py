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
        # The rat starts at (0,0)
        self.belief[0] = 1.0 
        
        # The rat is given a 1000 step headstart before the game begins
        for _ in range(1000):
            self.belief = self.belief @ self.T
            
    def update(self, board_state: board.Board, noise_enum: enums.Noise, distance_estimate: int):
        # 1. Prediction step: multiply by transition matrix
        self.belief = self.belief @ self.T
        
        # 2. Update step: factor in sensor data
        likelihood = np.zeros(self.num_cells)
        worker_loc = board_state.player_worker.get_location()
        
        for i in range(self.num_cells):
            y, x = divmod(i, 8)
            cell_loc = (x, y)
            cell_type = board_state.get_cell(cell_loc)
            
            # Calculate noise likelihood
            prob_noise = self._get_noise_prob(cell_type, noise_enum)
            
            # Calculate distance likelihood
            true_dist = abs(worker_loc[0] - x) + abs(worker_loc[1] - y)
            prob_dist = self._get_dist_prob(true_dist, distance_estimate)
            
            likelihood[i] = prob_noise * prob_dist
            
        self.belief = self.belief * likelihood
        
        # Normalize the belief distribution
        total_prob = np.sum(self.belief)
        if total_prob > 0:
            self.belief /= total_prob
        else:
            # Fallback in case of severe rounding errors to prevent crashing
            self.belief = np.ones(self.num_cells) / self.num_cells
            
    def _get_noise_prob(self, cell_type: enums.Cell, noise_enum: enums.Noise) -> float:
        # Probabilities mapped from the assignment tables
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
        
        # Address the edge case where estimates cannot be less than 0
        if true_dist == 0 and est_dist == 0:
            # It could be the 'correct' roll (0.7) or the 'one less' roll (0.12) which got floored to 0
            return 0.82 
            
        if diff == -1: return 0.12
        if diff == 0: return 0.70
        if diff == 1: return 0.12
        if diff == 2: return 0.06
        
        return 0.0

class PlayerAgent:
    def __init__(self, board_state: board.Board, transition_matrix=None, time_left: Callable = None):
        self.T = np.array(transition_matrix) if transition_matrix is not None else None
        self.tracker = RatTracker(self.T) if self.T is not None else None
        
    def commentate(self):
        return "Executing systematic floor coverage."

    def play(self, board_state: board.Board, sensor_data: Tuple, time_left: Callable):
        noise_val, dist_val = sensor_data
        
        # Check if we successfully caught the rat last turn. If so, reset the tracker.
        last_search_loc, last_search_result = board_state.player_search
        if last_search_result and self.tracker:
            self.tracker = RatTracker(self.T)
            
        # Update our belief about the rat's location
        if self.tracker:
            self.tracker.update(board_state, noise_val, dist_val)
            
        moves = board_state.get_valid_moves(exclude_search=False)
        if not moves:
            return None
            
        # 1. Check if searching is the optimal expected value
        if self.tracker:
            best_cell_idx = int(np.argmax(self.tracker.belief))
            best_prob = self.tracker.belief[best_cell_idx]
            
            # Points EV formula: (Prob * 4) + ((1 - Prob) * -2). 
            # We search if EV > 1.5, meaning Prob > ~0.58
            if best_prob > 0.60:
                y, x = divmod(best_cell_idx, 8)
                search_move = move.Move.search((x, y))
                if search_move in moves:
                    return search_move
        
        # 2. Filter out search moves and evaluate standard movement
        reg_moves = [m for m in moves if m.move_type != enums.MoveType.SEARCH]
        if not reg_moves:
            # Fallback if only search is available (rare)
            return random.choice(moves)
            
        best_move = reg_moves[0]
        best_score = -float('inf')
        
        for m in reg_moves:
            score = self.evaluate_move(board_state, m)
            if score > best_score:
                best_score = score
                best_move = m
                
        return best_move
        
    def evaluate_move(self, current_board: board.Board, m: move.Move) -> float:
        score = 0.0
        
        if m.move_type == enums.MoveType.PRIME:
            score += 1.0
        elif m.move_type == enums.MoveType.CARPET:
            # We heavily weight carpeting longer stretches
            score += enums.CARPET_POINTS_TABLE[m.roll_length] * 2.5 
        elif m.move_type == enums.MoveType.PLAIN:
            score += 0.0
            
        # Look ahead one step to evaluate positional advantage
        next_board = current_board.forecast_move(m, check_ok=False)
        if next_board:
            my_loc = next_board.player_worker.get_location()
            
            # Penalize moving to the edges of the board where you can get stuck
            center_dist = abs(my_loc[0] - 3.5) + abs(my_loc[1] - 3.5)
            score -= center_dist * 0.2
            
            # Optional: Add points if moving closer to the highest probability rat location
            if self.tracker:
                best_cell_idx = int(np.argmax(self.tracker.belief))
                ry, rx = divmod(best_cell_idx, 8)
                rat_dist = abs(my_loc[0] - rx) + abs(my_loc[1] - ry)
                score -= rat_dist * 0.1
                
        return score