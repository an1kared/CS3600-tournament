from collections.abc import Callable
from typing import List, Set, Tuple
import random

from game import board, move, enums
from game.enums import MoveType, Direction, CARPET_POINTS_TABLE, Cell

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.transition_matrix = transition_matrix

    def commentate(self):
        return ""

    def play(self, board, sensor_data, time_left):
        moves = board.get_valid_moves()
        moves = [m for m in moves if not (m.move_type == MoveType.CARPET and m.roll_length == 1)]
    
        best_move = None
        best_score = -9999

        for m in moves:
            future = board.forecast_move(m)
            if future is None:
                continue
            score = self.minimax(future, depth=2, is_my_turn=False)
            if score > best_score:
                best_score = score
                best_move = m

        return best_move if best_move else random.choice(board.get_valid_moves())

    def minimax(self, board, depth, is_my_turn):
        if depth == 0 or board.is_game_over():
            return self.heuristic(board)

        moves = board.get_valid_moves()
        if not moves:
            return self.heuristic(board)

        if is_my_turn:
            best = -9999
            for m in moves:
                future = board.forecast_move(m)
                if future is None:
                    continue
                future.reverse_perspective()
                score = self.minimax(future, depth - 1, False)
                best = max(best, score)
            return best
        else:
            best = 9999
            for m in moves:
                future = board.forecast_move(m)
                if future is None:
                    continue
                future.reverse_perspective()
                score = self.minimax(future, depth - 1, True)
                best = min(best, score)
            return best

    def heuristic(self, board):
        my_points = board.player_worker.get_points()
        opp_points = board.opponent_worker.get_points()
        my_pos = board.player_worker.get_location()

        my_carpet_options = sum(
            CARPET_POINTS_TABLE[m.roll_length]
            for m in board.get_valid_moves(enemy=False)
            if m.move_type == MoveType.CARPET and m.roll_length >= 2
        )

        primed_adjacent = sum(
            1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
            if board.is_valid_cell((my_pos[0]+dx, my_pos[1]+dy))
            and board.get_cell((my_pos[0]+dx, my_pos[1]+dy)) == Cell.PRIMED
        )

        open_adjacent = sum(
            1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
            if board.is_valid_cell((my_pos[0]+dx, my_pos[1]+dy))
            and board.get_cell((my_pos[0]+dx, my_pos[1]+dy)) == Cell.SPACE
        )

        return (
            (my_points - opp_points) * 100
            + my_carpet_options * 20
            + primed_adjacent * 10
            + open_adjacent * 2
        )

    def open_space_nearby(self, board, m):
        from game.enums import loc_after_direction
        dest = loc_after_direction(board.player_worker.get_location(), m.direction)
        score = 0
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = dest[0]+dx, dest[1]+dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                if board.get_cell((nx, ny)) == Cell.SPACE:
                    score += 1
        return score