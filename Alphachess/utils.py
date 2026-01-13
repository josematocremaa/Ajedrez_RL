import chess
import numpy as np
import torch

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
                 chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
    for color in [chess.WHITE, chess.BLACK]:
        offset = 0 if color == chess.WHITE else 6
        for piece_type, layer in piece_map.items():
            for sq in board.pieces(piece_type, color):
                row, col = divmod(sq, 8)
                tensor[layer + offset, row, col] = 1
    return torch.from_numpy(tensor).unsqueeze(0)

def move_to_idx(move):
    return move.from_square * 64 + move.to_square

def get_reward(board):
    # Valores: P=1, N=3, B=3, R=5, Q=9
    vals = {1:1, 2:3, 3:3, 4:5, 5:9, 6:0}
    score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            val = vals[p.piece_type]
            score += val if p.color == chess.WHITE else -val
    return score / 50.0 # Normalizado