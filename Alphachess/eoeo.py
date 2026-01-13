import torch
import chess
import random
import os
from model import AlphaChessNet
from utils import board_to_tensor

# --- CONFIGURACIÓN DEL TEST ---
TOTAL_GAMES = 50
AI_DEPTH = 2  # Puedes subirlo a 3, pero para 50 partidas D:2 es más rápido
NN_WEIGHT = 60000 

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

def evaluate_board(board, model):
    if board.is_checkmate():
        return 99999 if board.turn == chess.BLACK else -99999

    # 80% Material
    material_score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_VALUES[p.piece_type]
            material_score += v if p.color == chess.WHITE else -v
    
    # 20% Red Neuronal
    with torch.no_grad():
        _, value = model(board_to_tensor(board))
        nn_score = value.item() * NN_WEIGHT

    return material_score + nn_score

def minimax(board, depth, alpha, beta, is_maximizing, model):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, model)

    if is_maximizing:
        max_v = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            v = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_v = max(max_v, v)
            alpha = max(alpha, v)
            if beta <= alpha: break
        return max_v
    else:
        min_v = float('inf')
        for move in board.legal_moves:
            board.push(move)
            v = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_v = min(min_v, v)
            beta = min(beta, v)
            if beta <= alpha: break
        return min_v

def run_benchmark():
    device = torch.device("cpu")
    model = AlphaChessNet().to(device)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    stats = {"wins": 0, "losses": 0, "draws": 0}

    print(f"🚀 Iniciando maratón de {TOTAL_GAMES} partidas (AI vs Random)...")
    
    for i in range(1, TOTAL_GAMES + 1):
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number < 100:
            if board.turn == chess.WHITE:
                # Rival Aleatorio
                move = random.choice(list(board.legal_moves))
                board.push(move)
            else:
                # Nuestra IA (Negras - intenta minimizar el score)
                best_move = None
                min_val = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    val = minimax(board, AI_DEPTH - 1, -float('inf'), float('inf'), True, model)
                    board.pop()
                    if val < min_val:
                        min_val = val
                        best_move = move
                board.push(best_move)

        # Resultados
        res = board.result()
        if res == "0-1":
            stats["wins"] += 1
            print(f"Partida {i}: Ganó la IA ✅")
        elif res == "1-0":
            stats["losses"] += 1
            print(f"Partida {i}: Perdió la IA ❌")
        else:
            stats["draws"] += 1
            print(f"Partida {i}: Tablas 🤝")

    print("\n" + "═"*30)
    print(f"  RESULTADOS FINALES")
    print("═"*30)
    print(f"Victorias IA: {stats['wins']}")
    print(f"Derrotas IA:  {stats['losses']}")
    print(f"Empates:      {stats['draws']}")
    
    win_rate = (stats['wins'] / TOTAL_GAMES) * 100
    print(f"Tasa de éxito: {win_rate:.1f}%")

if __name__ == "__main__":
    run_benchmark()