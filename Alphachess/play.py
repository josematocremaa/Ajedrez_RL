import torch
import chess
from model import AlphaChessNet
from utils import board_to_tensor, move_to_idx

# --- CONFIGURACIÓN ---
DEPTH = 4 
# Un peón vale 100. La red (que va de -1 a 1) multiplicada por 600
# significa que la "intuición" puede valer hasta 6 peones de diferencia.
NN_WEIGHT = 600 

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

def evaluate_board(board, model):
    """Evaluación: 80% Material / 20% Red Neuronal."""
    if board.is_checkmate():
        if board.turn == chess.WHITE: return -99999
        else: return 99999

    # 1. MATERIAL (El 80% del peso)
    material_score = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v = PIECE_VALUES[p.piece_type]
            material_score += v if p.color == chess.WHITE else -v
    
    # 2. RED NEURONAL (El 20% del peso / Posicional)
    with torch.no_grad():
        _, value = model(board_to_tensor(board))
        # Ajustamos: si value.item() es 1.0 (blancas ganan), suma 600.
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

def play():
    device = torch.device("cpu")
    model = AlphaChessNet().to(device)
    
    if not os.path.exists("best_model.pth"):
        print("Error: No se encuentra best_model.pth. Entrena la IA primero.")
        return
        
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    
    board = chess.Board()
    print("\n" + "═"*40)
    print("      IA DÍA 5 - BALANCE 80/20")
    print("  (Material Sólido + Intuición de Red)")
    print("═"*40)
    
    while not board.is_game_over():
        print(f"\n{board}\n")
        if board.turn == chess.WHITE:
            move_str = input("Tu movimiento (UCI): ")
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    board.push(move)
                else: print("❌ Movimiento ilegal.")
            except: print("⚠️ Formato incorrecto (usa ej: e2e4).")
        else:
            print(f"🤖 IA Pensando (Profundidad {DEPTH})...")
            best_move = None
            min_val = float('inf')
            
            # Ordenamos para optimizar la búsqueda
            legal_moves = list(board.legal_moves)
            
            for move in legal_moves:
                board.push(move)
                val = minimax(board, DEPTH - 1, -float('inf'), float('inf'), True, model)
                board.pop()
                if val < min_val:
                    min_val = val
                    best_move = move
            
            print(f"🎯 IA elige: {best_move} | Eval: {min_val:.2f}")
            board.push(best_move)

    print(f"\nFIN DEL JUEGO: {board.result()}")

if __name__ == "__main__":
    import os
    play()