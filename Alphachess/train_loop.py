import torch
import torch.optim as optim
import chess
import pickle
import os
import random
from collections import deque
from model import AlphaChessNet
from utils import board_to_tensor, move_to_idx, get_reward

# --- CONFIGURACIÓN DE ÉLITE (DÍA 5) ---
GAMES_PER_ITER = 2      # Calidad sobre cantidad (2 partidas meditadas)
TRAIN_DEPTH = 3         # El 'Maestro' interno calcula a prof 3
LEARNING_RATE = 0.0002  # Aprendizaje lento y estable
MEMORY_SIZE = 20000     # Estados en RAM
BATCH_SIZE = 64
MODEL_PATH = "best_model.pth"

def minimax_teacher(model, board, depth, alpha, beta, is_maximizing):
    """Motor Minimax que guía a la Red Neuronal con lógica material."""
    if depth == 0 or board.is_game_over():
        with torch.no_grad():
            _, value = model(board_to_tensor(board))
            # Combinamos Red (-1 a 1) con Material (puntos reales)
            # Multiplicamos el valor de la red para que tenga peso estratégico
            return (value.item() * 5.0) + get_reward(board)

    legal_moves = list(board.legal_moves)
    # Ordenar por instinto de red para podar rápido
    if is_maximizing:
        max_v = -float('inf')
        for move in legal_moves[:8]: # Beam search: solo los 8 mejores
            board.push(move)
            v = minimax_teacher(model, board, depth - 1, alpha, beta, False)
            board.pop()
            max_v = max(max_v, v)
            alpha = max(alpha, v)
            if beta <= alpha: break
        return max_v
    else:
        min_v = float('inf')
        for move in legal_moves[:8]:
            board.push(move)
            v = minimax_teacher(model, board, depth - 1, alpha, beta, True)
            board.pop()
            min_v = min(min_v, v)
            beta = min(beta, v)
            if beta <= alpha: break
        return min_v

def get_master_move(model, board):
    """Usa el pensamiento profundo para decidir qué jugada guardar en memoria."""
    best_move = None
    max_val = -float('inf')
    
    with torch.no_grad():
        policy, _ = model(board_to_tensor(board))
    
    # Seleccionamos los candidatos basados en la intuición de la red
    moves = sorted(list(board.legal_moves), key=lambda m: policy[0][move_to_idx(m)].item(), reverse=True)

    for move in moves[:10]:
        board.push(move)
        # Evaluamos la respuesta del oponente
        val = -minimax_teacher(model, board, TRAIN_DEPTH - 1, -float('inf'), float('inf'), False)
        board.pop()
        if val > max_val:
            max_val = val
            best_move = move
    return best_move

def train():
    device = torch.device("cpu")
    model = AlphaChessNet().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Cerebro previo cargado. Iniciando entrenamiento con guía material...")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)

    iteration = 1
    while True:
        model.eval()
        for g in range(GAMES_PER_ITER):
            board = chess.Board()
            game_states = []
            print(f"Iter {iteration} | Partida {g+1}...", end=" ", flush=True)
            
            while not board.is_game_over() and board.fullmove_number < 65:
                # La IA 'piensa' antes de actuar
                move = get_master_move(model, board)
                
                # Ruido de exploración (10%) para descubrir nuevas jugadas
                if random.random() < 0.1: move = random.choice(list(board.legal_moves))
                
                game_states.append((board_to_tensor(board), move_to_idx(move)))
                board.push(move)
            
            # Resultado final
            res = 1 if board.result() == "1-0" else (-1 if board.result() == "0-1" else 0)
            for s, m_idx in game_states:
                memory.append((s, m_idx, res))
            print(f"OK ({board.fullmove_number} jugadas)")

        # Actualización de la Red Neuronal
        model.train()
        if len(memory) > BATCH_SIZE:
            batch = random.sample(list(memory), BATCH_SIZE)
            states = torch.cat([b[0] for b in batch])
            targets_p = torch.tensor([b[1] for b in batch])
            targets_v = torch.tensor([b[2] for b in batch], dtype=torch.float32)

            optimizer.zero_grad()
            p_out, v_out = model(states)
            loss = torch.nn.functional.nll_loss(p_out, targets_p) + torch.nn.functional.mse_loss(v_out.view(-1), targets_v)
            loss.backward()
            optimizer.step()
            print(f"   [Loss: {loss.item():.4f}]")

        if iteration % 10 == 0:
            torch.save(model.state_dict(), MODEL_PATH)
        iteration += 1

if __name__ == "__main__":
    train()