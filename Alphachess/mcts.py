import numpy as np
import math
import torch

class MCTSNode:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state # Tablero de chess.Board
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select(self):
        best_child = None
        best_uct = -np.inf
        for child in self.children:
            uct = child.q_value + self.args['pb_c_puct'] * child.prior * \
                  math.sqrt(self.visit_count) / (1 + child.visit_count)
            if uct > best_uct:
                best_uct = uct
                best_child = child
        return best_child

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                # Aquí conviertes el índice de acción a movimiento de ajedrez real
                # Simplificado: solo expandir si es legal
                pass # (Implementar mapeo de acción a chess.Move)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent:
            self.parent.backpropagate(-value) # Turno del oponente