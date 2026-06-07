"""
ai_opponent.py
--------------
Adversário simples para jogar de Pretas (ou Brancas) contra o jogador.

Implementa uma busca negamax com poda alfa-beta e uma avaliação baseada em
material + tabelas posicionais (piece-square tables). A profundidade padrão
(depth=2) deixa o oponente rápido e divertido para uma demonstração, sem
travar a interface. Aumente `depth` para um jogo mais forte (e mais lento).
"""

import random
import chess

# Valor base de cada peça (em centésimos de peão)
VALORES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Tabelas posicionais (ponto de vista das Brancas; espelhadas para as Pretas).
PEAO = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
CAVALO = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
BISPO = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
TORRE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]
DAMA = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
REI = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
TABELAS = {
    chess.PAWN: PEAO, chess.KNIGHT: CAVALO, chess.BISHOP: BISPO,
    chess.ROOK: TORRE, chess.QUEEN: DAMA, chess.KING: REI,
}


def avaliar(board):
    """Avaliacao estatica do tabuleiro do ponto de vista de quem tem a vez."""
    if board.is_checkmate():
        return -99999  # quem deve jogar esta em mate -> pessimo
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for sq, piece in board.piece_map().items():
        valor = VALORES[piece.piece_type]
        tabela = TABELAS[piece.piece_type]
        # Para as Pretas, espelhamos a casa verticalmente.
        idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        pos = tabela[idx]
        if piece.color == chess.WHITE:
            score += valor + pos
        else:
            score -= valor + pos

    # Pontuacao positiva = bom para as Brancas. Ajusta para quem tem a vez.
    return score if board.turn == chess.WHITE else -score


def _negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return avaliar(board)

    melhor = -10**9
    # Ordena: capturas primeiro melhora a poda alfa-beta.
    moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)
    for move in moves:
        board.push(move)
        valor = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        if valor > melhor:
            melhor = valor
        if melhor > alpha:
            alpha = melhor
        if alpha >= beta:
            break  # poda
    return melhor


class AIOpponent:
    def __init__(self, depth=2):
        self.depth = depth

    def choose(self, board):
        """Escolhe o melhor lance para a cor que tem a vez."""
        melhor_valor = -10**9
        melhores = []
        moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)
        for move in moves:
            board.push(move)
            valor = -_negamax(board, self.depth - 1, -10**9, 10**9)
            board.pop()
            if valor > melhor_valor:
                melhor_valor = valor
                melhores = [move]
            elif valor == melhor_valor:
                melhores.append(move)
        if not melhores:
            return None
        return random.choice(melhores)  # desempate aleatorio = mais variado
