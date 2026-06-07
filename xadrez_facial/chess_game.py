"""
chess_game.py
-------------
Camada de regras do xadrez. Encapsula a biblioteca python-chess (instalada como
pacote `chess`) e expõe uma interface simples baseada em "cursor + clique" para
ser controlada pelo rosto:

    - selecionar uma casa (primeiro clique pega a peça)
    - clicar em outra casa (segundo clique tenta mover)
    - clicar de novo na mesma casa cancela a seleção

Todas as regras do xadrez (roque, en passant, promoção, xeque, xeque-mate,
empate) são tratadas pelo python-chess.
"""

import chess


class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.selected = None          # casa selecionada (0..63) ou None
        self.legal_targets = set()    # casas de destino legais a partir da seleção
        self.last_move = None         # último lance (chess.Move) para destacar
        self.status = "Vez das Brancas"
        self.last_san = None          # notação algébrica do último lance

    # ------------------------------------------------------------------ #
    # Consultas auxiliares
    # ------------------------------------------------------------------ #
    @staticmethod
    def square(file, rank):
        """file 0..7 (a..h), rank 0..7 (1..8) -> índice de casa 0..63."""
        return chess.square(file, rank)

    def piece_at(self, file, rank):
        return self.board.piece_at(chess.square(file, rank))

    @property
    def turn(self):
        return self.board.turn  # chess.WHITE (True) ou chess.BLACK (False)

    def is_over(self):
        return self.board.is_game_over()

    def king_in_check_square(self):
        """Retorna a casa do rei em xeque (para destacar), ou None."""
        if self.board.is_check():
            return self.board.king(self.board.turn)
        return None

    # ------------------------------------------------------------------ #
    # Interação por cursor/clique
    # ------------------------------------------------------------------ #
    def click(self, file, rank):
        """
        Processa um "clique" (piscada) na casa (file, rank).
        Retorna uma string com o resultado:
        "selecionou" | "moveu" | "cancelou" | "trocou" | "invalido" | "ilegal"
        """
        sq = chess.square(file, rank)
        piece = self.board.piece_at(sq)

        # Nada selecionado ainda -> tentar selecionar uma peça da vez
        if self.selected is None:
            if piece is not None and piece.color == self.board.turn:
                self.selected = sq
                self._compute_targets()
                return "selecionou"
            return "invalido"

        # Clique na própria casa selecionada -> cancela
        if sq == self.selected:
            self.clear_selection()
            return "cancelou"

        # Tentar mover da casa selecionada para a casa clicada
        move = self._build_move(self.selected, sq)
        if move in self.board.legal_moves:
            self.last_san = self.board.san(move)
            self.board.push(move)
            self.last_move = move
            self.clear_selection()
            self._update_status()
            return "moveu"

        # Clique em outra peça própria -> troca a seleção
        if piece is not None and piece.color == self.board.turn:
            self.selected = sq
            self._compute_targets()
            return "trocou"

        # Movimento ilegal
        return "ilegal"

    def push_external(self, move):
        """Aplica um lance já escolhido externamente (ex.: a IA)."""
        self.last_san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.clear_selection()
        self._update_status()

    def clear_selection(self):
        self.selected = None
        self.legal_targets = set()

    def reset(self):
        self.board.reset()
        self.clear_selection()
        self.last_move = None
        self.last_san = None
        self.status = "Vez das Brancas"

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _build_move(self, frm, to):
        """Monta o lance, promovendo peão automaticamente para dama."""
        piece = self.board.piece_at(frm)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN:
            if chess.square_rank(to) in (0, 7):
                promotion = chess.QUEEN
        return chess.Move(frm, to, promotion=promotion)

    def _compute_targets(self):
        self.legal_targets = {
            m.to_square for m in self.board.legal_moves
            if m.from_square == self.selected
        }

    def _update_status(self):
        if self.board.is_checkmate():
            vencedor = "Brancas" if self.board.turn == chess.BLACK else "Pretas"
            self.status = f"Xeque-mate! {vencedor} vencem"
        elif self.board.is_stalemate():
            self.status = "Empate por afogamento (stalemate)"
        elif self.board.is_insufficient_material():
            self.status = "Empate por material insuficiente"
        elif self.board.can_claim_threefold_repetition():
            self.status = "Empate por repeticao"
        elif self.board.is_check():
            vez = "Brancas" if self.board.turn == chess.WHITE else "Pretas"
            self.status = f"Xeque! Vez das {vez}"
        else:
            vez = "Brancas" if self.board.turn == chess.WHITE else "Pretas"
            self.status = f"Vez das {vez}"
