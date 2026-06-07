"""
board_ui.py
-----------
Desenha TUDO em uma unica janela do OpenCV:
  - o tabuleiro de xadrez (com pecas em Unicode renderizadas via Pillow)
  - destaques: cursor, casa selecionada, lances legais, ultimo lance, xeque
  - um painel lateral com a webcam, status do jogo e indicadores do rosto
    (barras de EAR de cada olho e a direcao da cabeca)

Nao depende de pygame. Usa OpenCV para formas/blends e Pillow para textos
acentuados e os simbolos de xadrez.
"""

import os
import numpy as np
import cv2
import chess
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- layout
SQUARE = 80
BOARD = SQUARE * 8                 # 640
BOARD_X0, BOARD_Y0 = 40, 40
PANEL_X = BOARD_X0 + BOARD + 40    # 720
PANEL_W = 470
W = PANEL_X + PANEL_W              # 1190
H = 770

CAM_X, CAM_Y, CAM_W, CAM_H = PANEL_X + 20, 20, 430, 322

# ---------------------------------------------------------------- cores (BGR)
BG = (28, 26, 24)
LIGHT_SQ = (210, 238, 238)
DARK_SQ = (86, 150, 118)
PANEL_BG = (44, 39, 36)
CURSOR = (0, 215, 255)
SEL = (0, 165, 255)
LAST = (90, 210, 235)
CHECK = (60, 60, 235)
LEGAL = (70, 70, 70)

# ---------------------------------------------------------------- fontes
def _existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

try:
    import matplotlib
    _MPL = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
except Exception:
    _MPL = ""

_WIN = r"C:\Windows\Fonts"
_PIECE_PATH = _existing(os.path.join(_WIN, "seguisym.ttf"),
                        os.path.join(_MPL, "DejaVuSans.ttf"))
_TEXT_PATH = _existing(os.path.join(_WIN, "segoeui.ttf"),
                       os.path.join(_MPL, "DejaVuSans.ttf"))
_BOLD_PATH = _existing(os.path.join(_WIN, "segoeuib.ttf"),
                       os.path.join(_MPL, "DejaVuSans-Bold.ttf"))

GLYPH = {
    chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
    chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟",
}


class BoardUI:
    def __init__(self):
        self._fonts = {}

    def _font(self, path, size):
        key = (path, size)
        if key not in self._fonts:
            try:
                self._fonts[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
            except Exception:
                self._fonts[key] = ImageFont.load_default()
        return self._fonts[key]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rect(file, rank):
        x0 = BOARD_X0 + file * SQUARE
        y0 = BOARD_Y0 + (7 - rank) * SQUARE
        return x0, y0, x0 + SQUARE, y0 + SQUARE

    # ------------------------------------------------------------------ #
    def render(self, game, cursor, state, mode_label, ai_thinking):
        canvas = np.full((H, W, 3), BG, np.uint8)

        # ---- casas base ----
        for rank in range(8):
            for file in range(8):
                x0, y0, x1, y1 = self._rect(file, rank)
                color = DARK_SQ if (file + rank) % 2 == 0 else LIGHT_SQ
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)

        # ---- destaques translucidos (sobre overlay) ----
        overlay = canvas.copy()
        board = game.board
        if game.last_move is not None:
            for sq in (game.last_move.from_square, game.last_move.to_square):
                self._fill_sq(overlay, sq, LAST)
        if game.selected is not None:
            self._fill_sq(overlay, game.selected, SEL)
        chk = game.king_in_check_square()
        if chk is not None:
            self._fill_sq(overlay, chk, CHECK)
        for sq in game.legal_targets:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x0, y0, x1, y1 = self._rect(f, r)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            if board.piece_at(sq) is not None:        # captura -> anel
                cv2.circle(overlay, (cx, cy), SQUARE // 2 - 4, LEGAL, 6)
            else:                                     # casa vazia -> ponto
                cv2.circle(overlay, (cx, cy), 13, LEGAL, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # ---- cursor (borda nitida, por cima) ----
        cf, cr = cursor
        x0, y0, x1, y1 = self._rect(cf, cr)
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), CURSOR, 4)

        # ---- painel lateral ----
        cv2.rectangle(canvas, (PANEL_X, 0), (W, H), PANEL_BG, -1)
        self._draw_camera(canvas, state)
        self._draw_ear_bars(canvas, state)

        # ---- textos e pecas (uma passada de Pillow) ----
        canvas = self._draw_text_layer(canvas, game, mode_label, ai_thinking, state)
        return canvas

    # ------------------------------------------------------------------ #
    def _fill_sq(self, img, sq, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0, x1, y1 = self._rect(f, r)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    def _draw_camera(self, canvas, state):
        frame = state.get("frame")
        if frame is not None:
            cam = cv2.resize(frame, (CAM_W, CAM_H))
            canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cam
        else:
            cv2.rectangle(canvas, (CAM_X, CAM_Y),
                          (CAM_X + CAM_W, CAM_Y + CAM_H), (60, 55, 50), -1)
        cv2.rectangle(canvas, (CAM_X, CAM_Y),
                      (CAM_X + CAM_W, CAM_Y + CAM_H), (90, 90, 100), 2)

    def _draw_ear_bars(self, canvas, state):
        # apenas as barras (em cv2); os rotulos sao desenhados na camada de texto
        for (ear, top) in [(state.get("left_ear", 0.0), 524),
                           (state.get("right_ear", 0.0), 600)]:
            x = PANEL_X + 20
            cv2.rectangle(canvas, (x, top), (x + 360, top + 18), (70, 65, 60), -1)
            t = max(0.0, min(1.0, (ear - 0.12) / (0.33 - 0.12)))
            if ear < 0.21:
                col = (60, 60, 235)         # fechado -> vermelho
            elif ear < 0.27:
                col = (40, 200, 235)        # intermediario -> amarelo
            else:
                col = (90, 210, 90)         # aberto -> verde
            cv2.rectangle(canvas, (x, top), (x + int(360 * t), top + 18), col, -1)
            cv2.rectangle(canvas, (x, top), (x + 360, top + 18), (120, 120, 130), 1)

    # ------------------------------------------------------------------ #
    def _draw_text_layer(self, canvas, game, mode_label, ai_thinking, state):
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)

        f_piece = self._font(_PIECE_PATH, 60)
        f_title = self._font(_BOLD_PATH, 28)
        f_status = self._font(_BOLD_PATH, 23)
        f_text = self._font(_TEXT_PATH, 19)
        f_small = self._font(_TEXT_PATH, 15)
        f_lab = self._font(_TEXT_PATH, 16)

        # rotulos do tabuleiro (a-h, 1-8)
        for i in range(8):
            file_ch = chr(ord("a") + i)
            x = BOARD_X0 + i * SQUARE + SQUARE // 2
            d.text((x, BOARD_Y0 + BOARD + 14), file_ch, font=f_lab,
                   fill=(170, 170, 180), anchor="mm")
            rank_ch = str(8 - i)
            y = BOARD_Y0 + i * SQUARE + SQUARE // 2
            d.text((BOARD_X0 - 18, y), rank_ch, font=f_lab,
                   fill=(170, 170, 180), anchor="mm")

        # pecas
        board = game.board
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None:
                continue
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x0, y0, x1, y1 = self._rect(f, r)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2 - 4
            glyph = GLYPH[piece.piece_type]
            if piece.color == chess.WHITE:
                fill, stroke = (248, 248, 248), (25, 25, 25)
            else:
                fill, stroke = (35, 35, 38), (215, 215, 215)
            d.text((cx, cy), glyph, font=f_piece, fill=fill,
                   stroke_width=2, stroke_fill=stroke, anchor="mm")

        # ---- painel: textos ----
        px = PANEL_X + 20
        d.text((px, 360), "Xadrez Facial", font=f_title, fill=(255, 255, 255), anchor="lm")
        d.text((px, 392), mode_label, font=f_text, fill=(170, 200, 255), anchor="lm")

        # status do jogo
        status = game.status
        scol = (235, 235, 240)
        low = status.lower()
        if "mate" in low or "xeque" in low:
            scol = (255, 120, 90)
        elif "empate" in low:
            scol = (255, 210, 120)
        if ai_thinking:
            status = "IA pensando..."
            scol = (255, 210, 120)
        d.text((px, 424), status, font=f_status, fill=scol, anchor="lm")

        if game.last_san:
            d.text((px, 456), f"Ultimo lance: {game.last_san}",
                   font=f_text, fill=(190, 190, 200), anchor="lm")

        # rotulos dos olhos (alinhados com as barras em cv2)
        d.text((px, 512), "Olho ESQUERDO  ->  clique", font=f_text,
               fill=(200, 230, 200), anchor="lm")
        if state.get("left_ear", 0.0) < 0.21 and state.get("found"):
            d.text((px + 372, 512), "PISCOU", font=f_small, fill=(255, 120, 90), anchor="lm")
        d.text((px, 588), "Olho DIREITO   ->  cancelar", font=f_text,
               fill=(230, 210, 200), anchor="lm")
        if state.get("right_ear", 0.0) < 0.21 and state.get("found"):
            d.text((px + 372, 588), "PISCOU", font=f_small, fill=(255, 120, 90), anchor="lm")

        # estado do rosto / calibracao
        if not state.get("found"):
            d.text((px, 648), "Rosto nao detectado", font=f_text, fill=(255, 130, 110), anchor="lm")
        elif not state.get("calibrated"):
            d.text((px, 648), "Calibrando... olhe para a tela", font=f_text, fill=(255, 210, 120), anchor="lm")
        else:
            d.text((px, 648), f"Direcao da cabeca: {state.get('hold_dir') or '-'}",
                   font=f_text, fill=(180, 210, 255), anchor="lm")
        d.text((px, 672), f"Backend: {state.get('backend', '-')}",
               font=f_small, fill=(140, 140, 150), anchor="lm")

        # ajuda (rodape)
        help_lines = [
            "Incline a cabeca e VOLTE ao centro = 1 passo (segure = repete)",
            "Olho esq = clique | olho dir = cancelar | C = recalibrar centro",
            "Teclado: setas/WASD, Enter, Backspace | 1 vs IA  2 2P  R  ESC",
        ]
        for i, line in enumerate(help_lines):
            d.text((px, 706 + i * 20), line, font=f_small, fill=(150, 150, 160), anchor="lm")

        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
