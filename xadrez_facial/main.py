"""
main.py  -  Xadrez controlado pelo rosto
=========================================
Projeto da cadeira de IA (Fundamentos de Reconhecimento Facial).

Como jogar:
  - Mova a CABECA (cima/baixo/lados) para mover o cursor amarelo no tabuleiro.
  - Pisque o OLHO ESQUERDO para CLICAR (selecionar uma peca e depois a casa de destino).
  - Pisque o OLHO DIREITO para CANCELAR a selecao.

Teclado (alternativa / backup para a apresentacao):
  - Setas ou W A S D : mover o cursor
  - Enter ou Espaco  : clicar (selecionar / mover)
  - Backspace        : cancelar selecao
  - C : recalibrar a posicao neutra da cabeca
  - 1 : modo Voce (Brancas) x IA
  - 2 : modo Dois jogadores (hot-seat)
  - R : reiniciar a partida
  - ESC ou Q : sair

Execute (dentro da pasta do projeto) com o Python do venv:
  .venv\\Scripts\\python.exe main.py
"""

import time
import sys
import cv2
import chess

from face_controller import FaceController
from chess_game import ChessGame
from ai_opponent import AIOpponent
from board_ui import BoardUI

WINDOW = "Xadrez Facial - Reconhecimento Facial (IA)"

# codigos de tecla das setas (cv2.waitKeyEx no Windows)
K_LEFT, K_UP, K_RIGHT, K_DOWN = 2424832, 2490368, 2555904, 2621440


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    print(__doc__)
    cap = open_camera()
    if cap is None:
        print(">> Webcam nao encontrada. O jogo abre mesmo assim em modo TECLADO.")

    controller = FaceController()
    game = ChessGame()
    ai = AIOpponent(depth=2)
    ui = BoardUI()

    cursor = [4, 1]          # comeca em e2
    mode = "AI"              # "AI" = voce(brancas) x IA ; "2P" = dois jogadores
    ai_deadline = None
    ai_thinking = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(f">> Backend de rosto: {controller.backend}")

    def mode_label():
        return "Modo: Voce (Brancas) x IA" if mode == "AI" else "Modo: 2 jogadores (hot-seat)"

    def human_turn():
        if game.is_over():
            return False
        if mode == "2P":
            return True
        return game.turn == chess.WHITE     # no modo IA, humano joga de Brancas

    def move_cursor(d):
        if d == "UP":
            cursor[1] = min(7, cursor[1] + 1)
        elif d == "DOWN":
            cursor[1] = max(0, cursor[1] - 1)
        elif d == "LEFT":
            cursor[0] = max(0, cursor[0] - 1)
        elif d == "RIGHT":
            cursor[0] = min(7, cursor[0] + 1)

    def do_click():
        if human_turn():
            game.click(cursor[0], cursor[1])

    def do_cancel():
        if game.selected is not None:
            game.clear_selection()

    running = True
    while running:
        frame = None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                frame = None

        state = controller.process(frame)

        # ---- entrada por rosto ----
        if state.get("step"):
            move_cursor(state["step"])
        if state.get("click"):
            do_click()
        if state.get("cancel"):
            do_cancel()

        # ---- jogada da IA ----
        now = time.time()
        if mode == "AI" and not game.is_over() and game.turn == chess.BLACK:
            if ai_deadline is None:
                ai_deadline = now + 0.6      # pequena pausa antes de responder
                ai_thinking = True
            elif now >= ai_deadline:
                mv = ai.choose(game.board)
                if mv is not None:
                    game.push_external(mv)
                ai_deadline = None
                ai_thinking = False
        else:
            ai_deadline = None
            ai_thinking = False

        # ---- desenhar ----
        canvas = ui.render(game, cursor, state, mode_label(), ai_thinking)
        cv2.imshow(WINDOW, canvas)

        # ---- teclado ----
        k = cv2.waitKeyEx(1)
        if k != -1:
            ch = chr(k).lower() if 0 <= k < 128 else None
            if k in (K_LEFT,) or ch == "a":
                move_cursor("LEFT")
            elif k in (K_RIGHT,) or ch == "d":
                move_cursor("RIGHT")
            elif k in (K_UP,) or ch == "w":
                move_cursor("UP")
            elif k in (K_DOWN,) or ch == "s":
                move_cursor("DOWN")
            elif k in (13, 32):              # Enter / Espaco
                do_click()
            elif k == 8:                     # Backspace
                do_cancel()
            elif ch == "c":
                controller.calibrate()
            elif ch == "1":
                mode = "AI"
            elif ch == "2":
                mode = "2P"
            elif ch == "r":
                game.reset()
                cursor[:] = [4, 1]
                ai_deadline, ai_thinking = None, False
            elif ch in ("q",) or k == 27:    # Q / ESC
                running = False

        # janela fechada no [X]
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            running = False

    if cap is not None:
        cap.release()
    controller.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
