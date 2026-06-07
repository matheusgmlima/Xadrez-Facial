"""Teste rapido sem camera: valida xadrez, IA e renderizacao."""
import numpy as np
import cv2
from chess_game import ChessGame
from ai_opponent import AIOpponent
from board_ui import BoardUI
from face_controller import FaceController

print("1) Logica do xadrez")
g = ChessGame()
assert g.click(4, 1) == "selecionou", "deveria selecionar o peao e2"
assert g.click(4, 3) == "moveu", "deveria mover e2-e4"
assert g.last_san == "e4", g.last_san
print("   e2-e4 OK | status:", g.status)

print("2) IA escolhe lance para as Pretas")
mv = AIOpponent(depth=2).choose(g.board)
assert mv is not None, "IA nao retornou lance"
print("   IA jogaria:", g.board.san(mv))

print("3) FaceController + frame preto (sem rosto)")
fc = FaceController()
st = fc.process(np.zeros((480, 640, 3), np.uint8))
assert st["frame"] is not None and st["found"] is False
print("   backend:", st["backend"], "| found:", st["found"])

print("4) Renderizacao")
ui = BoardUI()
g.select if False else None
g.clear_selection()
g.click(4, 6)  # selecionar peao preto e7 (vez das pretas)
img = ui.render(g, [4, 4], st, "Modo: Voce (Brancas) x IA", False)
print("   imagem:", img.shape, img.dtype)
cv2.imwrite("preview.png", img)
print("   salvo preview.png")

fc.close()
print("\nTUDO OK")
