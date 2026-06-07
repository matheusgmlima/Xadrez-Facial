"""Teste da logica de movimento da cabeca (sem camera)."""
import time
from face_controller import FaceController

fc = FaceController()

# calibra no centro (neutro = 0,0)
for _ in range(fc._calib_target):
    fc._update_logic(0.0, 0.0, 0.30, 0.30)
assert fc.calibrated, "deveria calibrar"

def feed(ox, oy, n=1):
    last = None
    for _ in range(n):
        last = fc._update_logic(ox, oy, 0.30, 0.30)
    return last

print("1) re-arma no centro")
feed(0.0, 0.0, 3)
assert fc._armed is True

print("2) flick: inclina forte -> 1 passo, segurando nao repete na hora")
s = fc._update_logic(-0.25, 0.0, 0.30, 0.30)   # off_x<0 => RIGHT (espelho)
assert s["step"] == "RIGHT", s["step"]
s2 = fc._update_logic(-0.25, 0.0, 0.30, 0.30)
assert s2["step"] is None, "nao deve repetir antes do hold_delay"

print("3) volta ao centro re-arma; inclina p/ cima dispara UP")
feed(0.0, 0.0, 6)
assert fc._armed is True
s3 = fc._update_logic(0.0, -0.25, 0.30, 0.30)   # off_y<0 => UP
assert s3["step"] == "UP", s3["step"]

print("4) ANTI-RUNAWAY: segurando 3s, numero de passos e limitado (sem infinito)")
feed(0.0, 0.0, 6)            # re-arma
count = 0
t0 = time.time()
while time.time() - t0 < 3.0:
    s = fc._update_logic(0.0, 0.30, 0.30, 0.30)   # DOWN, segurando
    if s["step"]:
        count += 1
limite = 1 + fc.max_repeats
assert count <= limite, f"passos={count} > limite={limite} (RUNAWAY!)"
print(f"   passos em 3s segurando = {count} (limite {limite}) OK")

print("5) wink esquerdo = clique")
feed(0.0, 0.0, 3)
s = fc._update_logic(0.0, 0.0, 0.10, 0.32)   # olho esq fechado, dir aberto
assert s["click"] is True, "deveria registrar clique"

fc.close()
print("\nMOVIMENTO OK - sem runaway")
