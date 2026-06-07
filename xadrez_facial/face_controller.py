"""
face_controller.py
------------------
Controle por rosto usando os FUNDAMENTOS de reconhecimento facial:

  1. DETECCAO + LANDMARKS  -> MediaPipe Face Mesh (468 pontos faciais)
  2. ESTIMATIVA DE POSE     -> posicao do nariz em relacao aos olhos = para onde
                               a cabeca aponta (move o cursor no tabuleiro)
  3. EAR (Eye Aspect Ratio) -> detecta piscada/wink de cada olho separadamente:
        - piscar o olho ESQUERDO  = "clique" (selecionar / mover)
        - piscar o olho DIREITO   = "cancelar" (soltar a peca)

Se o MediaPipe nao estiver disponivel, cai para um backend simples em OpenCV
(Haar Cascade) que detecta rosto e olhos. O MediaPipe e bem mais preciso.

Convencoes:
  - O frame e processado SEM espelhar (assim "olho esquerdo" = olho esquerdo
    real da pessoa). A imagem exibida e espelhada (efeito espelho), mais natural.
  - direcao da cabeca para a direita -> cursor para a direita, etc.
"""

import os
import time
import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MP = True
except Exception:                      # pragma: no cover
    _HAS_MP = False

# Modelo do MediaPipe Tasks (baixado em models/face_landmarker.task)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "face_landmarker.task")


# Indices dos landmarks do MediaPipe Face Mesh
NOSE_TIP = 1
# Olho direito (lado direito da pessoa): cantos + pontos sup/inf
R_EYE = [33, 160, 158, 133, 153, 144]
# Olho esquerdo (lado esquerdo da pessoa)
L_EYE = [362, 385, 387, 263, 373, 380]


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _ear(pts):
    """Eye Aspect Ratio a partir dos 6 pontos [p1..p6] (em pixels)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class FaceController:
    def __init__(self,
                 ear_closed=0.21,      # abaixo disso o olho conta como fechado
                 ear_open=0.27,        # acima disso o olho conta como aberto
                 wink_diff=0.06,       # assimetria minima p/ distinguir wink de piscada
                 deadzone=0.04,        # zona "centro": dentro dela o gesto re-arma
                 trigger=0.085,        # passar disso dispara um passo (gatilho)
                 hold_delay=0.6,       # tempo segurando antes de comecar a auto-repetir
                 repeat_rate=0.45,     # intervalo entre passos na auto-repeticao
                 max_repeats=7,        # limite de passos seguidos sem voltar ao centro
                 smooth=0.4,           # suavizacao (0=sem, ~1=travado)
                 stable_band=0.03,     # variacao max. p/ considerar a cabeca "parada"
                 stable_time=0.8,      # tempo parado p/ re-centrar automaticamente
                 idle_recenter=1.0):   # so re-centra se nao houve passo recente
        self.ear_closed = ear_closed
        self.ear_open = ear_open
        self.wink_diff = wink_diff
        self.deadzone = deadzone
        self.trigger = max(trigger, deadzone + 0.01)
        self.hold_delay = hold_delay
        self.repeat_rate = repeat_rate
        self.max_repeats = max_repeats
        self.smooth = smooth
        self.stable_band = stable_band
        self.stable_time = stable_time
        self.idle_recenter = idle_recenter

        # Calibracao (posicao neutra da cabeca)
        self.calibrated = False
        self._neutral = None
        self._calib_samples = []
        self._calib_target = 5

        # Offset suavizado (relativo ao neutro)
        self._sx = 0.0
        self._sy = 0.0
        self._dx = 0.0          # ultimos valores (p/ HUD)
        self._dy = 0.0

        # Estado de wink (histerese por olho)
        self._left_closed = False
        self._right_closed = False

        # Controle de movimento (gesto "inclina-e-volta" + auto-repeticao)
        self._armed = False      # True quando a cabeca passou pelo centro
        self._active_dir = None  # direcao da auto-repeticao em andamento
        self._next_repeat = 0.0
        self._repeat_count = 0
        self._last_step_time = 0.0

        # Deteccao de "cabeca parada" p/ re-centragem automatica
        self._stable_ref = None
        self._stable_since = 0.0

        self.backend = "nenhum"
        self._landmarker = None
        self._ts = 0
        self._face_cascade = None
        self._eye_cascade = None

        if _HAS_MP and os.path.exists(_MODEL_PATH):
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.backend = "MediaPipe Face Landmarker"
        else:                                                  # fallback
            base = cv2.data.haarcascades
            self._face_cascade = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            self._eye_cascade = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
            self.backend = "OpenCV Haar (fallback)"

    # ------------------------------------------------------------------ #
    def calibrate(self):
        """Recalibra: o 'centro' passa a ser a pose atual da cabeca."""
        self.calibrated = False
        self._neutral = None
        self._calib_samples = []
        self._armed = False
        self._active_dir = None
        self._repeat_count = 0
        self._stable_ref = None

    # ------------------------------------------------------------------ #
    def process(self, frame):
        """
        Recebe um frame BGR da webcam e retorna um dicionario de estado +
        o frame anotado (espelhado, pronto para exibir).
        """
        if frame is None:
            return self._empty_state(None)

        if self.backend.startswith("MediaPipe"):
            return self._process_mediapipe(frame)
        return self._process_haar(frame)

    # ------------------------------------------------------------------ #
    def _process_mediapipe(self, frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 1
        result = self._landmarker.detect_for_video(mp_image, self._ts)
        disp = cv2.flip(frame, 1)  # imagem espelhada para exibir

        if not result.face_landmarks:
            return self._empty_state(disp)

        lm = result.face_landmarks[0]

        def px(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        nose = px(NOSE_TIP)
        r_pts = [px(i) for i in R_EYE]
        l_pts = [px(i) for i in L_EYE]
        right_ear = _ear(r_pts)
        left_ear = _ear(l_pts)

        # Pose da cabeca: nariz em relacao ao ponto medio entre os olhos,
        # normalizado pela largura do rosto (robusto a distancia da camera).
        r_outer, l_outer = px(33), px(263)
        eye_mid = (r_outer + l_outer) / 2.0
        face_w = _dist(r_outer, l_outer) or 1.0
        off_x = (nose[0] - eye_mid[0]) / face_w
        off_y = (nose[1] - eye_mid[1]) / face_w

        state = self._update_logic(off_x, off_y, left_ear, right_ear)

        # ---- desenhar overlays na imagem espelhada ----
        def fx(p):  # converte ponto (frame original) para a imagem espelhada
            return (int(w - p[0]), int(p[1]))

        l_closed = state["left_ear"] < self.ear_closed
        r_closed = state["right_ear"] < self.ear_closed
        cv2.polylines(disp, [np.array([fx(p) for p in r_pts], np.int32)],
                      True, (0, 0, 255) if r_closed else (0, 220, 0), 2)
        cv2.polylines(disp, [np.array([fx(p) for p in l_pts], np.int32)],
                      True, (0, 0, 255) if l_closed else (0, 220, 0), 2)
        cv2.circle(disp, fx(nose), 4, (255, 200, 0), -1)
        cv2.circle(disp, fx(eye_mid), 3, (255, 120, 0), -1)

        # ---- HUD: posicao da cabeca x zona morta (verde) e gatilho (azul) ----
        cxh, cyh, R = 92, 96, 70
        ddx, ddy = state.get("dx", 0.0), state.get("dy", 0.0)
        dz = int(self.deadzone / self.trigger * R * 0.82)
        tg = int(R * 0.82)
        cv2.rectangle(disp, (cxh - R, cyh - R), (cxh + R, cyh + R), (35, 35, 35), -1)
        cv2.rectangle(disp, (cxh - tg, cyh - tg), (cxh + tg, cyh + tg), (200, 120, 60), 1)
        cv2.rectangle(disp, (cxh - dz, cyh - dz), (cxh + dz, cyh + dz), (90, 200, 90), 1)
        # ponto da cabeca (x espelhado p/ bater com a imagem)
        ppx = int(cxh - max(-1.5, min(1.5, ddx / self.trigger)) * R * 0.82)
        ppy = int(cyh + max(-1.5, min(1.5, ddy / self.trigger)) * R * 0.82)
        if abs(ddx) < self.deadzone and abs(ddy) < self.deadzone:
            dot = (90, 230, 90)      # no centro (verde)
        elif abs(ddx) >= self.trigger or abs(ddy) >= self.trigger:
            dot = (60, 60, 240)      # disparando (vermelho)
        else:
            dot = (60, 200, 240)     # zona intermediaria (amarelo)
        cv2.circle(disp, (ppx, ppy), 7, dot, -1)
        cv2.putText(disp, "CENTRO", (cxh - R, cyh + R + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)

        state["frame"] = disp
        return state

    # ------------------------------------------------------------------ #
    def _process_haar(self, frame):
        """Fallback simples: posicao do rosto move o cursor; contagem de olhos
        detecta piscada. Menos preciso (sem distinguir wink esquerdo/direito
        com confiabilidade)."""
        h, w = frame.shape[:2]
        disp = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
        if len(faces) == 0:
            return self._empty_state(disp)

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + fw / 2.0, y + fh / 2.0
        # offset normalizado: rosto fora do centro do frame
        off_x = (cx - w / 2.0) / w
        off_y = (cy - h / 2.0) / h

        roi = gray[y:y + fh, x:x + fw]
        eyes = self._eye_cascade.detectMultiScale(roi, 1.1, 6, minSize=(25, 25))
        # menos de 2 olhos detectados -> consideramos "olho fechado"
        ear_proxy = 0.30 if len(eyes) >= 2 else 0.10
        state = self._update_logic(off_x, off_y, ear_proxy, 0.30)

        fx2 = int(w - (x + fw))
        cv2.rectangle(disp, (fx2, y), (fx2 + fw, y + fh), (0, 220, 0), 2)
        state["frame"] = disp
        return state

    # ------------------------------------------------------------------ #
    def _update_logic(self, off_x, off_y, left_ear, right_ear):
        now = time.time()

        # ---- calibracao inicial (media de poucos frames) ----
        if not self.calibrated:
            self._calib_samples.append((off_x, off_y))
            if len(self._calib_samples) >= self._calib_target:
                arr = np.array(self._calib_samples)
                self._neutral = [float(arr[:, 0].mean()), float(arr[:, 1].mean())]
                self.calibrated = True
                self._sx = self._sy = 0.0
                self._armed = False
                self._active_dir = None
                self._repeat_count = 0
                self._stable_ref = (off_x, off_y)
                self._stable_since = now
                self._last_step_time = now
            return {
                "found": True, "step": None, "click": False, "cancel": False,
                "left_ear": left_ear, "right_ear": right_ear, "hold_dir": None,
                "dx": 0.0, "dy": 0.0, "armed": False,
                "calibrated": False, "backend": self.backend, "frame": None,
            }

        # ---- offset relativo ao neutro + suavizacao ----
        a = self.smooth
        self._sx = a * self._sx + (1 - a) * (off_x - self._neutral[0])
        self._sy = a * self._sy + (1 - a) * (off_y - self._neutral[1])
        dx, dy = self._sx, self._sy

        # ---- re-centragem automatica: cabeca parada por um tempo vira o centro ----
        # (conserta calibracao ruim e mudanca de postura; so age quando NAO ha
        #  movimento acontecendo, para nao atrapalhar uma inclinacao intencional)
        if self._stable_ref is None:
            self._stable_ref = (off_x, off_y)
            self._stable_since = now
        if abs(off_x - self._stable_ref[0]) + abs(off_y - self._stable_ref[1]) > self.stable_band:
            self._stable_ref = (off_x, off_y)     # mexeu -> reinicia o cronometro
            self._stable_since = now
        idle = (now - self._last_step_time) > self.idle_recenter
        if idle and (now - self._stable_since) > self.stable_time:
            self._neutral = [off_x, off_y]        # este ponto passa a ser o centro
            self._sx = self._sy = 0.0
            dx = dy = 0.0
            self._armed = True
            self._active_dir = None
            self._repeat_count = 0
            self._stable_since = now              # evita re-centrar todo frame

        # ---- direcao candidata (so se passar do gatilho); horizontal invertido ----
        cand = None
        if abs(dx) >= self.trigger or abs(dy) >= self.trigger:
            if abs(dx) >= abs(dy):
                cand = "RIGHT" if dx < 0 else "LEFT"
            else:
                cand = "DOWN" if dy > 0 else "UP"

        inside = abs(dx) < self.deadzone and abs(dy) < self.deadzone

        # ---- maquina de estados: "inclina-e-volta" + auto-repeticao limitada ----
        step = None
        if inside:
            self._armed = True            # voltou ao centro -> pronto p/ novo gesto
            self._active_dir = None
            self._repeat_count = 0
        elif cand is not None:
            if self._armed:               # gesto novo: 1 passo imediato
                step = cand
                self._armed = False
                self._active_dir = cand
                self._repeat_count = 0
                self._next_repeat = now + self.hold_delay
            elif cand == self._active_dir and now >= self._next_repeat:
                if self._repeat_count < self.max_repeats:
                    step = cand           # segurou na mesma direcao -> auto-repete
                    self._repeat_count += 1
                    self._next_repeat = now + self.repeat_rate
                else:
                    self._active_dir = None   # atingiu o limite -> trava ate centrar
            elif cand != self._active_dir:
                self._active_dir = None   # trocou de direcao sem centrar -> trava
        # zona intermediaria (entre deadzone e trigger): nao faz nada

        if step is not None:
            self._last_step_time = now
        self._dx, self._dy = dx, dy

        # ---- winks (clique = olho esquerdo, cancelar = olho direito) ----
        click = cancel = False
        # esquerdo
        if not self._left_closed:
            if left_ear < self.ear_closed and (right_ear - left_ear) > self.wink_diff:
                self._left_closed = True
                click = True
        elif left_ear > self.ear_open:
            self._left_closed = False
        # direito
        if not self._right_closed:
            if right_ear < self.ear_closed and (left_ear - right_ear) > self.wink_diff:
                self._right_closed = True
                cancel = True
        elif right_ear > self.ear_open:
            self._right_closed = False

        return {
            "found": True, "step": step, "click": click, "cancel": cancel,
            "left_ear": left_ear, "right_ear": right_ear, "hold_dir": cand,
            "dx": dx, "dy": dy, "armed": self._armed,
            "calibrated": True, "backend": self.backend, "frame": None,
        }

    # ------------------------------------------------------------------ #
    def _empty_state(self, frame):
        # rosto perdido: zera estados de movimento para nao "grudar"
        self._armed = False
        self._active_dir = None
        self._repeat_count = 0
        return {
            "found": False, "step": None, "click": False, "cancel": False,
            "left_ear": 0.0, "right_ear": 0.0, "hold_dir": None,
            "dx": 0.0, "dy": 0.0, "armed": False,
            "calibrated": self.calibrated, "backend": self.backend,
            "frame": frame,
        }

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
