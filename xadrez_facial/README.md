# ♟️ Xadrez Facial — Xadrez controlado pelo rosto

Projeto da cadeira de **Inteligência Artificial** (seminário de *Fundamentos de
Reconhecimento Facial*). Você joga xadrez **sem mouse e sem teclado**: move a
cabeça para deslocar o cursor pelo tabuleiro e **pisca os olhos** para clicar.

É também uma demonstração prática dos fundamentos apresentados no seminário:
**detecção facial → landmarks → estimativa de pose → interpretação de gestos**.

---

## 🎯 Como funciona (ligação com o seminário)

| Etapa do reconhecimento facial | Onde aparece no projeto |
|---|---|
| **Detecção facial** | O MediaPipe localiza o rosto na imagem da webcam |
| **Landmarks (pontos-chave)** | 478 pontos do rosto (olhos, nariz, contorno) |
| **Alinhamento / pose** | Posição do nariz em relação aos olhos = direção da cabeça |
| **Interpretação de gestos** | *Eye Aspect Ratio* (EAR) detecta a piscada de cada olho |

### Controles por rosto
- 🧭 **Mover a cabeça** (cima/baixo/esquerda/direita) → move o **cursor amarelo**.
- 😉 **Piscar o olho ESQUERDO** → **clique** (seleciona a peça; clique de novo na casa de destino para mover).
- 😉 **Piscar o olho DIREITO** → **cancelar** a seleção.

### Controles por teclado (backup para a apresentação)
| Tecla | Ação |
|---|---|
| Setas / `W A S D` | mover o cursor |
| `Enter` ou `Espaço` | clicar (selecionar / mover) |
| `Backspace` | cancelar seleção |
| `C` | recalibrar a posição neutra da cabeça |
| `1` | modo **Você (Brancas) × IA** |
| `2` | modo **2 jogadores** (hot-seat) |
| `R` | reiniciar a partida |
| `ESC` ou `Q` | sair |

---

## 🚀 Como rodar

### Jeito fácil (Windows)
Dê **dois cliques em `run.bat`**. Ele cria o ambiente, instala tudo, baixa o
modelo e abre o jogo.

### Manual
```powershell
# dentro da pasta xadrez_facial
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py
```

> O modelo `models/face_landmarker.task` já vem incluído. Se faltar, baixe de:
> https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

### Testar sem webcam
```powershell
.venv\Scripts\python.exe smoke_test.py
```
Valida a lógica do xadrez, a IA e a renderização (gera `preview.png`).

---

## 📦 Requisitos

- **Python 3.14** (testado) ou 3.10–3.13.
- Dependências (`requirements.txt`): `mediapipe`, `opencv-contrib-python`,
  `chess`, `numpy`, `Pillow`.
- Uma **webcam**.

> ⚠️ Use **`opencv-contrib-python`** (não `opencv-python`). O MediaPipe já
> instala o contrib; ter os dois ao mesmo tempo quebra o módulo `cv2`.

---

## 🗂️ Estrutura

```
xadrez_facial/
├── main.py             # loop principal (webcam + jogo + interface)
├── face_controller.py  # MediaPipe: landmarks, pose da cabeça e EAR (piscadas)
├── chess_game.py       # regras do xadrez (python-chess) + cursor/clique
├── ai_opponent.py      # IA adversária (negamax + alfa-beta + material)
├── board_ui.py         # desenho do tabuleiro e do painel (OpenCV + Pillow)
├── smoke_test.py       # teste rápido sem webcam
├── requirements.txt
├── run.bat
└── models/
    └── face_landmarker.task
```

---

## 🕹️ Como o controle da cabeça funciona (importante)

O movimento é por **gesto, não por "empurrar"**: você **inclina a cabeça e volta
ao centro** para dar **1 passo** (como teclas de seta). Se **segurar** inclinado,
ele passa a repetir num ritmo controlado (para atravessar o tabuleiro). Isso
elimina o antigo problema de o cursor "andar infinito" para um lado.

No canto da câmera há um **HUD**: um ponto que mostra a posição da sua cabeça,
o quadrado **verde** (centro/zona morta) e o **azul** (gatilho). Em repouso o
ponto deve ficar no verde; se não ficar, aperte **C** ou fique parado ~1s (ele
**re-centra sozinho**).

## 🔧 Ajuste fino (se precisar)

Em `face_controller.py`, no construtor de `FaceController`:

- `deadzone` (padrão `0.04`): tamanho da zona "centro" (re-arma o gesto).
- `trigger` (padrão `0.085`): o quanto inclinar para disparar 1 passo. **Menor =
  mais sensível** (dispara com menos inclinação).
- `hold_delay` / `repeat_rate` (padrão `0.6` / `0.45`): atraso para começar a
  auto-repetir ao segurar, e o intervalo entre os passos repetidos.
- `max_repeats` (padrão `7`): teto de passos seguidos sem voltar ao centro
  (trava de segurança contra movimento infinito).
- `ear_closed` / `ear_open` (padrão `0.21` / `0.27`): limiares da piscada.
- `wink_diff` (padrão `0.06`): o quão assimétrica a piscada precisa ser para
  contar como *wink* de um olho só (evita disparar numa piscada normal).

**Dicas de uso:** boa iluminação frontal ajuda muito; ao abrir, olhe para a tela
e fique parado um instante (calibra/re-centra sozinho) — aperte `C` para
recalibrar quando quiser; se não conseguir piscar um olho só, use
`Enter`/`Backspace`. Para testar a lógica sem câmera: `python test_movimento.py`.

---

## 🧠 Sobre a IA adversária

`ai_opponent.py` usa **negamax com poda alfa-beta** e avaliação por **material +
tabelas posicionais** (piece-square tables). A profundidade padrão é `depth=2`
(rápida para demo). Aumente em `AIOpponent(depth=...)` para um jogo mais forte.
