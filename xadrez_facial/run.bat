@echo off
REM ============================================================
REM  Xadrez Facial - inicializador para Windows
REM  Cria o ambiente virtual (se preciso), instala dependencias,
REM  baixa o modelo do MediaPipe e roda o jogo.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Criando ambiente virtual com Python 3.14...
    py -3.14 -m venv .venv
    .venv\Scripts\python.exe -m pip install --upgrade pip
    echo [setup] Instalando dependencias...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)

if not exist "models\face_landmarker.task" (
    echo [setup] Baixando modelo do MediaPipe...
    if not exist "models" mkdir models
    powershell -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -OutFile 'models\face_landmarker.task'"
)

echo [run] Iniciando o jogo...
.venv\Scripts\python.exe main.py
pause
