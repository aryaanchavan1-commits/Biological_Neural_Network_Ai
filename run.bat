@echo off
title BIO-NN Research Framework
color 0B

echo ============================================================
echo   BIO-NN: Biologically Inspired Neural Network
echo   Research Framework Launcher
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if venv exists
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment found.
)

echo.
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/5] Installing dependencies...
pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>nul
pip install snntorch -q 2>nul
pip install numpy pyyaml matplotlib seaborn scikit-learn pandas psutil -q 2>nul
pip install fastapi uvicorn jinja2 -q 2>nul
pip install -e . -q 2>nul

echo.
echo [4/5] Verifying installation...
python -c "import torch; import snntorch; print(f'  PyTorch: {torch.__version__}'); print(f'  snnTorch: {snntorch.__version__}')"
if errorlevel 1 (
    echo [ERROR] Installation verification failed
    pause
    exit /b 1
)

echo.
echo [5/5] Starting BIO-NN Dashboard...
echo.
echo ============================================================
echo   Dashboard URL:  http://localhost:8000
echo   3D Network:     http://localhost:8000/network
echo   API Data:       http://localhost:8000/api/data
echo ============================================================
echo.
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m bio_nn.visualization.dashboard.server

pause
