@echo off
TITLE Helmet Detection & Road Safety Analyzer
echo ========================================================
echo Starting Helmet Detection & Road Safety Analyzer
echo Streamlit + YOLOv8 Deep Learning + ByteTrack
echo ========================================================

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating .venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [INFO] Upgrading pip and installing requirements...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies from requirements.txt.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate.bat
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo [WARNING] Streamlit not detected in .venv. Installing dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo [INFO] Model checkpoint check...
if exist "models\best.pt" (
    echo [INFO] Model found at models\best.pt
) else if exist "runs\train\helmet_detector2\weights\best.pt" (
    echo [INFO] Model found at legacy path runs\train\...
) else (
    echo [ERROR] Trained weights file not found!
    echo        Expected: models\best.pt
    echo        Legacy:   runs\train\helmet_detector2\weights\best.pt
    pause
    exit /b 1
)

echo [INFO] Launching Streamlit application on default port 8501...
python -m streamlit run app.py
if errorlevel 1 (
    echo [ERROR] Application failed to launch or exited with an error.
    pause
    exit /b 1
)
pause
