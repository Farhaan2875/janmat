@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" -c "import pandas, flask, numpy, sklearn, joblib" >nul 2>&1
if errorlevel 1 (
  echo JanMat dependencies are not installed for this Python interpreter.
  echo Run: %PYTHON% -m pip install -r requirements.txt
  pause
  exit /b 1
)

start "JanMat model server" /b "%PYTHON%" web_app.py
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5000"
echo JanMat is running at http://127.0.0.1:5000
echo Keep this window open while using the application.
pause
