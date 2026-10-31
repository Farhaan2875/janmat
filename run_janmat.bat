@echo off
setlocal
cd /d "%~dp0"

set "PYTHON="
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -c "import pandas, flask, numpy, sklearn, joblib" >nul 2>&1
  if not errorlevel 1 set "PYTHON=venv\Scripts\python.exe"
)

if not defined PYTHON (
  python -c "import pandas, flask, numpy, sklearn, joblib" >nul 2>&1
  if not errorlevel 1 set "PYTHON=python"
)

if not defined PYTHON (
  echo JanMat dependencies are not installed.
  echo Install them with:
  echo python -m pip install -r requirements.txt
  pause
  exit /b 1
)

start "JanMat model server" /b "%PYTHON%" web_app.py
set "READY="
for /l %%I in (1,1,15) do (
  powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/models -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 (
    set "READY=1"
    goto :ready
  )
  timeout /t 1 /nobreak >nul
)

:ready
if not defined READY (
  echo JanMat could not start. Check the Python error above.
  pause
  exit /b 1
)

start "" "http://127.0.0.1:5000"
echo JanMat is running at http://127.0.0.1:5000
echo Close this window to stop using the launcher.
pause
