@echo off
rem Double-click in File Explorer, or run from Command Prompt. Works from any folder.
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
set "KEY_FILE=data\runtime\kiosk_key.txt"
if not exist "%KEY_FILE%" ( echo [!] Run 1_setup first. & pause & exit /b 1 )
set /p DASHBOARD_KIOSK_KEY=<"%KEY_FILE%"
set PYTHONPATH=src
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8000"
echo Nurse dashboard: http://127.0.0.1:8000   (close this window or press Ctrl+C to stop)
.venv-dashboard\Scripts\python -m uvicorn nurse_dashboard.app:app --port 8000
pause
