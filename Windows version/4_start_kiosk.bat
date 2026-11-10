@echo off
rem Double-click in File Explorer, or run from Command Prompt. Works from any folder.
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
set "KEY_FILE=data\runtime\kiosk_key.txt"
if not exist "%KEY_FILE%" ( echo [!] Run 1_setup first. & pause & exit /b 1 )
set /p NURSE_DASHBOARD_KEY=<"%KEY_FILE%"
set PYTHONPATH=src
echo Starting the kiosk (press q in the camera window to quit)...
.venv-kiosk\Scripts\python -m triage_kiosk.main
echo Kiosk stopped.
pause
