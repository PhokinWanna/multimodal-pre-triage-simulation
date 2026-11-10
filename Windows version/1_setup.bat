@echo off
rem Double-click in File Explorer, or run from Command Prompt. Works from any folder.
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
set "KEY_FILE=data\runtime\kiosk_key.txt"
echo === Multimodal Pre-Triage: Windows setup ===
py -3.11 --version >nul 2>&1
if not errorlevel 1 goto :have_python
echo [!] Python 3.11 not found. Install Python 3.11 64-bit from https://www.python.org/downloads/windows/
echo     and tick "Add python.exe to PATH" and "py launcher". Then run this again.
goto :fail
:have_python

echo --- kiosk environment (.venv-kiosk)
if not exist .venv-kiosk ( py -3.11 -m venv .venv-kiosk || goto :fail )
.venv-kiosk\Scripts\python -m pip install --upgrade pip -q
.venv-kiosk\Scripts\python -m pip install -r "Windows version\requirements-kiosk-windows.txt" || goto :fail

echo --- dashboard environment (.venv-dashboard)
if not exist .venv-dashboard ( py -3.11 -m venv .venv-dashboard || goto :fail )
.venv-dashboard\Scripts\python -m pip install --upgrade pip -q
.venv-dashboard\Scripts\python -m pip install -r requirements-dashboard.txt || goto :fail

if not exist data\runtime mkdir data\runtime
if exist "%KEY_FILE%" goto :have_key
.venv-kiosk\Scripts\python -c "import secrets; print(secrets.token_urlsafe(24), end='')" > "%KEY_FILE%"
echo --- created the shared kiosk key in %KEY_FILE%
:have_key

echo --- voice prompts (needs internet once)
set PYTHONPATH=src
.venv-kiosk\Scripts\python scripts\audio\generate_templates.py
if errorlevel 1 echo [!] Skipped: no internet. Run setup again later.

echo --- local LLM
where ollama >nul 2>&1
if errorlevel 1 goto :no_ollama
ollama pull llama3.2
goto :ollama_done
:no_ollama
echo [!] Ollama not installed: download it from https://ollama.com/download/windows, then run setup again.
:ollama_done

echo --- dashboard account
.venv-dashboard\Scripts\python -m nurse_dashboard list-users | findstr /C:"No users yet" >nul
if errorlevel 1 goto :users_done
echo Create the first admin account for the nurse dashboard:
.venv-dashboard\Scripts\python -m nurse_dashboard create-admin
:users_done

echo.
echo === Setup finished. ===
echo Optional Thai system voice: Settings ^> Time ^& language ^> Language ^& region ^> add Thai ^> Language options ^> Speech.
echo Next: 2_start_dashboard, then 3_preflight_check, then 4_start_kiosk.
pause
exit /b 0

:fail
echo.
echo [!] Setup stopped because of the error above.
pause
exit /b 1
