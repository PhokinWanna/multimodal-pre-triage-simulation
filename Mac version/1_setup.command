#!/bin/bash
# Double-click in Finder, or run from Terminal. Works from any folder.
cd "$(dirname "$0")/.." || exit 1
KEY_FILE="data/runtime/kiosk_key.txt"
set -e
echo "=== Multimodal Pre-Triage: Mac setup ==="
PY=$(command -v python3.11 || true)
if [ -z "$PY" ]; then
  echo "[!] Python 3.11 not found. Install it with:  brew install python@3.11   (then run this again)"; exit 1
fi
if command -v brew >/dev/null 2>&1; then
  brew list portaudio >/dev/null 2>&1 || brew install portaudio      # needed by PyAudio (microphone)
else
  echo "[!] Homebrew not found (https://brew.sh). It is needed to install portaudio for the microphone."; exit 1
fi

echo "--- kiosk environment (.venv-kiosk)"
[ -d .venv-kiosk ] || "$PY" -m venv .venv-kiosk
.venv-kiosk/bin/python -m pip install --upgrade pip -q
.venv-kiosk/bin/python -m pip install -r "Mac version/requirements-kiosk-mac.txt"

echo "--- dashboard environment (.venv-dashboard)"
[ -d .venv-dashboard ] || "$PY" -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install --upgrade pip -q
.venv-dashboard/bin/python -m pip install -r requirements-dashboard.txt

mkdir -p data/runtime
if [ ! -s "$KEY_FILE" ]; then
  .venv-kiosk/bin/python -c "import secrets; print(secrets.token_urlsafe(24), end='')" > "$KEY_FILE"
  echo "--- created the shared kiosk key in $KEY_FILE"
fi

echo "--- voice prompts (needs internet once)"
PYTHONPATH=src .venv-kiosk/bin/python scripts/audio/generate_templates.py || echo "[!] Skipped: no internet. Run setup again later."

echo "--- local LLM"
if command -v ollama >/dev/null 2>&1; then
  ollama pull llama3.2 || echo "[!] Open the Ollama app, then run setup again."
else
  echo "[!] Ollama not installed: download it from https://ollama.com, open it, then run setup again."
fi

echo "--- dashboard account"
if PYTHONPATH=src .venv-dashboard/bin/python -m nurse_dashboard list-users | grep -q "No users yet"; then
  echo "Create the first (admin) account for the nurse dashboard:"
  PYTHONPATH=src .venv-dashboard/bin/python -m nurse_dashboard create-admin
fi

echo
echo "=== Setup finished. ==="
echo "Optional: Thai voice 'Kanya' - System Settings > Accessibility > Spoken Content > System voice > Manage Voices > Thai."
echo "Next: 2_start_dashboard, then 3_preflight_check, then 4_start_kiosk."
read -r -p "Press Enter to close..." _
