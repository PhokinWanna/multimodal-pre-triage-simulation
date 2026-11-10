#!/bin/bash
# Double-click in Finder, or run from Terminal. Works from any folder.
cd "$(dirname "$0")/.." || exit 1
KEY_FILE="data/runtime/kiosk_key.txt"
[ -s "$KEY_FILE" ] || { echo "[!] Run 1_setup first."; read -r -p "Press Enter..." _; exit 1; }
export NURSE_DASHBOARD_KEY="$(cat "$KEY_FILE")"
export PYTHONPATH=src
echo "Starting the kiosk (press q in the camera window to quit)..."
.venv-kiosk/bin/python -m triage_kiosk.main
read -r -p "Kiosk stopped. Press Enter to close..." _
