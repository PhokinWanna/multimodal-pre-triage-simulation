#!/bin/bash
# Double-click in Finder, or run from Terminal. Works from any folder.
cd "$(dirname "$0")/.." || exit 1
KEY_FILE="data/runtime/kiosk_key.txt"
[ -s "$KEY_FILE" ] || { echo "[!] Run 1_setup first."; read -r -p "Press Enter..." _; exit 1; }
export DASHBOARD_KIOSK_KEY="$(cat "$KEY_FILE")"
export PYTHONPATH=src
(sleep 4; open "http://127.0.0.1:8000") &
echo "Nurse dashboard: http://127.0.0.1:8000   (close this window or press Ctrl+C to stop)"
exec .venv-dashboard/bin/python -m uvicorn nurse_dashboard.app:app --port 8000
