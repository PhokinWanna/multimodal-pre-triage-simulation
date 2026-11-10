# Windows version (Windows 10 / 11, 64-bit)

The code is shared with the Mac version (`src/`); this folder only holds the Windows
setup and start scripts. Double-click them in File Explorer (if SmartScreen asks:
More info → Run anyway), or run them in Command Prompt.

| Step | Script | What it does |
|---|---|---|
| 1 | `1_setup.bat` | Once: Python environments, voice prompts, `ollama pull llama3.2`, shared kiosk key, first dashboard (admin) account |
| 2 | `2_start_dashboard.bat` | Starts the nurse dashboard and opens http://127.0.0.1:8000 |
| 3 | `3_preflight_check.bat` | Checks camera, mic, speaker, voices, Whisper, Ollama and the dashboard (should end with `0 FAIL`) |
| 4 | `4_start_kiosk.bat` | Starts the kiosk (press **q** in the camera window to quit) |

**Before step 1:**
- [Python 3.11 (64-bit)](https://www.python.org/downloads/windows/) — tick *Add python.exe to PATH*
  and *py launcher* in the installer.
- [Ollama for Windows](https://ollama.com/download/windows) — open it once.

**Permissions:** Settings → Privacy & security → Camera / Microphone → turn on
*Let desktop apps access*. Close Zoom/Teams so the camera is free.

**What is different from the Mac:**
- Prompts play with pygame; the camera uses DirectShow (opens faster on Windows).
- All interview questions are pre-recorded MP3s (made by `1_setup.bat`, needs internet once),
  so no Thai system voice is needed.
- Optional Thai system voice (*Microsoft Pattara*, only used with `INTERVIEW_MODE=llm`):
  Settings → Time & language → Language & region → add Thai → Language options → Speech.
- If the wrong microphone or camera is used: run step 3, note the number of the right device,
  then before step 4 type `set MIC_DEVICE_INDEX=2` / `set CAMERA_INDEX=1` in the same window
  (or add the line to `4_start_kiosk.bat`).

Useful settings: see the main [README](../README.md#getting-started). Test scenarios:
[docs/demo/REAL_WORLD_TEST.md](../docs/demo/REAL_WORLD_TEST.md).
