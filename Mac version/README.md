# Mac version (macOS 12+, Apple Silicon M1 or newer)

The code is shared with the Windows version (`src/`); this folder only holds the Mac
setup and start scripts. Double-click them in Finder (first time: right-click → Open),
or run them in Terminal, e.g. `bash "Mac version/1_setup.command"`.

| Step | Script | What it does |
|---|---|---|
| 1 | `1_setup.command` | Once: portaudio, Python environments, voice prompts, `ollama pull llama3.2`, shared kiosk key, first dashboard (admin) account |
| 2 | `2_start_dashboard.command` | Starts the nurse dashboard and opens http://127.0.0.1:8000 |
| 3 | `3_preflight_check.command` | Checks camera, mic, speaker, voices, Whisper, Ollama and the dashboard (should end with `0 FAIL`) |
| 4 | `4_start_kiosk.command` | Starts the kiosk (press **q** in the camera window to quit) |

**Before step 1:** install [Homebrew](https://brew.sh), Python 3.11 (`brew install python@3.11`)
and [Ollama](https://ollama.com) (open the app once).

**Permissions:** allow Camera and Microphone for Terminal when macOS asks
(System Settings → Privacy & Security).

**Optional:** Thai voice *Kanya* for spoken questions — System Settings → Accessibility →
Spoken Content → System voice → Manage Voices → Thai. Without it the pre-recorded
questions are used.

Intel Macs: `ai-edge-litert` has no Intel-Mac build; install `tensorflow==2.15.0` into
`.venv-kiosk` instead (the skin model then runs through TensorFlow Lite).

Useful settings: see the main [README](../README.md#getting-started). Test scenarios:
[docs/demo/REAL_WORLD_TEST.md](../docs/demo/REAL_WORLD_TEST.md).
