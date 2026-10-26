# Kiosk application package

`main.py` runs the kiosk state machine and joins its components:

- [`vision/`](vision/) captures pose/face telemetry and evaluates skin input.
- [`audio/`](audio/) handles microphone input and local/static prompt playback.
- [`triage/`](triage/) dispatches conversation actions and composes a triage
  result through the local Ollama endpoint, with a deterministic fallback.
- `consent.py` clears an unconsented case and creates a minimal
  nurse-assistance event.
- `tts_privacy.py` restricts external gTTS text to fixed generic phrases.

Resource paths are resolved from the repository root in `config.py`.
The dashboard is not implemented; queued events are not proof that a nurse
receives them. This simulation is not a medical device.
