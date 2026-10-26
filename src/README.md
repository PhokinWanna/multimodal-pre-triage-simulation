# Application source

The kiosk application lives in [`triage_kiosk/`](triage_kiosk/):

- `config.py` resolves model, audio, evidence, and outbox paths relative to
  the repository root.
- `main.py` coordinates camera telemetry, consent, symptom dialogue,
  finalization, and case dispatch.
- `vision/`, `audio/`, and `triage/` contain the perception, interaction,
  and reasoning components.

Start the package from the repository root as described in the main
[README](../README.md). Hardware, Ollama, and dashboard integration must be
available for a complete live workflow.
