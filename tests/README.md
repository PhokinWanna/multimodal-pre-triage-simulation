# Automated tests

This folder contains fast unit tests for privacy-sensitive behavior such as
consent refusal and external TTS text restrictions. Run them from the
repository root with the kiosk environment active:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

These tests do not emulate a physical camera, microphone, Ollama server, or
nurse dashboard. They are not end-to-end or clinical validation.
