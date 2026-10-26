# Audio-template generator

`generate_templates.py` reads the fixed prompt strings and output filenames
from `SystemConfig` and generates missing MP3 files under `assets/audio/`.
Existing non-empty files are retained.

This is an optional maintainer tool; the checked-in audio templates are
already available at runtime. It requires the kiosk environment and network
access to gTTS. The text submitted is the fixed prompt text in
`src/triage_kiosk/config.py`, not patient speech or LLM output.

From the repository root:

```bash
PYTHONPATH=src python scripts/audio/generate_templates.py
```
