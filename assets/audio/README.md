# Audio prompt assets

This folder contains the pre-rendered Thai (`th_*.mp3`) and English
(`en_*.mp3`) prompts referenced by `SystemConfig.TEMPLATES`. The audio engine
loads these files by filename from this folder.

Keep template names synchronized with `src/triage_kiosk/config.py`. Some
existing prompts describe a nurse notification even though the dashboard is
not implemented; treat those statements as simulation text until integration
is completed.

The optional generator is documented in
[`../../scripts/audio/README.md`](../../scripts/audio/README.md).
