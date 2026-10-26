# Audio subsystem

`engine.py` records microphone input with SpeechRecognition and uses local
Faster-Whisper when initialized. Cloud Google Speech Recognition is disabled
by default and requires explicit `ALLOW_CLOUD_SPEECH_RECOGNITION=1`.

Static replies are played from [`../../../assets/audio/`](../../../assets/audio/).
Dynamic gTTS is limited to the fixed generic phrases in `tts_privacy.py`.
The optional template-generation script uses gTTS for fixed prompt text; see
[`../../../scripts/audio/README.md`](../../../scripts/audio/README.md).

Microphone and speaker access are required for live operation.
