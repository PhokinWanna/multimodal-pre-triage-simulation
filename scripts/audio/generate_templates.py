"""generate_templates.py - pre-render every fixed kiosk sentence as MP3 (needs internet once).

1. The 9 classic prompts per language (greeting, PDPA, ...) -> assets/audio/<lang>_<key>.mp3
2. Every other fixed sentence (interview questions, re-asks, "anything else", cancel,
   consent re-ask, posture prompts, off-topic reply) -> assets/audio/fixed/<lang>_<hash>.mp3

With these files the kiosk speaks offline on any OS (Windows needs no Thai system voice).
Only fixed kiosk text is sent to gTTS - never patient or LLM text.

    PYTHONPATH=src python scripts/audio/generate_templates.py          # only missing files
    PYTHONPATH=src python scripts/audio/generate_templates.py --force  # re-render everything
"""

import argparse
import os
import sys

from gtts import gTTS

from triage_kiosk.config import SystemConfig
from triage_kiosk.tts_privacy import fixed_audio_filename, fixed_kiosk_texts


def _render(text: str, lang: str, path: str, force: bool) -> bool:
    if not force and os.path.exists(path) and os.path.getsize(path) > 0:
        return False
    gTTS(text=text, lang=lang, slow=False).save(path)
    return True


def build_audio_templates(force: bool = False) -> int:
    os.makedirs(SystemConfig.AUDIO_TEMPLATE_DIR, exist_ok=True)
    os.makedirs(SystemConfig.FIXED_AUDIO_DIR, exist_ok=True)
    made = 0
    try:
        for lang, clips in SystemConfig.TEMPLATES.items():
            for key, filename in clips.items():
                path = os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, filename)
                if _render(SystemConfig.SCRIPT_TEXTS[lang][key], lang, path, force):
                    print(f"  [+] {filename}")
                    made += 1
        for lang, text in fixed_kiosk_texts():
            filename = fixed_audio_filename(lang, text)
            if _render(text, lang, os.path.join(SystemConfig.FIXED_AUDIO_DIR, filename), force):
                print(f"  [+] fixed/{filename}  {text[:50]}")
                made += 1
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[!] Could not reach the text-to-speech service ({e}). Connect to the internet and run again.")
    print(f"[ok] Voice prompts ready ({made} new file(s)) in {SystemConfig.AUDIO_TEMPLATE_DIR}")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render files that already exist")
    build_audio_templates(force=ap.parse_args().force)
