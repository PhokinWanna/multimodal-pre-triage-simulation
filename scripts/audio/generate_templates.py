"""
generate_templates.py - Audio Template Pre-Baking Tool
Pre-renders all static MP3 audio clips for zero-latency, privacy-compliant playback.
"""

import os
from gtts import gTTS
from triage_kiosk.config import SystemConfig

def build_audio_templates():
    output_dir = SystemConfig.AUDIO_TEMPLATE_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Pre-rendering Audio Templates into: {output_dir}")

    total_files = 0
    for lang, clips in SystemConfig.TEMPLATES.items():
        for key, filename in clips.items():
            filepath = os.path.join(output_dir, filename)
            text = SystemConfig.SCRIPT_TEXTS[lang][key]

            # Generate only if missing or zero bytes
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                print(f"  [+] Generating [{lang}] {key} -> {filename}...")
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(filepath)
                total_files += 1
            else:
                print(f"  [i] Already cached: {filename}")

    print(f"[✓] Audio Template Caching Complete. {total_files} new files rendered.")

if __name__ == "__main__":
    build_audio_templates()