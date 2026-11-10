import sys
import types
import unittest

for _n in ("speech_recognition", "gtts", "pygame", "faster_whisper"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.gTTS = object
        _m.mixer = types.SimpleNamespace(init=lambda: None, quit=lambda: None)
        sys.modules[_n] = _m

from triage_kiosk.audio.engine import pick_voices  # noqa: E402

LISTING = """Albert              en_US    # Hello! My name is Albert.
Kanya (Enhanced)    th_TH    # สวัสดีค่ะ ฉันชื่อกันยา
Samantha            en_US    # Hello! My name is Samantha.
Thomas              fr_FR    # Bonjour, je m'appelle Thomas.
"""


class PickVoicesTests(unittest.TestCase):
    def test_picks_configured_and_enhanced_voices(self):
        self.assertEqual(pick_voices(LISTING), {"th": "Kanya (Enhanced)", "en": "Samantha"})

    def test_missing_thai_voice(self):
        self.assertEqual(pick_voices(LISTING.replace("th_TH", "xx_XX")), {"en": "Samantha"})
