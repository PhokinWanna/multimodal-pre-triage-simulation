"""Windows/macOS code paths, tested with fakes (runs on any OS)."""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

for _n in ("speech_recognition", "gtts", "pygame", "faster_whisper"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.gTTS = object
        _m.mixer = types.SimpleNamespace(init=lambda: None, quit=lambda: None)
        sys.modules[_n] = _m

from triage_kiosk import platform_support as plat  # noqa: E402
from triage_kiosk.audio import engine as audio_engine  # noqa: E402
from triage_kiosk.config import SystemConfig  # noqa: E402
from triage_kiosk.tts_privacy import fixed_audio_filename, fixed_kiosk_texts, is_safe_fixed_text  # noqa: E402

WIN_VOICES = [
    ("Microsoft David Desktop - English (United States)", "409"),
    ("Microsoft Zira Desktop - English (United States)", "409"),
    ("Microsoft Pattara - Thai (Thailand)", "41E"),
]


class PickSapiVoiceTests(unittest.TestCase):
    def test_thai_onecore_voice_found_by_lcid(self):
        self.assertEqual(plat.pick_sapi_voice(WIN_VOICES, "th", "Pattara"), "Microsoft Pattara - Thai (Thailand)")

    def test_english_prefers_configured_then_known_names(self):
        self.assertIn("Zira", plat.pick_sapi_voice(WIN_VOICES, "en", "Zira"))
        self.assertIn("Zira", plat.pick_sapi_voice(WIN_VOICES, "en", "NotInstalled"))

    def test_no_thai_voice(self):
        self.assertIsNone(plat.pick_sapi_voice(WIN_VOICES[:2], "th", "Pattara"))


class OpenCameraTests(unittest.TestCase):
    def test_windows_tries_directshow_first(self):
        calls = []

        class Cap:
            def __init__(self, *args):
                calls.append(args)

            def isOpened(self):
                return True

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(VideoCapture=Cap, CAP_DSHOW=700, CAP_MSMF=1400)
        with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(plat, "IS_WINDOWS", True):
            self.assertIsNotNone(plat.open_camera(0))
        self.assertEqual(calls, [(0, 700)])

    def test_returns_none_when_no_camera(self):
        class Cap:
            def __init__(self, *args):
                pass

            def isOpened(self):
                return False

            def release(self):
                pass

        fake_cv2 = types.SimpleNamespace(VideoCapture=Cap, CAP_DSHOW=700, CAP_MSMF=1400)
        with patch.dict(sys.modules, {"cv2": fake_cv2}), patch.object(plat, "IS_WINDOWS", True):
            self.assertIsNone(plat.open_camera(0))


class FixedTextTests(unittest.TestCase):
    def test_catalogue_has_all_interview_sentences_in_both_languages(self):
        texts = fixed_kiosk_texts()
        self.assertTrue({l for l, _ in texts} == {"th", "en"})
        self.assertTrue(is_safe_fixed_text(SystemConfig.SPOKEN_PROMPTS["th"]["anything_else_more"]))
        self.assertFalse(is_safe_fixed_text("ผมปวดหัวมาก"))          # patient text never allowed
        self.assertEqual(len({fixed_audio_filename(l, t) for l, t in texts}), len(texts))


def _engine(voices):
    eng = audio_engine.AudioEngine.__new__(audio_engine.AudioEngine)
    eng.local_voices = voices
    eng.played, eng.spoken = [], []
    eng._play_audio_file = lambda path, block=True: eng.played.append(os.path.basename(path))
    eng.play_template = lambda key, lang="th", block=True: eng.played.append(f"template:{key}")
    eng.speak_text = lambda text, lang="th", block=True: (eng.spoken.append(text) or True) if lang in voices else False
    return eng


class SpeakFixedOrderTests(unittest.TestCase):
    def setUp(self):
        self.lang, self.text = fixed_kiosk_texts()[1]
        self.tmp = tempfile.TemporaryDirectory()
        open(os.path.join(self.tmp.name, fixed_audio_filename(self.lang, self.text)), "wb").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_windows_uses_prerecorded_file_first(self):
        eng = _engine({"th": "Pattara", "en": "Zira"})
        with patch.object(audio_engine, "IS_MAC", False), patch.object(SystemConfig, "FIXED_AUDIO_DIR", self.tmp.name):
            self.assertEqual(eng.speak_fixed(self.text, self.lang), "file")
        self.assertEqual(eng.spoken, [])

    def test_mac_keeps_live_voice_first(self):
        eng = _engine({"th": "Kanya", "en": "Samantha"})
        with patch.object(audio_engine, "IS_MAC", True), patch.object(SystemConfig, "FIXED_AUDIO_DIR", self.tmp.name):
            self.assertEqual(eng.speak_fixed(self.text, self.lang), "say")

    def test_windows_without_file_uses_sapi_voice(self):
        eng = _engine({"th": "Pattara", "en": "Zira"})
        with patch.object(audio_engine, "IS_MAC", False), patch.object(SystemConfig, "FIXED_AUDIO_DIR", "/nonexistent"):
            self.assertEqual(eng.speak_fixed(self.text, self.lang), "say")


class WindowsVoiceDetectionTests(unittest.TestCase):
    def test_detects_with_fake_sapi(self):
        with patch.object(audio_engine.plat, "list_windows_voices", return_value=WIN_VOICES):
            found = audio_engine.detect_windows_voices()
        self.assertEqual(found["th"], "Microsoft Pattara - Thai (Thailand)")
        self.assertIn("Zira", found["en"])

    def test_missing_pywin32_is_not_fatal(self):
        with patch.object(audio_engine.plat, "list_windows_voices", side_effect=ImportError("No module named win32com")):
            self.assertEqual(audio_engine.detect_windows_voices(), {})


if __name__ == "__main__":
    unittest.main()
