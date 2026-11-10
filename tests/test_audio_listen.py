"""AudioEngine.listen with a fake microphone/Whisper (no hardware)."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

for _n in ("speech_recognition", "gtts", "pygame", "faster_whisper"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        _m.gTTS = object
        _m.mixer = types.SimpleNamespace(init=lambda: None, quit=lambda: None)
        sys.modules[_n] = _m

from triage_kiosk.audio import engine as audio_engine  # noqa: E402

# Fake Silero VAD: speech everywhere unless a test says otherwise.
FAKE_VAD = types.ModuleType("faster_whisper.vad")
FAKE_VAD.VadOptions = lambda **kw: kw
FAKE_VAD.speech = True
FAKE_VAD.get_speech_timestamps = lambda audio, opts=None: (
    [{"start": 0, "end": len(audio)}] if FAKE_VAD.speech else [])


class _WaitTimeout(Exception):
    pass


class _Unknown(Exception):
    pass


class FakeMic:
    opened = closed = 0

    def __init__(self, device_index=None):
        self.stream = types.SimpleNamespace(pyaudio_stream=types.SimpleNamespace(
            get_read_available=lambda: 0, read=lambda n, exception_on_overflow=False: b""))

    def __enter__(self):
        FakeMic.opened += 1
        return self

    def __exit__(self, *a):
        FakeMic.closed += 1


class FakeAudioData:
    sample_rate, sample_width = 16000, 2
    frame_data = b"\x00\x01" * 16000

    def get_raw_data(self, convert_rate=None, convert_width=None):
        return self.frame_data


class FakeRecognizer:
    pause_threshold, energy_threshold = 1.2, 300

    def __init__(self, error=None):
        self.error = error

    def listen(self, source, timeout=None, phrase_time_limit=None):
        if self.error:
            raise self.error
        return FakeAudioData()


class FakeWhisper:
    def __init__(self, probs=None, speech=True):
        self.probs, self.speech, self.languages = probs or [("th", 0.7), ("en", 0.2)], speech, []

    def detect_language(self, audio=None, vad_filter=False, vad_parameters=None):
        # the real detect_language() crashes on a dict of VAD options ('dict' has no 'threshold')
        assert len(audio) > 0 and not isinstance(vad_parameters, dict)
        return self.probs[0][0], self.probs[0][1], self.probs

    def transcribe(self, audio, language=None, **kw):
        self.languages.append(language)
        return [types.SimpleNamespace(text=" สวัสดี ")], None


def make_engine(whisper, recognizer=None):
    eng = audio_engine.AudioEngine.__new__(audio_engine.AudioEngine)
    eng.whisper_model = whisper
    eng.recognizer = recognizer or FakeRecognizer()
    eng._mic = eng._source = None
    eng.last_language = (None, 0.0)
    return eng


FAKE_SR = types.SimpleNamespace(Microphone=FakeMic, WaitTimeoutError=_WaitTimeout, UnknownValueError=_Unknown)


@patch.dict(sys.modules, {"faster_whisper.vad": FAKE_VAD})
@patch.object(audio_engine, "sr", FAKE_SR)
class ListenTests(unittest.TestCase):
    def setUp(self):
        FakeMic.opened = FakeMic.closed = 0
        FAKE_VAD.speech = True

    def test_microphone_opened_once_and_not_closed_between_replies(self):
        eng = make_engine(FakeWhisper())
        for _ in range(3):
            self.assertEqual(eng.listen("th"), "สวัสดี")
        self.assertEqual((FakeMic.opened, FakeMic.closed), (1, 0))

    def test_auto_chooses_only_thai_or_english(self):
        w = FakeWhisper(probs=[("zh", 0.5), ("en", 0.3), ("th", 0.1)])
        eng = make_engine(w)
        eng.listen("auto")
        self.assertEqual(w.languages, ["en"])
        self.assertEqual(eng.last_language[0], "en")

    def test_auto_with_no_speech_skips_whisper(self):
        FAKE_VAD.speech = False
        w = FakeWhisper()
        eng = make_engine(w)
        self.assertEqual(eng.listen("auto"), "")
        self.assertEqual(w.languages, [])

    def test_silence_timeout_returns_empty(self):
        eng = make_engine(FakeWhisper(), FakeRecognizer(error=_WaitTimeout()))
        self.assertEqual(eng.listen("th"), "")

    def test_device_error_raises_and_reopens_next_time(self):
        eng = make_engine(FakeWhisper(), FakeRecognizer(error=OSError("unplugged")))
        with self.assertRaises(audio_engine.AudioDeviceError):
            eng.listen("th")
        self.assertIsNone(eng._source)


if __name__ == "__main__":
    unittest.main()
