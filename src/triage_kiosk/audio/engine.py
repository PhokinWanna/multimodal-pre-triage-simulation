"""Audio engine: on-device speech-to-text and voice-prompt playback.

Uses Faster-Whisper for local transcription, plays pre-recorded Thai/English
prompts (afplay on macOS, pygame on Windows/Linux), speaks with the OFFLINE system
voice (macOS `say`, Windows SAPI) and limits external TTS (gTTS) to fixed sentences.
"""

import logging
import os
import re
import time
import tempfile
import threading
import subprocess
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from ..config import SystemConfig
from ..tts_privacy import fixed_audio_filename, is_safe_fixed_text, safe_off_topic_text
from .. import platform_support as plat
from ..intent import find_wake_word

logger = logging.getLogger(__name__)

IS_MAC, IS_WINDOWS = plat.IS_MAC, plat.IS_WINDOWS
_MIXER_READY = False

if not IS_MAC:
    import pygame
    try:
        pygame.mixer.init()
        _MIXER_READY = True
    except Exception as _e:  # noqa: BLE001  (no sound device, e.g. a headless Linux box)
        logging.getLogger(__name__).warning(f"No audio output device ({_e}); voice prompts will be silent.")

class AudioDeviceError(RuntimeError):
    """The microphone or the speech engine cannot be used (unplugged, no permission, not installed)."""


def detect_local_voices() -> dict:
    """Return {lang: voice} for the offline voices that are installed (macOS `say`, Windows SAPI)."""
    if not SystemConfig.LOCAL_TTS_ENABLED:
        return {}
    if IS_WINDOWS:
        return detect_windows_voices()
    if not IS_MAC:
        return {}
    try:
        listing = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True,
                                 timeout=10).stdout
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Offline TTS unavailable: {e}")
        return {}
    return pick_voices(listing)


def detect_windows_voices() -> dict:
    try:
        voices = plat.list_windows_voices()
    except Exception as e:  # noqa: BLE001  (pywin32 missing / SAPI broken)
        logger.warning(f"Windows offline voices unavailable ({e}); install pywin32 (requirements in 'Windows version').")
        return {}
    found = {}
    for lang, wanted in SystemConfig.LOCAL_TTS_VOICES_WINDOWS.items():
        choice = plat.pick_sapi_voice(voices, lang, wanted)
        if choice:
            found[lang] = choice
            logger.info(f"Offline voice for '{lang}': {choice}")
        else:
            logger.warning(f"No Windows '{lang}' voice installed; fixed prompts use the pre-recorded audio.")
    return found


def pick_voices(say_listing: str) -> dict:
    """Choose an installed voice per language from the output of `say -v ?`.

    Tries the configured voice first, then any installed voice whose locale matches
    (th_TH / en_*). Enhanced/Premium variants are written "Name (Enhanced)".
    """
    entries = []  # (full_name, locale)
    for line in say_listing.splitlines():
        m = re.match(r"^(.+?)\s{2,}([a-z]{2}_[A-Z]{2})\b", line)
        if m:
            entries.append((m.group(1).strip(), m.group(2)))
    found = {}
    for lang, wanted in SystemConfig.LOCAL_TTS_VOICES.items():
        prefix = "th_" if lang == "th" else "en_"
        same_lang = [name for name, loc in entries if loc.startswith(prefix)]
        exact = [n for n in same_lang if n == wanted or n.split(" (")[0] == wanted]
        preferred_en = [n for n in same_lang if n.split(" (")[0] in ("Samantha", "Daniel", "Karen", "Alex")]
        choice = (exact or (preferred_en if lang == "en" else []) or same_lang or [None])[0]
        if choice:
            found[lang] = choice
            logger.info(f"Offline voice for '{lang}': {choice}")
        else:
            logger.warning(f"No offline '{lang}' voice installed; follow-up questions in '{lang}' "
                           f"will use the generic recording.")
    return found


class AudioEngine:
    def __init__(self):
        logger.info(f"Initializing Audio Engine Subsystems (Platform: {plat.PLATFORM_NAME})...")
        self._is_speaking = False
        self.audio_lock = threading.Lock()
        self._mic = self._source = None
        self.last_language = (None, 0.0)

        # 1. Initialize Microphone Listener with Snappy 1.5s Pause Threshold
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = SystemConfig.SPEECH_PAUSE_THRESHOLD
        self.recognizer.energy_threshold = SystemConfig.MIC_ENERGY_THRESHOLD
        self.recognizer.dynamic_energy_threshold = SystemConfig.MIC_DYNAMIC_ENERGY
        self.recognizer.dynamic_energy_adjustment_ratio = SystemConfig.MIC_DYNAMIC_RATIO

        # Pre-calibrate microphone once at boot (re-done at the start of every encounter)
        self.calibrate(0.6)

        # 2. Load Local Fast-Whisper Model (CTranslate2 int8 Quantization + 4 CPU Threads)
        self.whisper_model = None
        self._init_faster_whisper()
        self.warmup()

        # 3. Offline voices for the LLM's follow-up questions (macOS `say`, on-device).
        self.local_voices = detect_local_voices()

    # -------------------------------------------------------------------------
    # MICROPHONE: one stream kept open for the whole session
    # -------------------------------------------------------------------------
    # Opening and closing a PortAudio stream for every reply can hang on macOS: the
    # 3rd Mac test froze inside pyaudio stop_stream() (CoreAudio). The stream is now
    # opened once and only closed at shutdown (or after a device error).
    def _open_source(self):
        if getattr(self, "_source", None) is None:
            mic_idx = SystemConfig.MICROPHONE_DEVICE_INDEX
            try:
                mic = sr.Microphone(device_index=mic_idx)
                source = mic.__enter__()
            except Exception as e:  # noqa: BLE001  (PyAudio raises OSError / AttributeError)
                raise AudioDeviceError(f"Microphone cannot be opened: {e}") from e
            self._mic, self._source = mic, source
            logger.info("Microphone stream opened (kept open for the session).")
        return self._source

    def _close_source(self) -> None:
        """Close the stream in a background thread so a hanging CoreAudio call cannot block the kiosk."""
        mic = getattr(self, "_mic", None)
        self._mic = self._source = None
        if mic is not None:
            def _close():
                try:
                    mic.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            threading.Thread(target=_close, daemon=True).start()

    def _flush_input(self, source) -> None:
        """Drop audio that queued up while the kiosk was speaking (avoids hearing itself)."""
        try:
            stream = source.stream.pyaudio_stream
            available = stream.get_read_available()
            if available > 0:
                stream.read(available, exception_on_overflow=False)
        except Exception:  # noqa: BLE001
            pass

    def calibrate(self, duration: float = None) -> None:
        """Measure room noise so the end of a reply (silence) is detected reliably.

        Call only when nobody is talking (e.g. right before the greeting). The threshold
        never drops below MIC_ENERGY_THRESHOLD.
        """
        duration = SystemConfig.MIC_CALIBRATE_SECONDS if duration is None else duration
        try:
            source = self._open_source()
            self._flush_input(source)
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self.recognizer.energy_threshold = max(self.recognizer.energy_threshold,
                                                   SystemConfig.MIC_ENERGY_THRESHOLD)
            logger.info(f"Mic calibrated: energy threshold {self.recognizer.energy_threshold:.0f}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Mic calibration skipped: {e}")

    def _detect_th_en(self, audio_float32) -> tuple:
        """Choose Thai or English (only these two) for this reply. (None, 0) = no speech found."""
        # Find speech first with Silero VAD (detect_language() does not accept a dict of VAD
        # options, and with no speech it would run on an empty array).
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            chunks = get_speech_timestamps(audio_float32, VadOptions(min_silence_duration_ms=500))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"VAD unavailable for language detection: {e}")
            chunks = [{"start": 0, "end": len(audio_float32)}]
        if not chunks:
            return None, 0.0                     # no speech at all
        speech = np.concatenate([audio_float32[c["start"]:c["end"]] for c in chunks])
        try:
            _, _, all_probs = self.whisper_model.detect_language(audio=speech, vad_filter=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Language detection failed, assuming Thai: {e}")
            return "th", 0.0
        probs = dict(all_probs)
        th, en = probs.get("th", 0.0), probs.get("en", 0.0)
        code = "th" if th >= en else "en"
        return code, max(th, en) / (th + en + 1e-9)

    def _init_faster_whisper(self):
        """Loads faster-whisper model on CPU using multi-threaded acceleration."""
        try:
            from faster_whisper import WhisperModel
            model_name = getattr(SystemConfig, "FASTER_WHISPER_MODEL", "medium")
            device_type = getattr(SystemConfig, "FASTER_WHISPER_DEVICE", "cpu")
            compute_type = getattr(SystemConfig, "FASTER_WHISPER_COMPUTE", "int8")
            threads = getattr(SystemConfig, "FASTER_WHISPER_CPU_THREADS", 4)

            logger.info(f"Loading Fast-Whisper [{model_name} | {compute_type} | Threads={threads}] on {device_type.upper()}...")
            self.whisper_model = WhisperModel(
                model_name,
                device=device_type,
                compute_type=compute_type,
                cpu_threads=threads
            )
            logger.info(f"Fast-Whisper [{model_name}] armed with multi-threaded CPU acceleration.")
        except Exception as e:
            logger.warning(f"faster-whisper initialization issue ({e}). Fallback armed.")
            self.whisper_model = None

    def warmup(self):
        """Pre-allocates CTranslate2 memory buffers on startup."""
        if self.whisper_model is not None:
            try:
                dummy_pcm = np.zeros(1600, dtype=np.float32)
                self.whisper_model.transcribe(dummy_pcm, beam_size=1)
                logger.info("Fast-Whisper memory buffers pre-warmed.")
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def _play_audio_file(self, filepath: str, block: bool = True):
        """Plays audio using native CoreAudio (afplay) on macOS, or Pygame on Windows."""
        if not os.path.exists(filepath):
            logger.warning(f"Audio file not found at: {filepath}")
            return

        def _worker():
            with self.audio_lock:
                self._is_speaking = True
                try:
                    if IS_MAC:
                        # Native macOS CoreAudio execution (Zero SDL2 collisions)
                        subprocess.run(["/usr/bin/afplay", filepath], check=True)
                    elif not _MIXER_READY:
                        logger.warning("No audio output device: prompt not played.")
                    else:
                        # Windows / Linux
                        pygame.mixer.music.load(filepath)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.04)
                        pygame.mixer.music.unload()
                except Exception as e:
                    logger.error(f"Audio Playback Failure: {e}")
                finally:
                    self._is_speaking = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        if block:
            thread.join()

    # -------------------------------------------------------------------------
    # ON-TOPIC MODE: Instant Playback from Local Template Cache (.mp3)
    # -------------------------------------------------------------------------
    def play_template(self, template_key: str, lang: str = "th", block: bool = True):
        """Plays pre-rendered audio templates directly from disk with zero latency."""
        filename = SystemConfig.TEMPLATES.get(lang, {}).get(template_key)
        if not filename:
            logger.warning(f"Template key '{template_key}' missing for language '{lang}'.")
            return

        filepath = os.path.join(SystemConfig.AUDIO_TEMPLATE_DIR, filename)
        logger.info(f"[Audio Cache]: Playing '{template_key}' ({lang}) -> {filename}")
        self._play_audio_file(filepath, block=block)

    # -------------------------------------------------------------------------
    # OFF-TOPIC MODE: gTTS receives only fixed, non-personalized text
    # -------------------------------------------------------------------------
    def can_speak(self, lang: str) -> bool:
        return lang in getattr(self, "local_voices", {})

    def speak_text(self, text: str, lang: str = "th", block: bool = True) -> bool:
        """Speak any text with the OFFLINE system voice. Returns False if no voice is available.

        Nothing leaves the device: macOS `say` / Windows SAPI synthesise locally.
        Patient-specific or LLM-generated text must only ever go through this method,
        never through gTTS.
        """
        text = (text or "").strip()
        if not text or not self.can_speak(lang):
            return False
        voice = self.local_voices[lang]

        def _worker():
            with self.audio_lock:
                self._is_speaking = True
                try:
                    if IS_WINDOWS:
                        plat.speak_windows(text, voice, SystemConfig.LOCAL_TTS_RATE)
                    else:
                        subprocess.run(["/usr/bin/say", "-v", voice, "-r", str(SystemConfig.LOCAL_TTS_RATE),
                                        "--", text], check=True, timeout=60)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Offline TTS failure: {e}")
                finally:
                    self._is_speaking = False

        logger.info(f"[Offline TTS {voice}]: \"{text}\"")
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        if block:
            thread.join()
        return True

    def speak_off_topic(self, lang: str = "th", block: bool = True):
        """Speak a fixed safe off-topic reply (offline voice first; gTTS only gets this fixed text)."""
        self.speak_fixed(safe_off_topic_text(lang), lang=lang, block=block)

    def speak_fixed(self, text: str, lang: str = "th", block: bool = True) -> str:
        """Speak one of the kiosk's FIXED sentences (no patient data).

        Order on macOS: offline voice -> pre-rendered file -> gTTS -> generic recording.
        Order elsewhere: pre-rendered file -> offline voice -> gTTS -> generic recording
        (the pre-rendered Thai sounds like the other recordings; a Windows Thai voice
        needs an extra speech pack). gTTS is allowed only because the text is fixed and
        pre-approved. Returns "say", "file", "gtts" or "template".
        """
        fixed_file = os.path.join(SystemConfig.FIXED_AUDIO_DIR, fixed_audio_filename(lang, text))
        if IS_MAC and self.speak_text(text, lang=lang, block=block):
            return "say"
        if os.path.isfile(fixed_file):
            logger.info(f"[Audio Cache fixed]: \"{text}\"")
            self._play_audio_file(fixed_file, block=block)
            return "file"
        if not IS_MAC and self.speak_text(text, lang=lang, block=block):
            return "say"
        if not is_safe_fixed_text(text):
            logger.error("Refusing to send non-approved text to gTTS; playing the generic recording.")
            self.play_template("inquiry", lang=lang, block=True)
            return "template"
        logger.info(f"[gTTS fixed text] ({lang}): \"{text}\"")
        temp_path = None
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
            tts.save(temp_path)
            self._play_audio_file(temp_path, block=True)
            return "gtts"
        except Exception as e:  # noqa: BLE001  (offline)
            logger.warning(f"gTTS unavailable; using the generic recording: {e}")
            self.play_template("inquiry", lang=lang, block=True)
            return "template"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # FAST SPEECH INGESTION: Dynamic Auto-Detection & Clean Decoding
    # -------------------------------------------------------------------------
    def listen(self, language: str = "th") -> str:
        """Record one reply and transcribe it with Faster-Whisper.

        ``language``: "th" / "en" forces that language; "auto" chooses between Thai and
        English only (sets ``self.last_language``), and skips Whisper when there is no speech.
        """
        transcript = ""
        self.last_language = (None, 0.0)
        if self.whisper_model is None and not SystemConfig.ALLOW_CLOUD_SPEECH_RECOGNITION:
            # Without this check every reply would silently look like "silence".
            raise AudioDeviceError(
                "Local Faster-Whisper is unavailable; cloud speech recognition is disabled. "
                "Install requirements-kiosk.txt or explicitly set ALLOW_CLOUD_SPEECH_RECOGNITION=1."
            )
        source = self._open_source()
        self._flush_input(source)
        logger.info(f"Mic Listening [pause={self.recognizer.pause_threshold}s, "
                    f"energy>{self.recognizer.energy_threshold:.0f}]...")
        try:
            t_rec = time.time()
            audio = self.recognizer.listen(
                source,
                timeout=SystemConfig.SPEECH_TIMEOUT,
                phrase_time_limit=SystemConfig.SPEECH_PHRASE_LIMIT
            )
            rec_seconds = time.time() - t_rec
            audio_seconds = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
            if audio_seconds >= SystemConfig.SPEECH_PHRASE_LIMIT - 0.5:
                logger.warning(f"Reply hit the {SystemConfig.SPEECH_PHRASE_LIMIT:.0f} s limit: the room may be "
                               f"too noisy to detect the end of speech (energy threshold "
                               f"{self.recognizer.energy_threshold:.0f}). Try MIC_ENERGY_THRESHOLD=600.")
        except sr.WaitTimeoutError:
            logger.info("Silence timeout: No speech detected.")
            return ""
        except OSError as e:                      # device unplugged / CoreAudio error
            self._close_source()
            raise AudioDeviceError(f"Microphone read failed: {e}") from e

        try:
            if self.whisper_model is not None:
                raw_pcm = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_float32 = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
                max_amp = np.max(np.abs(audio_float32))
                if max_amp > 0.01:                 # volume normalisation
                    audio_float32 = (audio_float32 / (max_amp + 1e-6)) * 0.9

                t_dec = time.time()
                if language in ("auto", None, ""):
                    lang_code, conf = self._detect_th_en(audio_float32)
                    self.last_language = (lang_code, conf)
                    if lang_code is None:
                        logger.info(f"[STT] waited+recorded {rec_seconds:.1f} s ({audio_seconds:.1f} s audio), "
                                    f"no speech found (skipped Whisper)")
                        return ""
                else:
                    lang_code = "th" if language.startswith("th") else "en"
                    self.last_language = (lang_code, 1.0)

                # Bounded decoding: one temperature (no 6-step retry ladder), no conditioning on
                # earlier text and a token cap, so noise cannot start a long hallucination loop.
                segments, _ = self.whisper_model.transcribe(
                    audio_float32,
                    beam_size=2,
                    language=lang_code,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    temperature=0.0,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    max_new_tokens=SystemConfig.WHISPER_MAX_NEW_TOKENS,
                )
                transcript = " ".join([seg.text for seg in segments]).strip()
                logger.info(f"[STT] waited+recorded {rec_seconds:.1f} s ({audio_seconds:.1f} s audio), "
                            f"Whisper ({lang_code}) decode {time.time() - t_dec:.1f} s, {len(transcript)} chars")
            elif SystemConfig.ALLOW_CLOUD_SPEECH_RECOGNITION:
                stt_lang = "en-US" if str(language).startswith("en") else "th-TH"
                transcript = self.recognizer.recognize_google(audio, language=stt_lang)
            logger.debug(f"Decoded Speech: \"{transcript}\"")
        except sr.UnknownValueError:
            logger.info("Audio captured but speech was unintelligible.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Speech decoding failed: {e}")

        return transcript.strip()

    # -------------------------------------------------------------------------
    # WAKE-WORD DETECTION
    # -------------------------------------------------------------------------
    @staticmethod
    def detect_wake_word(utterance: str) -> tuple[bool, str]:
        """Detects wake words in ambient listening."""
        return find_wake_word(utterance, SystemConfig.WAKE_WORDS_TH, SystemConfig.WAKE_WORDS_EN)

    def release(self):
        self._close_source()
        if not IS_MAC and _MIXER_READY:
            pygame.mixer.quit()