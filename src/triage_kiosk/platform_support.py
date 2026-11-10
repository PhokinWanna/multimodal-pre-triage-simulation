"""Operating-system differences in one place (macOS and Windows; Linux works as "other").

| Part                    | macOS                         | Windows                                   |
|-------------------------|-------------------------------|-------------------------------------------|
| Pre-recorded prompts    | afplay                        | pygame                                    |
| Offline voice (any text)| `say` (Kanya / Samantha)      | SAPI via pywin32 (Pattara / Zira / David) |
| Camera                  | AVFoundation (OpenCV default) | DirectShow (faster to open than MSMF)     |
| Console                 | UTF-8                         | switched to UTF-8 so Thai logs print      |
"""

from __future__ import annotations

import logging
import platform
import sys

SYSTEM = platform.system()
IS_MAC = SYSTEM == "Darwin"
IS_WINDOWS = SYSTEM == "Windows"
PLATFORM_NAME = "macOS" if IS_MAC else "Windows" if IS_WINDOWS else SYSTEM or "unknown"

logger = logging.getLogger(__name__)


def configure_console_utf8() -> None:
    """Windows consoles may default to a code page that cannot print Thai: use UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001  (not a text stream / already wrapped)
            pass


def open_camera(index: int):
    """Open the webcam with the best backend for this OS; returns an opened cv2.VideoCapture or None."""
    import cv2

    attempts = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")] if IS_WINDOWS else []
    attempts.append((None, "default"))
    for backend, name in attempts:
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            logger.info(f"Camera {index} opened ({name} backend).")
            return cap
        if cap is not None:
            cap.release()
    return None


# ---------------------------------------------------------------- Windows SAPI voices
_TH_LCIDS = {"41e"}
_EN_LCIDS = {"409", "809", "c09", "1009", "1409", "4009", "3409", "1809"}
_ONECORE_VOICES = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"


def _sapi_tokens():
    """All installed SAPI voice tokens, including Windows 10/11 "OneCore" voices.

    The Thai voice (Microsoft Pattara) is a OneCore voice, which classic SAPI does not list
    by default, so the OneCore category is enumerated as well. Must be called on the thread
    that will speak (COM objects belong to one thread).
    """
    import win32com.client  # pywin32

    voice = win32com.client.Dispatch("SAPI.SpVoice")
    tokens = [voice.GetVoices().Item(i) for i in range(voice.GetVoices().Count)]
    try:
        cat = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
        cat.SetId(_ONECORE_VOICES, False)
        found = cat.EnumerateTokens()
        tokens += [found.Item(i) for i in range(found.Count)]
    except Exception as e:  # noqa: BLE001
        logger.debug(f"OneCore voices not available: {e}")
    return voice, tokens


def _token_info(token) -> tuple[str, str]:
    try:
        lang = (token.GetAttribute("Language") or "").lower()
    except Exception:  # noqa: BLE001
        lang = ""
    return token.GetDescription(), lang


def pick_sapi_voice(voices: list[tuple[str, str]], lang: str, wanted: str | None = None) -> str | None:
    """Choose a voice description for 'th' or 'en' from [(description, LCID hex list), ...]."""
    def lcids(s):
        return {p.strip().lstrip("0") or "0" for p in s.split(";") if p.strip()}

    target = _TH_LCIDS if lang == "th" else _EN_LCIDS
    matching = [d for d, l in voices if lcids(l) & target
                or (lang == "th" and "thai" in d.lower()) or (lang == "en" and "english" in d.lower())]
    if wanted:
        exact = [d for d in matching if wanted.lower() in d.lower()]
        if exact:
            return exact[0]
    if lang == "en":
        for name in ("Zira", "David", "Mark", "Hazel", "George"):
            for d in matching:
                if name.lower() in d.lower():
                    return d
    return matching[0] if matching else None


def list_windows_voices() -> list[tuple[str, str]]:
    import pythoncom

    pythoncom.CoInitialize()
    try:
        _, tokens = _sapi_tokens()
        seen, out = set(), []
        for t in tokens:
            d, l = _token_info(t)
            if d not in seen:
                seen.add(d)
                out.append((d, l))
        return out
    finally:
        pythoncom.CoUninitialize()


def speak_windows(text: str, voice_description: str, rate_wpm: int = 175) -> None:
    """Speak synchronously with a SAPI voice (offline; nothing leaves the PC)."""
    import pythoncom

    pythoncom.CoInitialize()
    try:
        voice, tokens = _sapi_tokens()
        for t in tokens:
            if _token_info(t)[0] == voice_description:
                voice.Voice = t
                break
        voice.Rate = max(-10, min(10, round((rate_wpm - 175) / 15)))   # SAPI rate: -10..10, 0 = normal
        voice.Speak(text)
    finally:
        pythoncom.CoUninitialize()
