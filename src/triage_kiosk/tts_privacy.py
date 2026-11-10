"""Privacy-safe text permitted for external text-to-speech synthesis."""

SAFE_OFF_TOPIC_TEXT = {
    "th": "ขอบคุณที่แบ่งปันค่ะ กลับมาคุยเรื่องอาการของคุณกันนะคะ",
    "en": "Thanks for sharing. Let's return to your symptoms.",
}


def safe_off_topic_text(language: str) -> str:
    """Return fixed, non-personalized text; never pass model or patient text to gTTS."""
    try:
        return SAFE_OFF_TOPIC_TEXT[language]
    except KeyError as e:
        raise ValueError(f"Unsupported TTS language: {language}") from e


def fixed_kiosk_texts() -> list[tuple[str, str]]:
    """Every fixed sentence the kiosk can say, as (language, text). None contains patient data.

    Used to (a) allow these texts - and only these - to go to gTTS, and (b) pre-render them
    as audio files (scripts/audio/generate_templates.py) so they play offline on any OS.
    """
    from .config import SystemConfig
    from .triage.intake import QUESTIONS, REPROMPTS
    from .triage.reasoning import ReasoningEngine

    out = []
    for lang in ("th", "en"):
        texts = [SAFE_OFF_TOPIC_TEXT[lang]]
        texts += list(QUESTIONS[lang].values()) + list(REPROMPTS[lang].values())
        texts += list(SystemConfig.SPOKEN_PROMPTS[lang].values())
        for gesture in ("HEAD_PAIN", "CHEST_PAIN", "ABDOMINAL_PAIN"):
            texts.append(ReasoningEngine.get_proactive_visual_prompt(gesture, lang))
        out += [(lang, t) for t in dict.fromkeys(texts) if t]
    return out


def is_safe_fixed_text(text: str) -> bool:
    """True only for the kiosk's own fixed sentences, which may go to an external TTS service."""
    return any(text == t for _, t in fixed_kiosk_texts())


def fixed_audio_filename(lang: str, text: str) -> str:
    """Stable file name for the pre-rendered audio of one fixed sentence."""
    import hashlib

    return f"{lang}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}.mp3"
