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


def is_safe_fixed_text(text: str) -> bool:
    """True only for the kiosk's own fixed sentences, which may go to an external TTS service."""
    from .config import SystemConfig
    from .triage.intake import all_fixed_texts

    allowed = set(SAFE_OFF_TOPIC_TEXT.values()) | all_fixed_texts()
    for lang in SystemConfig.SPOKEN_PROMPTS.values():
        allowed.update(lang.values())
    return text in allowed
