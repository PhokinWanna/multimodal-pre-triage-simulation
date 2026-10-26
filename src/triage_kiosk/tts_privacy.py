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
