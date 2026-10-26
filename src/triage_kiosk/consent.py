"""Consent-related handoff helpers."""


def prepare_pdpa_refusal_handoff(case_container: dict) -> dict[str, str]:
    """Discard the unconsented case and return a minimal nurse-assistance event."""
    case_container.clear()
    return {
        "event_type": "nurse_assistance_requested",
        "reason": "consent_refused",
    }
