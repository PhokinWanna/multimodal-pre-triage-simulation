"""Consent-related handoff helpers."""

ALLOWED_HANDOFF_REASONS = {"consent_refused", "consent_unclear", "kiosk_fault"}


def prepare_pdpa_refusal_handoff(case_container: dict, reason: str = "consent_refused") -> dict[str, str]:
    """Discard the unconsented case and return a minimal nurse-assistance event."""
    if reason not in ALLOWED_HANDOFF_REASONS:
        raise ValueError(f"unknown handoff reason: {reason}")
    case_container.clear()
    return {
        "event_type": "nurse_assistance_requested",
        "reason": reason,
    }


def kiosk_fault_event() -> dict[str, str]:
    """Anonymous event telling staff the kiosk itself needs attention (contains no patient data)."""
    return {"event_type": "nurse_assistance_requested", "reason": "kiosk_fault"}
