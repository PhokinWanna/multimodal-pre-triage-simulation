"""Turn system values into the words and colours a triage nurse expects.

Everything shown on the main screens goes through this module, so no internal
term (audit_mode, SCENARIO_A_CONFIRMED, chest_clutching ...) reaches the nurse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------- triage levels
# ESI number (what the kiosk computes) -> MOPH ED Triage name used in Thai EDs.
# Level codes: Weerakhama et al. (2019). Colours follow common Thai practice
# (red, pink, yellow, green, white) but are NOT yet confirmed against the official
# MOPH guideline; they live only in the CSS variables --lvl-1 ... --lvl-5.


@dataclass(frozen=True)
class Level:
    number: int
    name: str
    code: str


LEVELS = {
    1: Level(1, "Resuscitation", "R"),
    2: Level(2, "Emergency", "E"),
    3: Level(3, "Urgency", "U"),
    4: Level(4, "Semi-urgency", "SU"),
    5: Level(5, "Non-urgency", "NU"),
}


def level(n: Optional[int]) -> Optional[Level]:
    return LEVELS.get(n) if n else None


# ---------------------------------------------------------------- kiosk values
EXIT_REASONS = {
    "NORMAL_COMPLETE": "Interview completed",
    "EMERGENCY_INTERRUPT": "Stopped early: emergency signs",
    "AMS_3_STRIKES": "Stopped: possible confusion (3 unclear answers)",
    "ABANDONED_TIMEOUT": "Patient left the kiosk",
    "USER_CANCELLED": "Patient cancelled at the kiosk",
    "HUMAN_NURSE_SUMMONED": "Patient asked for a nurse",
    "PDPA_DENIED": "Patient declined consent",
    "SYSTEM_FAULT": "Stopped: kiosk hardware/software fault",
}

ASSISTANCE_REASONS = {
    "consent_refused": "Declined consent",
    "consent_unclear": "Consent answer unclear - please ask in person",
    "kiosk_fault": "Kiosk fault (microphone / speech / software) - check the kiosk",
}


def assistance_reason(value: Optional[str]) -> str:
    return ASSISTANCE_REASONS.get(value or "", value or "Unknown")

VISUAL_SIGNS = {
    "chest_clutching": "Holding chest (3 s or longer)",
    "abdominal_clutching": "Holding abdomen (3 s or longer)",
    "head_clutching": "Holding head (3 s or longer)",
}

SKIN_STATUS = {
    "NORMAL": "No facial redness detected",
    "RED_FLUSHING": "Face appears flushed (red)",   # value the kiosk vision engine sends
    "FLUSHING": "Face appears flushed (red)",       # older/demo spelling
    "UNKNOWN": "Face colour could not be checked",
}


def is_flushed(skin_status: Optional[str]) -> bool:
    return skin_status in ("RED_FLUSHING", "FLUSHING")

SPEAKERS = {
    "patient": "Patient",
    "pdpa_response": "Patient (consent)",
    "kiosk_proactive": "Kiosk",
    "kiosk_agent": "Kiosk",
    "kiosk_action": "Kiosk",
}

SCENARIOS = {
    "SCENARIO_A_CONFIRMED": "Spoken complaint and camera signs agree",
    "SCENARIO_B_SUBJECTIVE_ONLY": "Based on the spoken complaint only",
    "SCENARIO_C_CLINICAL_CONFLICT": "Spoken complaint and camera signs disagree: check carefully",
}

LANGUAGES = {"th": "Thai", "en": "English"}

DISPOSITIONS = {
    "ER_RESUS": "ER: resuscitation room now",
    "ER_TREATMENT": "ER: treatment area",
    "FAST_TRACK": "Fast track",
    "OPD_QUEUE": "OPD queue",
    "OTHER": "Other (see note)",
}

SERVICE_POINTS = {"ED": "Emergency triage", "OPD": "OPD screening"}


def exit_reason(value: Optional[str]) -> str:
    return EXIT_REASONS.get(value or "", value or "Unknown")


def speaker(role: Optional[str]) -> str:
    return SPEAKERS.get(role or "", "Kiosk")


def is_patient(role: Optional[str]) -> bool:
    return (role or "").startswith("patient") or role == "pdpa_response"


def chief_complaint(conversation: list[dict]) -> Optional[str]:
    """First thing the patient said after giving consent."""
    for turn in conversation or []:
        if turn.get("role") == "patient" and (turn.get("text") or "").strip():
            return turn["text"].strip()
    return None


def is_fallback(audit_mode: Optional[str]) -> bool:
    """True when the suggestion came from the rule-based fallback, not the validated LLM path."""
    return bool(audit_mode) and audit_mode != "LLM_INFERENCE_VALIDATED"


# ---------------------------------------------------------------- vital signs
# Normal-range entry limits (reject typing mistakes) and the adult ESI
# "danger-zone" values from the ESI Implementation Handbook, 5th ed. (2023).
# NOTE: danger-zone values were read from the handbook's figure by automated
# extraction and are pending a visual check (see research doc, fact check #4).
VITAL_LIMITS = {
    "heart_rate": (20, 250),
    "resp_rate": (4, 60),
    "systolic_bp": (50, 260),
    "diastolic_bp": (30, 160),
    "temperature_c": (30.0, 43.0),
    "spo2": (50, 100),
    "pain_score": (0, 10),
}

DANGER_ZONE = {"heart_rate": 100, "resp_rate": 20, "spo2": 92}


def vital_flags(case) -> list[str]:
    flags = []
    if case.heart_rate is not None and case.heart_rate > DANGER_ZONE["heart_rate"]:
        flags.append(f"Heart rate {case.heart_rate} /min is above {DANGER_ZONE['heart_rate']}")
    if case.resp_rate is not None and case.resp_rate > DANGER_ZONE["resp_rate"]:
        flags.append(f"Respiratory rate {case.resp_rate} /min is above {DANGER_ZONE['resp_rate']}")
    if case.spo2 is not None and case.spo2 < DANGER_ZONE["spo2"]:
        flags.append(f"SpO₂ {case.spo2}% is below {DANGER_ZONE['spo2']}%")
    return flags


def has_vitals(case) -> bool:
    return any(getattr(case, k) is not None for k in VITAL_LIMITS)


# ---------------------------------------------------------------- time
def to_local(dt: Optional[datetime], tz: str) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz))


def clock(dt: Optional[datetime], tz: str) -> str:
    local = to_local(dt, tz)
    return local.strftime("%H:%M") if local else "—"


def date_time(dt: Optional[datetime], tz: str) -> str:
    local = to_local(dt, tz)
    return local.strftime("%d %b %Y, %H:%M") if local else "—"


def waited_minutes(received_at: datetime, now: Optional[datetime] = None) -> int:
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return max(0, int((now - received_at).total_seconds() // 60))


def utc_iso(dt: Optional[datetime]) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat() if dt else ""
