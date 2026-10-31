"""Turn kiosk JSON into database rows.

Design rule: the kiosk's outbox retries ONE record until it gets 200/201, and every
later record waits behind it. So this module never rejects a well-formed JSON object.
Anything unexpected is stored and flagged ``needs_check`` for a nurse/admin to look at.

Pydantic models describe the *expected* contract; they are used to produce readable
"check notes", while the values themselves are extracted defensively.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from . import audit
from .models import AssistanceEvent, Case
from .settings import Settings

ASSISTANCE_EVENT_TYPE = "nurse_assistance_requested"
VERDICT_OPTIONAL_EXITS = {"USER_CANCELLED"}  # kiosk skips the SOAP verdict for these
KNOWN_EXIT_REASONS = {
    "NORMAL_COMPLETE", "EMERGENCY_INTERRUPT", "AMS_3_STRIKES", "PDPA_DENIED",
    "ABANDONED_TIMEOUT", "USER_CANCELLED", "HUMAN_NURSE_SUMMONED",
}
MAX_TEXT = 20_000


# ----------------------------------------------------------------------------
# Expected contract (documentation + check notes)
# ----------------------------------------------------------------------------
class KioskVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")
    esi_level: int = Field(ge=1, le=5)
    urgency: str
    subjective: str
    objective: str
    assessment: str
    plan: str
    audit_mode: str
    scenario_type: Optional[str] = None


class KioskTurn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    text: str


class KioskCase(BaseModel):
    model_config = ConfigDict(extra="allow")
    case_id: str
    start_time: str
    language: Literal["th", "en"]
    pdpa_consented: bool
    exit_reason: str
    conversation: list[KioskTurn]
    cumulative_visual_signs: dict[str, bool]
    facial_skin_status: str
    visual_evidence_path: Optional[str] = None
    triage_verdict: Optional[KioskVerdict] = None


class KioskAssistanceEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_type: Literal["nurse_assistance_requested"]
    reason: str
    event_id: Optional[str] = None


def _notes_from(err: ValidationError) -> list[str]:
    notes = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "payload"
        notes.append(f"{loc}: {e['msg']}")
    return notes


# ----------------------------------------------------------------------------
# Defensive extraction helpers
# ----------------------------------------------------------------------------
def _str(value: Any, limit: int = MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    return str(value)[:limit]


def _esi(value: Any) -> Optional[int]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 5 else None


def safe_evidence_path(path: Any, evidence_dir: str) -> Optional[str]:
    """Return the real path only if it is a .jpg inside ``evidence_dir``."""
    if not isinstance(path, str) or not path:
        return None
    root = os.path.realpath(evidence_dir)
    real = os.path.realpath(path)
    try:
        inside = os.path.commonpath([root, real]) == root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside or not real.lower().endswith((".jpg", ".jpeg")):
        return None
    return real


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
@dataclass
class IngestResult:
    kind: Literal["case", "assistance_event"]
    created: bool          # False = duplicate delivery (already stored)
    identifier: str
    needs_check: bool = False


def ingest_payload(session: Session, payload: dict[str, Any], settings: Settings) -> IngestResult:
    if payload.get("event_type") == ASSISTANCE_EVENT_TYPE:
        return _ingest_assistance_event(session, payload)
    return _ingest_case(session, payload, settings)


def _ingest_assistance_event(session: Session, payload: dict[str, Any]) -> IngestResult:
    event_id = _str(payload.get("event_id"), 64)
    if event_id:
        existing = session.exec(select(AssistanceEvent).where(AssistanceEvent.event_id == event_id)).first()
        if existing:
            audit.record(session, "kiosk", "duplicate_delivery", "assistance_event", event_id)
            session.commit()
            return IngestResult("assistance_event", False, event_id)

    # Store only the minimal, anonymous fields — never anything else that may be sent.
    event = AssistanceEvent(event_id=event_id, reason=_str(payload.get("reason"), 64) or "unspecified")
    session.add(event)
    try:
        session.flush()
    except IntegrityError:  # same event_id arrived twice at the same moment
        session.rollback()
        return IngestResult("assistance_event", False, event_id or "")
    audit.record(session, "kiosk", "assistance_event_received", "assistance_event", str(event.id),
                 reason=event.reason)
    session.commit()
    return IngestResult("assistance_event", True, event_id or str(event.id))


def _ingest_case(session: Session, payload: dict[str, Any], settings: Settings) -> IngestResult:
    notes: list[str] = []
    try:
        KioskCase.model_validate(payload)
    except ValidationError as err:
        notes.extend(_notes_from(err))

    case_id = _str(payload.get("case_id"), 64)
    if not case_id:
        case_id = f"UNKNOWN-{uuid.uuid4().hex[:12]}"
        notes.append("case_id missing: dashboard assigned a temporary id")

    existing = session.exec(select(Case).where(Case.case_id == case_id)).first()
    if existing:
        existing.delivery_count += 1
        audit.record(session, "kiosk", "duplicate_delivery", "case", case_id,
                     delivery_count=existing.delivery_count)
        session.commit()
        return IngestResult("case", False, case_id, existing.needs_check)

    consented = payload.get("pdpa_consented") is True
    exit_reason = _str(payload.get("exit_reason"), 48)
    if exit_reason and exit_reason not in KNOWN_EXIT_REASONS:
        notes.append(f"unknown exit_reason '{exit_reason}'")

    case = Case(case_id=case_id, start_time=_str(payload.get("start_time"), 32),
                language=_str(payload.get("language"), 8), pdpa_consented=consented,
                exit_reason=exit_reason)

    if not consented:
        # PDPA data minimisation: without recorded consent keep only who/when/why.
        notes.append("consent not recorded: personal data discarded by the dashboard")
        case.raw_payload = {k: payload.get(k) for k in ("case_id", "start_time", "language", "exit_reason")}
    else:
        signs = payload.get("cumulative_visual_signs") or {}
        if isinstance(signs, dict):
            case.chest_clutching = signs.get("chest_clutching") is True
            case.abdominal_clutching = signs.get("abdominal_clutching") is True
            case.head_clutching = signs.get("head_clutching") is True
        case.skin_status = _str(payload.get("facial_skin_status"), 32)

        conversation = payload.get("conversation")
        if isinstance(conversation, list):
            case.conversation = [
                {"role": _str(t.get("role"), 32) or "unknown", "text": _str(t.get("text")) or ""}
                for t in conversation if isinstance(t, dict)
            ]

        raw_evidence = payload.get("visual_evidence_path")
        case.evidence_path = safe_evidence_path(raw_evidence, settings.evidence_dir)
        if raw_evidence and case.evidence_path is None:
            notes.append("evidence image outside the allowed folder: not linked")

        verdict = payload.get("triage_verdict")
        if isinstance(verdict, dict):
            case.ai_esi = _esi(verdict.get("esi_level"))
            case.ai_urgency = _str(verdict.get("urgency"), 32)
            case.scenario_type = _str(verdict.get("scenario_type"), 64)
            case.subjective = _str(verdict.get("subjective"))
            case.objective = _str(verdict.get("objective"))
            case.assessment = _str(verdict.get("assessment"))
            case.plan = _str(verdict.get("plan"))
            case.audit_mode = _str(verdict.get("audit_mode"), 64)
        if case.ai_esi is None and exit_reason not in VERDICT_OPTIONAL_EXITS:
            notes.append("no valid AI ESI level: nurse must triage manually")
        case.raw_payload = payload

    # De-duplicate notes, keep order.
    case.check_notes = list(dict.fromkeys(notes))
    case.needs_check = bool(case.check_notes)

    session.add(case)
    try:
        session.flush()
    except IntegrityError:  # the same case_id arrived twice at the same moment
        session.rollback()
        return IngestResult("case", False, case_id)
    audit.record(session, "kiosk", "case_received", "case", case_id,
                 ai_esi=case.ai_esi, needs_check=case.needs_check)
    session.commit()
    return IngestResult("case", True, case_id, case.needs_check)
