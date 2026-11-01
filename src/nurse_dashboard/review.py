"""Human-in-the-loop review: the nurse confirms or changes the suggested level.

The AI suggestion on the case is never overwritten. Every decision is stored as a
new row in ``reviews`` (so corrections keep their history) and the case keeps a
copy of the latest decision for fast queue and report queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from . import audit, labels
from .auth import require_user
from .db import get_session
from .models import Case, CaseStatus, Review, ReviewDecision, User, utcnow

router = APIRouter(include_in_schema=False)

MAX_REASON = 1000


@dataclass
class DecisionInput:
    final_esi: Optional[int] = None
    disposition: Optional[str] = None
    reason: str = ""
    unidentified_ok: bool = False
    errors: dict[str, str] = field(default_factory=dict)


def parse_decision(form, case: Case) -> DecisionInput:
    d = DecisionInput()
    raw_level = (form.get("final_esi") or "").strip()
    if raw_level.isdigit() and int(raw_level) in labels.LEVELS:
        d.final_esi = int(raw_level)
    else:
        d.errors["final_esi"] = "Choose a triage level"

    disposition = (form.get("disposition") or "").strip()
    if disposition in labels.DISPOSITIONS:
        d.disposition = disposition
    else:
        d.errors["disposition"] = "Choose where the patient goes"

    d.reason = (form.get("reason") or "").strip()
    if len(d.reason) > MAX_REASON:
        d.errors["reason"] = f"Too long (max {MAX_REASON} characters)"
    changed = case.ai_esi is not None and d.final_esi is not None and d.final_esi != case.ai_esi
    if changed and not d.reason:
        d.errors["reason"] = "Give a reason when you change the suggested level"
    if d.disposition == "OTHER" and not d.reason:
        d.errors["reason"] = "Describe where the patient goes"

    d.unidentified_ok = form.get("unidentified_ok") == "1"
    if not (case.hn or case.patient_name) and not d.unidentified_ok:
        d.errors["identity"] = "Record the HN or name first, or tick 'Patient not yet identified'"
    return d


def decision_type(ai_esi: Optional[int], final_esi: int) -> ReviewDecision:
    if ai_esi is None:
        return ReviewDecision.ASSIGNED
    return ReviewDecision.CONFIRMED if ai_esi == final_esi else ReviewDecision.REVISED


def triage_difference(ai_esi: Optional[int], final_esi: Optional[int]) -> Optional[str]:
    """Compare the kiosk with the nurse: lower ESI number = more urgent.

    'under' = the kiosk suggested a LESS urgent level than the nurse chose (under-triage).
    'over'  = the kiosk suggested a MORE urgent level than the nurse chose (over-triage).
    """
    if ai_esi is None or final_esi is None:
        return None
    if ai_esi == final_esi:
        return "same"
    return "under" if ai_esi > final_esi else "over"


def latest_reviews(session: Session, case: Case) -> list[Review]:
    return session.exec(select(Review).where(Review.case_pk == case.id).order_by(Review.id.desc())).all()


@router.post("/cases/{case_pk}/review")
async def submit_review(case_pk: int, request: Request, user: User = Depends(require_user),
                        session: Session = Depends(get_session)):
    case = session.get(Case, case_pk)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    form = await request.form()
    d = parse_decision(form, case)
    if d.errors:
        from .pages import render_case  # local import avoids a circular import
        return render_case(request, session, user, case, decision_errors=d.errors, decision_form=dict(form),
                           status_code=422)

    now = utcnow()
    decision = decision_type(case.ai_esi, d.final_esi)
    session.add(Review(case_pk=case.id, reviewer_id=user.id, decision=decision, ai_esi=case.ai_esi,
                       final_esi=d.final_esi, disposition=d.disposition,
                       patient_identified=bool(case.hn or case.patient_name),
                       reason=d.reason or None, reviewed_at=now))
    first_review = case.status != CaseStatus.REVIEWED
    if case.status == CaseStatus.NEW:
        case.acknowledged_at, case.acknowledged_by = now, user.id
    case.status = CaseStatus.REVIEWED
    case.final_esi = d.final_esi
    case.disposition = d.disposition
    case.reviewed_at = now
    session.add(case)
    audit.record(session, user.username, "case_reviewed" if first_review else "case_review_corrected",
                 "case", case.case_id, decision=decision.value, ai_esi=case.ai_esi,
                 final_esi=d.final_esi, disposition=d.disposition)
    session.commit()
    return RedirectResponse(f"/cases/{case.id}?reviewed=1", status_code=303)
