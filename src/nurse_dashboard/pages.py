"""Staff pages: live queue, case page (triage note), nurse actions, live updates."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import func
from sqlmodel import Session, select

from . import audit, labels
from .auth import require_user
from .db import get_session
from .ingest import safe_evidence_path
from .models import AssistanceEvent, AuditLog, Case, CaseStatus, EventStatus, User, utcnow
from .templating import templates

router = APIRouter(include_in_schema=False)

ACTIVE = (CaseStatus.NEW, CaseStatus.ACKNOWLEDGED)
URGENT_LEVELS = (1, 2)


# ---------------------------------------------------------------- helpers
def _ctx(request: Request, user: User, **extra) -> dict:
    s = request.app.state.settings
    return {"user": user, "L": labels, "tz": s.timezone,
            "service_point": labels.SERVICE_POINTS.get(s.service_point, s.service_point), **extra}


def sort_key(case: Case) -> tuple:
    """Queue order: level first (no level = needs the nurse, sits after level 2), then longest wait."""
    if case.ai_esi is not None:
        rank = float(case.ai_esi)
    elif case.exit_reason == "USER_CANCELLED":
        rank = 5.5
    else:
        rank = 2.5
    return (rank, case.received_at)


def queue_data(session: Session) -> dict:
    cases = sorted(session.exec(select(Case).where(Case.status.in_(ACTIVE))).all(), key=sort_key)
    events = session.exec(
        select(AssistanceEvent).where(AssistanceEvent.status == EventStatus.OPEN)
        .order_by(AssistanceEvent.received_at)
    ).all()
    urgent = [c for c in cases if c.status == CaseStatus.NEW and c.ai_esi in URGENT_LEVELS]
    return {"cases": cases, "events": events, "urgent": urgent}


def change_marker(session: Session) -> str:
    """Changes whenever any case, alert or staff action is recorded."""
    last_audit = session.exec(select(func.max(AuditLog.id))).one() or 0
    return str(last_audit)


def _get_case(session: Session, case_pk: int) -> Case:
    case = session.get(Case, case_pk)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _acknowledge(session: Session, case: Case, user: User) -> bool:
    if case.status != CaseStatus.NEW:
        return False
    case.status = CaseStatus.ACKNOWLEDGED
    case.acknowledged_at = utcnow()
    case.acknowledged_by = user.id
    session.add(case)
    audit.record(session, user.username, "case_acknowledged", "case", case.case_id)
    return True


# ---------------------------------------------------------------- queue
@router.get("/", response_class=HTMLResponse)
def queue_page(request: Request, user: User = Depends(require_user), session: Session = Depends(get_session)):
    return templates.TemplateResponse(request, "queue.html",
                                      _ctx(request, user, active="queue", **queue_data(session)))


@router.get("/queue/fragment", response_class=HTMLResponse)
def queue_fragment(request: Request, user: User = Depends(require_user), session: Session = Depends(get_session)):
    return templates.TemplateResponse(request, "_queue.html", _ctx(request, user, **queue_data(session)))


@router.get("/events")
async def live_events(request: Request, user: User = Depends(require_user)):
    """Server-Sent Events: tells open browsers to refresh when anything changes.

    It checks the database every 2 s, so it also works with PostgreSQL and several
    server processes (no in-memory message bus needed).
    """
    engine = request.app.state.engine

    def marker() -> str:
        with Session(engine) as s:
            return change_marker(s)

    async def stream():
        last = await asyncio.to_thread(marker)
        yield f"event: hello\ndata: {last}\n\n"
        ticks = 0
        while not await request.is_disconnected():
            await asyncio.sleep(2)
            ticks += 1
            current = await asyncio.to_thread(marker)
            if current != last:
                last = current
                yield f"event: changed\ndata: {current}\n\n"
            elif ticks % 10 == 0:
                yield ": keep-alive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- case page
def render_case(request: Request, session: Session, user: User, case: Case, *, errors=None, form=None,
                saved=None, reviewed=None, decision_errors=None, decision_form=None, status_code: int = 200):
    from .review import latest_reviews  # local import avoids a circular import

    reviews = latest_reviews(session, case)
    people = {u.id: u for u in session.exec(select(User)).all()}
    return templates.TemplateResponse(request, "case.html", _ctx(
        request, user, active="queue", case=case, people=people,
        ack_user=people.get(case.acknowledged_by), nurse_user=people.get(case.nurse_data_by),
        reviews=reviews, errors=errors or {}, form=form, saved=saved, reviewed=reviewed,
        decision_errors=decision_errors or {}, decision_form=decision_form,
    ), status_code=status_code)


@router.get("/cases/{case_pk}", response_class=HTMLResponse)
def case_page(case_pk: int, request: Request, user: User = Depends(require_user),
              session: Session = Depends(get_session), saved: Optional[str] = None,
              reviewed: Optional[str] = None):
    case = _get_case(session, case_pk)
    audit.record(session, user.username, "case_viewed", "case", case.case_id)
    session.commit()
    return render_case(request, session, user, case, saved=saved, reviewed=reviewed)


@router.post("/cases/{case_pk}/ack")
def acknowledge(case_pk: int, request: Request, user: User = Depends(require_user),
                session: Session = Depends(get_session)):
    case = _get_case(session, case_pk)
    _acknowledge(session, case, user)
    session.commit()
    back = request.headers.get("referer", "")
    return RedirectResponse(f"/cases/{case.id}" if f"/cases/{case.id}" in back else "/", status_code=303)


@router.post("/cases/{case_pk}/nurse-data", response_class=HTMLResponse)
async def save_nurse_data(case_pk: int, request: Request, user: User = Depends(require_user),
                          session: Session = Depends(get_session)):
    """Save what the nurse completes at the desk: identification, vital signs, background."""
    case = _get_case(session, case_pk)
    form = await request.form()
    errors: dict[str, str] = {}
    values: dict[str, object] = {}

    for key, (low, high) in labels.VITAL_LIMITS.items():
        raw = (form.get(key) or "").strip()
        if raw == "":
            values[key] = None
            continue
        try:
            num = float(raw) if key == "temperature_c" else int(raw)
        except ValueError:
            errors[key] = "Enter a number"
            continue
        if not (low <= num <= high):
            errors[key] = f"Must be between {low} and {high}"
            continue
        values[key] = num

    for key, limit in (("hn", 32), ("patient_name", 128), ("allergies", 2000),
                       ("medications", 2000), ("past_history", 2000), ("nurse_note", 2000)):
        text = (form.get(key) or "").strip()
        if len(text) > limit:
            errors[key] = f"Too long (max {limit} characters)"
        values[key] = text or None

    if errors:
        return render_case(request, session, user, case, errors=errors, form=dict(form), status_code=422)

    changed = [k for k, v in values.items() if getattr(case, k) != v]
    for k, v in values.items():
        setattr(case, k, v)
    if changed:
        case.nurse_data_at = utcnow()
        case.nurse_data_by = user.id
        # Field names only: personal values are not copied into the audit log.
        audit.record(session, user.username, "nurse_data_saved", "case", case.case_id, fields=changed)
    _acknowledge(session, case, user)  # entering data means the nurse has seen the patient
    session.add(case)
    session.commit()
    return RedirectResponse(f"/cases/{case.id}?saved=1#nurse", status_code=303)


@router.get("/cases/{case_pk}/evidence")
def evidence_image(case_pk: int, request: Request, user: User = Depends(require_user),
                   session: Session = Depends(get_session)):
    case = _get_case(session, case_pk)
    settings = request.app.state.settings
    path = safe_evidence_path(case.evidence_path, settings.evidence_dir)  # re-checked on every request
    if path is None or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="No photo")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- assistance alerts
@router.post("/assistance/{event_pk}/attend")
def attend_assistance(event_pk: int, user: User = Depends(require_user), session: Session = Depends(get_session)):
    event = session.get(AssistanceEvent, event_pk)
    if event is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if event.status == EventStatus.OPEN:
        event.status = EventStatus.ATTENDED
        event.attended_at = utcnow()
        event.attended_by = user.id
        session.add(event)
        audit.record(session, user.username, "assistance_attended", "assistance_event", str(event.id))
        session.commit()
    return RedirectResponse("/", status_code=303)
