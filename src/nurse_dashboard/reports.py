"""History search, statistics (kiosk vs nurse) and de-identified CSV export."""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from statistics import median
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import or_
from sqlmodel import Session, select

from . import audit, labels
from .auth import require_admin, require_user
from .db import get_session
from .models import AssistanceEvent, Case, CaseStatus, User
from .review import triage_difference
from .templating import templates

router = APIRouter(include_in_schema=False)

PAGE_SIZE = 50


# ---------------------------------------------------------------- shared filters
def _parse_date(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def utc_bounds(day_from: Optional[date], day_to: Optional[date], tz: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Local calendar days -> naive UTC range [start, end)."""
    zone = ZoneInfo(tz)

    def conv(d: date) -> datetime:
        return datetime.combine(d, time.min, zone).astimezone(timezone.utc).replace(tzinfo=None)

    start = conv(day_from) if day_from else None
    end = conv(day_to + timedelta(days=1)) if day_to else None
    return start, end


def _filtered(session: Session, start, end):
    q = select(Case)
    if start:
        q = q.where(Case.received_at >= start)
    if end:
        q = q.where(Case.received_at < end)
    return q


def minutes_between(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round((b - a).total_seconds() / 60, 1)


def _ctx(request: Request, user: User, **extra) -> dict:
    s = request.app.state.settings
    return {"user": user, "L": labels, "tz": s.timezone,
            "service_point": labels.SERVICE_POINTS.get(s.service_point, s.service_point), **extra}


def _default_range(request: Request, days: int = 7) -> tuple[date, date]:
    today = datetime.now(ZoneInfo(request.app.state.settings.timezone)).date()
    return today - timedelta(days=days - 1), today


# ---------------------------------------------------------------- history
@router.get("/history", response_class=HTMLResponse)
def history(request: Request, user: User = Depends(require_user), session: Session = Depends(get_session),
            date_from: Optional[str] = None, date_to: Optional[str] = None, level: Optional[str] = None,
            status: Optional[str] = None, q: Optional[str] = None, page: int = 1):
    d_from, d_to = _parse_date(date_from), _parse_date(date_to)
    if d_from is None and d_to is None and not q:
        d_from, d_to = _default_range(request)
    start, end = utc_bounds(d_from, d_to, request.app.state.settings.timezone)
    query = _filtered(session, start, end)
    if level and level.isdigit():
        n = int(level)
        query = query.where(or_(Case.final_esi == n, (Case.final_esi.is_(None)) & (Case.ai_esi == n)))
    if status in {s.value for s in CaseStatus}:
        query = query.where(Case.status == CaseStatus(status))
    if q and q.strip():
        term = f"%{q.strip()[:64]}%"
        query = query.where(or_(Case.case_id.ilike(term), Case.hn.ilike(term), Case.patient_name.ilike(term)))
        audit.record(session, user.username, "history_searched", detail_text_length=len(q.strip()))
        session.commit()

    page = max(1, page)
    rows = session.exec(query.order_by(Case.received_at.desc()).offset((page - 1) * PAGE_SIZE)
                        .limit(PAGE_SIZE + 1)).all()
    has_next = len(rows) > PAGE_SIZE
    people = {u.id: u for u in session.exec(select(User)).all()}
    return templates.TemplateResponse(request, "history.html", _ctx(
        request, user, active="history", cases=rows[:PAGE_SIZE], people=people, page=page, has_next=has_next,
        f={"date_from": d_from.isoformat() if d_from else "", "date_to": d_to.isoformat() if d_to else "",
           "level": level or "", "status": status or "", "q": q or ""},
    ))


# ---------------------------------------------------------------- statistics
def compute_stats(cases: list[Case], events: list[AssistanceEvent]) -> dict:
    reviewed = [c for c in cases if c.final_esi is not None]
    with_both = [c for c in reviewed if c.ai_esi is not None]
    diff = Counter(triage_difference(c.ai_esi, c.final_esi) for c in with_both)
    matrix = {(a, n): 0 for a in labels.LEVELS for n in labels.LEVELS}
    for c in with_both:
        matrix[(c.ai_esi, c.final_esi)] += 1
    ack = [m for m in (minutes_between(c.received_at, c.acknowledged_at) for c in cases) if m is not None]
    rev = [m for m in (minutes_between(c.received_at, c.reviewed_at) for c in cases) if m is not None]

    def pct(part: int, whole: int) -> Optional[float]:
        return round(100 * part / whole, 1) if whole else None

    return {
        "total": len(cases),
        "reviewed": len(reviewed),
        "waiting": sum(1 for c in cases if c.status in (CaseStatus.NEW, CaseStatus.ACKNOWLEDGED)),
        "compared": len(with_both),
        "agree": diff["same"], "under": diff["under"], "over": diff["over"],
        "agree_pct": pct(diff["same"], len(with_both)),
        "under_pct": pct(diff["under"], len(with_both)),
        "over_pct": pct(diff["over"], len(with_both)),
        "matrix": matrix,
        "ai_levels": Counter(c.ai_esi for c in cases),
        "final_levels": Counter(c.final_esi for c in reviewed),
        "median_ack": round(median(ack), 1) if ack else None,
        "median_review": round(median(rev), 1) if rev else None,
        "fallback": sum(1 for c in cases if labels.is_fallback(c.audit_mode)),
        "needs_check": sum(1 for c in cases if c.needs_check),
        "no_level": sum(1 for c in cases if c.ai_esi is None),
        "exit_reasons": Counter(c.exit_reason for c in cases).most_common(),
        "dispositions": Counter(c.disposition for c in reviewed if c.disposition).most_common(),
        "assistance": len(events),
    }


@router.get("/stats", response_class=HTMLResponse)
def stats(request: Request, user: User = Depends(require_user), session: Session = Depends(get_session),
          date_from: Optional[str] = None, date_to: Optional[str] = None):
    d_from, d_to = _parse_date(date_from), _parse_date(date_to)
    if d_from is None and d_to is None:
        d_from, d_to = _default_range(request, days=30)
    start, end = utc_bounds(d_from, d_to, request.app.state.settings.timezone)
    cases = session.exec(_filtered(session, start, end)).all()
    ev_q = select(AssistanceEvent)
    if start:
        ev_q = ev_q.where(AssistanceEvent.received_at >= start)
    if end:
        ev_q = ev_q.where(AssistanceEvent.received_at < end)
    events = session.exec(ev_q).all()
    return templates.TemplateResponse(request, "stats.html", _ctx(
        request, user, active="stats", s=compute_stats(cases, events),
        f={"date_from": d_from.isoformat() if d_from else "", "date_to": d_to.isoformat() if d_to else ""},
    ))


# ---------------------------------------------------------------- CSV export (admin only, de-identified)
EXPORT_COLUMNS = [
    "case_id", "received_local", "language", "exit_reason", "chest_clutching", "abdominal_clutching",
    "head_clutching", "skin_status", "ai_esi", "ai_urgency", "scenario_type", "audit_mode", "needs_check",
    "final_esi", "triage_difference", "disposition", "minutes_to_seen", "minutes_to_decision",
    "heart_rate", "resp_rate", "systolic_bp", "diastolic_bp", "temperature_c", "spo2", "pain_score",
    "status",
]


def export_rows(cases: list[Case], tz: str):
    zone = ZoneInfo(tz)
    for c in cases:
        local = c.received_at.replace(tzinfo=timezone.utc).astimezone(zone)
        yield {
            "case_id": c.case_id, "received_local": local.strftime("%Y-%m-%d %H:%M:%S"),
            "language": c.language, "exit_reason": c.exit_reason,
            "chest_clutching": int(c.chest_clutching), "abdominal_clutching": int(c.abdominal_clutching),
            "head_clutching": int(c.head_clutching), "skin_status": c.skin_status,
            "ai_esi": c.ai_esi, "ai_urgency": c.ai_urgency, "scenario_type": c.scenario_type,
            "audit_mode": c.audit_mode, "needs_check": int(c.needs_check), "final_esi": c.final_esi,
            "triage_difference": triage_difference(c.ai_esi, c.final_esi), "disposition": c.disposition,
            "minutes_to_seen": minutes_between(c.received_at, c.acknowledged_at),
            "minutes_to_decision": minutes_between(c.received_at, c.reviewed_at),
            "heart_rate": c.heart_rate, "resp_rate": c.resp_rate, "systolic_bp": c.systolic_bp,
            "diastolic_bp": c.diastolic_bp, "temperature_c": c.temperature_c, "spo2": c.spo2,
            "pain_score": c.pain_score, "status": c.status.value,
        }


@router.get("/export.csv")
def export_csv(request: Request, user: User = Depends(require_admin), session: Session = Depends(get_session),
               date_from: Optional[str] = None, date_to: Optional[str] = None):
    """De-identified export for research: no HN, name, conversation, notes or photo."""
    tz = request.app.state.settings.timezone
    start, end = utc_bounds(_parse_date(date_from), _parse_date(date_to), tz)
    cases = session.exec(_filtered(session, start, end).order_by(Case.received_at)).all()
    audit.record(session, user.username, "data_exported", rows=len(cases), date_from=date_from, date_to=date_to)
    session.commit()

    buf = io.StringIO()
    buf.write("﻿")  # BOM so Excel opens Thai text correctly
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    writer.writerows(export_rows(cases, tz))
    name = f"triage_cases_{date_from or 'all'}_{date_to or 'all'}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": f'attachment; filename="{name}"',
                                      "Cache-Control": "no-store"})
