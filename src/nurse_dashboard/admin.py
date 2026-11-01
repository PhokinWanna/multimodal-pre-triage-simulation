"""Admin pages: staff accounts and the audit log."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, func, select

from . import audit, labels
from .auth import require_admin
from .db import get_session
from .models import AuditLog, Role, User
from .security import PasswordPolicyError, create_user, set_password
from .templating import templates

router = APIRouter(prefix="/admin", include_in_schema=False)

AUDIT_PAGE = 100

# Plain-language names for audit actions shown to admins.
ACTIONS = {
    "login": "Signed in", "logout": "Signed out", "login_failed": "Wrong password",
    "login_blocked": "Sign-in blocked (too many attempts)", "user_created": "Account created",
    "password_changed": "Password changed", "user_activated": "Account activated",
    "user_deactivated": "Account deactivated", "case_received": "Case received from kiosk",
    "duplicate_delivery": "Kiosk resent a case (ignored)", "assistance_event_received": "Nurse-assistance alert",
    "assistance_attended": "Alert attended", "case_viewed": "Case opened", "case_acknowledged": "Marked as seen",
    "nurse_data_saved": "Nurse data saved", "case_reviewed": "Decision made",
    "case_review_corrected": "Decision changed", "history_searched": "History searched",
    "data_exported": "Data exported (CSV)", "retention_purge": "Old data deleted (retention)",
    "kiosk_rejected": "Kiosk request rejected (wrong key)",
}


def _ctx(request: Request, user: User, **extra) -> dict:
    s = request.app.state.settings
    return {"user": user, "L": labels, "tz": s.timezone, "settings": s, "ACTIONS": ACTIONS,
            "service_point": labels.SERVICE_POINTS.get(s.service_point, s.service_point), **extra}


def _render_users(request, session, user, message=None, error=None, form=None, status_code=200):
    users = session.exec(select(User).order_by(User.is_active.desc(), User.username)).all()
    return templates.TemplateResponse(request, "admin_users.html", _ctx(
        request, user, active="admin", users=users, message=message, error=error, form=form or {},
    ), status_code=status_code)


def _active_admins(session: Session) -> int:
    return session.exec(select(func.count()).select_from(User)
                        .where(User.role == Role.ADMIN, User.is_active == True)).one()  # noqa: E712


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request, user: User = Depends(require_admin), session: Session = Depends(get_session),
               message: Optional[str] = None):
    return _render_users(request, session, user, message=message)


@router.post("/users", response_class=HTMLResponse)
async def add_user(request: Request, user: User = Depends(require_admin), session: Session = Depends(get_session)):
    form = await request.form()
    role = Role.ADMIN if form.get("role") == "admin" else Role.NURSE
    try:
        new = create_user(session, form.get("username", ""), form.get("password", ""), role=role,
                          display_name=form.get("display_name", ""), actor=user.username)
    except (ValueError, PasswordPolicyError) as e:
        session.rollback()
        return _render_users(request, session, user, error=str(e), form=dict(form), status_code=422)
    return RedirectResponse(f"/admin/users?message=Account+'{new.username}'+created", status_code=303)


@router.post("/users/{user_id}/active")
async def set_active(user_id: int, request: Request, user: User = Depends(require_admin),
                     session: Session = Depends(get_session)):
    target = session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404)
    make_active = (await request.form()).get("active") == "1"
    if not make_active:
        if target.id == user.id:
            return _render_users(request, session, user, error="You cannot deactivate your own account.",
                                 status_code=422)
        if target.role == Role.ADMIN and _active_admins(session) <= 1:
            return _render_users(request, session, user, error="At least one active admin is required.",
                                 status_code=422)
    target.is_active = make_active
    session.add(target)
    audit.record(session, user.username, "user_activated" if make_active else "user_deactivated",
                 "user", target.username)
    session.commit()
    state = "activated" if make_active else "deactivated"
    return RedirectResponse(f"/admin/users?message=Account+'{target.username}'+{state}", status_code=303)


@router.post("/users/{user_id}/password", response_class=HTMLResponse)
async def reset_password(user_id: int, request: Request, user: User = Depends(require_admin),
                         session: Session = Depends(get_session)):
    target = session.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404)
    try:
        set_password(session, target, (await request.form()).get("password", ""), actor=user.username)
    except PasswordPolicyError as e:
        session.rollback()
        return _render_users(request, session, user, error=f"{target.username}: {e}", status_code=422)
    return RedirectResponse(f"/admin/users?message=Password+reset+for+'{target.username}'", status_code=303)


@router.get("/audit", response_class=HTMLResponse)
def audit_page(request: Request, user: User = Depends(require_admin), session: Session = Depends(get_session),
               action: Optional[str] = None, actor: Optional[str] = None, page: int = 1):
    q = select(AuditLog)
    if action:
        q = q.where(AuditLog.action == action)
    if actor:
        q = q.where(AuditLog.actor == actor.strip().lower())
    page = max(1, page)
    rows = session.exec(q.order_by(AuditLog.id.desc()).offset((page - 1) * AUDIT_PAGE).limit(AUDIT_PAGE + 1)).all()
    return templates.TemplateResponse(request, "admin_audit.html", _ctx(
        request, user, active="admin", rows=rows[:AUDIT_PAGE], has_next=len(rows) > AUDIT_PAGE, page=page,
        f={"action": action or "", "actor": actor or ""},
    ))
