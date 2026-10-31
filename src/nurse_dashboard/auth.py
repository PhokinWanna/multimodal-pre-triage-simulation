"""Login, logout and the 'who is logged in' dependencies."""

from __future__ import annotations

from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlmodel import Session

from . import audit
from .db import get_session
from .models import Role, User
from .security import authenticate, normalise_username, session_version
from .templating import templates

router = APIRouter()


class LoginRequired(Exception):
    """Raised by page dependencies; turned into a redirect to /login."""

    def __init__(self, next_url: str = "/"):
        self.next_url = next_url


class AdminRequired(Exception):
    pass


def safe_next(url: Optional[str]) -> str:
    """Only allow redirects to local paths (blocks open-redirect tricks)."""
    if not url or not url.startswith("/") or url.startswith("//") or "\\" in url:
        return "/"
    return url


def current_user_optional(request: Request, session: Session = Depends(get_session)) -> Optional[User]:
    uid = request.session.get("uid")
    if uid is None:
        return None
    user = session.get(User, uid)
    if user is None or not user.is_active or request.session.get("ver") != session_version(user):
        request.session.clear()
        return None
    return user


def require_user(request: Request, user: Optional[User] = Depends(current_user_optional)) -> User:
    if user is None:
        path = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        raise LoginRequired(path)
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != Role.ADMIN:
        raise AdminRequired()
    return user


def login_redirect(exc: LoginRequired) -> RedirectResponse:
    return RedirectResponse(f"/login?next={quote(safe_next(exc.next_url))}", status_code=303)


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
def login_page(request: Request, next: str = "/", user: Optional[User] = Depends(current_user_optional)):
    if user is not None:
        return RedirectResponse(safe_next(next), status_code=303)
    return templates.TemplateResponse(request, "login.html", {"next": safe_next(next), "error": None})


@router.post("/login", include_in_schema=False)
def login_submit(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
    next: str = Form("/"),
    session: Session = Depends(get_session),
) -> Response:
    guard = request.app.state.login_guard
    name = normalise_username(username)[:64]

    def fail(message: str, status: int = 401) -> Response:
        return templates.TemplateResponse(
            request, "login.html", {"next": safe_next(next), "error": message, "username": name},
            status_code=status,
        )

    if guard.is_locked(name):
        audit.record(session, name or "?", "login_blocked", "user", name)
        session.commit()
        return fail("Too many wrong attempts. Please wait 5 minutes and try again.", 429)

    user = authenticate(session, name, password)
    if user is None:
        guard.failed(name)
        audit.record(session, name or "?", "login_failed", "user", name)
        session.commit()
        return fail("Wrong username or password.")

    guard.succeeded(name)
    request.session.clear()
    request.session.update({"uid": user.id, "ver": session_version(user)})
    audit.record(session, user.username, "login", "user", user.username)
    session.commit()
    return RedirectResponse(safe_next(next), status_code=303)


@router.post("/logout", include_in_schema=False)
def logout(request: Request, session: Session = Depends(get_session),
           user: Optional[User] = Depends(current_user_optional)) -> Response:
    if user is not None:
        audit.record(session, user.username, "logout", "user", user.username)
        session.commit()
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
