"""FastAPI application factory: kiosk API, login and staff pages."""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from . import admin, audit, auth, pages, reports, review
from .auth import AdminRequired, LoginRequired, login_redirect
from .db import get_session, init_db, make_engine
from .ingest import ingest_payload
from .retention import purge_loop
from .security import LoginGuard
from .settings import Settings, get_settings, resolve_secret_key
from .templating import STATIC_DIR, templates


logger = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(engine)
        if not settings.kiosk_key:
            logger.warning("DASHBOARD_KIOSK_KEY is not set: the kiosk endpoint accepts any caller. "
                           "This is acceptable only when the dashboard listens on 127.0.0.1.")
        purge_task = asyncio.create_task(purge_loop(engine, settings))
        yield
        purge_task.cancel()
        with suppress(asyncio.CancelledError):
            await purge_task
        engine.dispose()

    app = FastAPI(title="Pre-Triage Nurse Dashboard", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine
    app.state.login_guard = LoginGuard()

    # Signed, HttpOnly login cookie. SameSite=strict stops other websites from
    # submitting forms with the nurse's login (basic CSRF protection).
    app.add_middleware(
        SessionMiddleware,
        secret_key=resolve_secret_key(settings),
        session_cookie="triage_session",
        max_age=settings.session_hours * 3600,
        same_site="strict",
        https_only=settings.cookie_secure,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(auth.router)
    app.include_router(pages.router)
    app.include_router(review.router)
    app.include_router(reports.router)
    app.include_router(admin.router)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        return login_redirect(exc)

    @app.exception_handler(Exception)
    async def _server_error(request: Request, exc: Exception):
        # Log the full error for staff/developers; show nurses a calm page without technical details.
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        if request.url.path.startswith("/api/"):
            return JSONResponse({"status": "error", "detail": "internal server error"}, status_code=500)
        return templates.TemplateResponse(
            request, "message.html",
            {"title": "Something went wrong",
             "message": "The dashboard could not show this page. Please go back to the queue and try again. "
                        "If it keeps happening, tell the admin (the error was recorded in the server log).",
             "user": None}, status_code=500,
        )

    @app.exception_handler(AdminRequired)
    async def _admin_required(request: Request, exc: AdminRequired):
        return templates.TemplateResponse(
            request, "message.html",
            {"title": "Admins only", "message": "This page is only available to admin accounts.",
             "user": None}, status_code=403,
        )

    # ------------------------------------------------------------------ kiosk API
    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__,
                "database": "sqlite" if settings.is_sqlite else "postgresql"}

    @app.post("/api/triage/submit_case", tags=["kiosk"])
    def submit_case(payload: Any = Body(...), session: Session = Depends(get_session),
                    x_kiosk_key: Optional[str] = Header(default=None)) -> JSONResponse:
        """Receive one kiosk record (a full case or an anonymous assistance event).

        * 201 — stored for the first time.
        * 200 — already stored (kiosk retry); safe to delete the spool file.
        * 400 — body is not a JSON object (the kiosk never sends this).
        * 5xx — temporary server/database problem; the kiosk keeps the file and retries.

        * 401 — DASHBOARD_KIOSK_KEY is set and the X-Kiosk-Key header is missing or wrong.

        Staff log in with passwords; the kiosk proves itself with a shared key instead.
        """
        if settings.kiosk_key and not hmac.compare_digest(x_kiosk_key or "", settings.kiosk_key):
            audit.record(session, "kiosk", "kiosk_rejected")
            session.commit()
            return JSONResponse({"status": "rejected", "detail": "invalid kiosk key"}, status_code=401)
        if not isinstance(payload, dict):
            return JSONResponse({"status": "rejected", "detail": "body must be a JSON object"}, status_code=400)
        result = ingest_payload(session, payload, settings)
        body = {"status": "created" if result.created else "duplicate", "kind": result.kind,
                "id": result.identifier, "needs_check": result.needs_check}
        return JSONResponse(body, status_code=201 if result.created else 200)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    return app


app = create_app()
