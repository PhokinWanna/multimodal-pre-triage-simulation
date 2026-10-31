"""FastAPI application factory: kiosk API, login and staff pages."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session
from starlette.middleware.sessions import SessionMiddleware

from . import __version__
from . import auth
from .auth import AdminRequired, LoginRequired, login_redirect, require_user
from .db import get_session, init_db, make_engine
from .ingest import ingest_payload
from .models import User
from .security import LoginGuard
from .settings import Settings, get_settings, resolve_secret_key
from .templating import STATIC_DIR, templates


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(engine)
        yield
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

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        return login_redirect(exc)

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
    def submit_case(payload: Any = Body(...), session: Session = Depends(get_session)) -> JSONResponse:
        """Receive one kiosk record (a full case or an anonymous assistance event).

        * 201 — stored for the first time.
        * 200 — already stored (kiosk retry); safe to delete the spool file.
        * 400 — body is not a JSON object (the kiosk never sends this).
        * 5xx — temporary server/database problem; the kiosk keeps the file and retries.

        This endpoint has no login because the kiosk calls it from the same computer
        (127.0.0.1). A shared API key will be added before any LAN deployment.
        """
        if not isinstance(payload, dict):
            return JSONResponse({"status": "rejected", "detail": "body must be a JSON object"}, status_code=400)
        result = ingest_payload(session, payload, settings)
        body = {"status": "created" if result.created else "duplicate", "kind": result.kind,
                "id": result.identifier, "needs_check": result.needs_check}
        return JSONResponse(body, status_code=201 if result.created else 200)

    # ------------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request, user: User = Depends(require_user)):
        # Replaced by the live queue in step 4.
        return templates.TemplateResponse(
            request, "message.html",
            {"title": "Signed in", "message": "The live queue will appear here (step 4).", "user": user},
        )

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        from fastapi.responses import FileResponse
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    return app


app = create_app()
