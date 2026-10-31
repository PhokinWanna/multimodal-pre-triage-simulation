"""Dashboard settings, read from environment variables with safe local defaults.

Database selection (the only thing that changes between SQLite and PostgreSQL):
    DASHBOARD_DATABASE_URL=sqlite:///data/dashboard/dashboard.db          (default)
    DASHBOARD_DATABASE_URL=postgresql+psycopg://triage:secret@localhost:5432/triage
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_sqlite_url() -> str:
    db_path = PROJECT_ROOT / "data" / "dashboard" / "dashboard.db"
    return f"sqlite:///{db_path}"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DASHBOARD_DATABASE_URL", _default_sqlite_url()))
    # Only evidence images inside this folder are ever stored or served.
    evidence_dir: str = field(
        default_factory=lambda: os.getenv(
            "DASHBOARD_EVIDENCE_DIR", str(PROJECT_ROOT / "data" / "runtime" / "evidence")
        )
    )
    retention_days: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_RETENTION_DAYS", "30")))
    # Signs the login cookie. If unset, a random key is created once and kept in
    # data/dashboard/secret_key (git-ignored), so logins survive a server restart.
    secret_key: Optional[str] = field(default_factory=lambda: os.getenv("DASHBOARD_SECRET_KEY") or None)
    secret_key_file: str = field(
        default_factory=lambda: os.getenv(
            "DASHBOARD_SECRET_KEY_FILE", str(PROJECT_ROOT / "data" / "dashboard" / "secret_key")
        )
    )
    # A login lasts one long shift at most.
    session_hours: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_SESSION_HOURS", "12")))
    # Set to 1 when the dashboard is served over HTTPS (LAN deployment).
    cookie_secure: bool = field(default_factory=lambda: os.getenv("DASHBOARD_COOKIE_SECURE", "0") == "1")

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


def get_settings() -> Settings:
    return Settings()


def resolve_secret_key(settings: Settings) -> str:
    """Return the configured key, or load/create the persistent one."""
    if settings.secret_key:
        return settings.secret_key
    path = Path(settings.secret_key_file)
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    import secrets

    key = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    try:
        path.chmod(0o600)  # readable by the owner only
    except OSError:
        pass
    return key
