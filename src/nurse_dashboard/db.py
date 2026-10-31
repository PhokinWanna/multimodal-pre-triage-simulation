"""Database engine and session helpers (works with SQLite and PostgreSQL)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from fastapi import Request
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .settings import Settings


def make_engine(settings: Settings) -> Engine:
    if settings.is_sqlite:
        engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")   # readers do not block the writer
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine
    return create_engine(settings.database_url, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    """Create the database folder (SQLite) and any missing tables."""
    from . import models  # noqa: F401  (registers the tables)

    if engine.url.get_backend_name() == "sqlite":
        db_file = engine.url.database
        if db_file and db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)

    SQLModel.metadata.create_all(engine)


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session
