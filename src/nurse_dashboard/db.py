"""Database engine and session helpers (works with SQLite and PostgreSQL)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import logging
from enum import Enum

from fastapi import Request
from sqlalchemy import event, inspect, literal, text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from .settings import Settings

logger = logging.getLogger(__name__)


class SchemaUpgradeError(RuntimeError):
    pass


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

    SQLModel.metadata.create_all(engine)   # creates tables that do not exist yet
    upgrade_schema(engine)                 # adds columns introduced after a table was created


def _model_defaults() -> dict[str, dict[str, object]]:
    """table name -> {column name: Python default} taken from the SQLModel classes."""
    defaults: dict[str, dict[str, object]] = {}
    stack = list(SQLModel.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        table = getattr(cls, "__table__", None)
        if table is None:
            continue
        defaults[table.name] = {name: f.default for name, f in cls.model_fields.items()}
    return defaults


def upgrade_schema(engine: Engine) -> list[str]:
    """Add columns that exist in the models but not yet in the database.

    ``create_all`` only creates missing *tables*; it never changes an existing one.
    Without this step, a database created by an earlier version of the dashboard
    fails with "no such column" (HTTP 500) after an update. Only additive changes
    are handled (new nullable columns, or new columns with a simple default), which
    is all this prototype needs; anything else stops start-up with a clear message.
    Returns the list of "table.column" names that were added.
    """
    from pydantic_core import PydanticUndefined

    inspector = inspect(engine)
    defaults = _model_defaults()
    added: list[str] = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'
                default = defaults.get(table.name, {}).get(column.name, PydanticUndefined)
                if isinstance(default, Enum):
                    default = default.value
                if default is not PydanticUndefined and default is not None and \
                        isinstance(default, (bool, int, float, str)):
                    rendered = literal(default).compile(dialect=engine.dialect,
                                                        compile_kwargs={"literal_binds": True})
                    ddl += f" DEFAULT {rendered}"
                    if not column.nullable:
                        ddl += " NOT NULL"
                elif not column.nullable:
                    raise SchemaUpgradeError(
                        f"Cannot add required column {table.name}.{column.name} automatically. "
                        f"Back up and delete the database, or migrate it by hand."
                    )
                conn.execute(text(ddl))
                added.append(f"{table.name}.{column.name}")
    if added:
        logger.warning(f"Database upgraded: added column(s) {', '.join(added)}")
    return added


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.engine) as session:
        yield session
