"""Database tables.

Times are stored as naive UTC datetimes (same behaviour on SQLite and PostgreSQL);
the web pages convert them to local time for display.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CaseStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REVIEWED = "REVIEWED"
    CLOSED = "CLOSED"


class EventStatus(str, Enum):
    OPEN = "OPEN"
    ATTENDED = "ATTENDED"


class ReviewDecision(str, Enum):
    CONFIRMED = "CONFIRMED"
    REVISED = "REVISED"


class Role(str, Enum):
    NURSE = "nurse"
    ADMIN = "admin"


class User(SQLModel, table=True):
    __tablename__ = "users"  # "user" is a reserved word in PostgreSQL

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=64)
    display_name: str = Field(default="", max_length=128)
    password_hash: str
    role: Role = Field(default=Role.NURSE)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)


class Case(SQLModel, table=True):
    __tablename__ = "cases"

    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: str = Field(index=True, unique=True, max_length=64)
    received_at: datetime = Field(default_factory=utcnow, index=True)
    delivery_count: int = 1                     # >1 means the kiosk retried

    # --- from the kiosk ---
    start_time: Optional[str] = None            # kiosk local time, as sent
    language: Optional[str] = Field(default=None, max_length=8)
    pdpa_consented: bool = False
    exit_reason: Optional[str] = Field(default=None, max_length=48)
    chest_clutching: bool = False
    abdominal_clutching: bool = False
    head_clutching: bool = False
    skin_status: Optional[str] = Field(default=None, max_length=32)
    evidence_path: Optional[str] = None
    conversation: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    # --- AI verdict (never overwritten by staff) ---
    ai_esi: Optional[int] = Field(default=None, index=True)
    ai_urgency: Optional[str] = None
    scenario_type: Optional[str] = None
    subjective: Optional[str] = Field(default=None, sa_column=Column(Text))
    objective: Optional[str] = Field(default=None, sa_column=Column(Text))
    assessment: Optional[str] = Field(default=None, sa_column=Column(Text))
    plan: Optional[str] = Field(default=None, sa_column=Column(Text))
    audit_mode: Optional[str] = None

    # --- data quality ---
    needs_check: bool = Field(default=False, index=True)
    check_notes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    raw_payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # --- staff workflow ---
    status: CaseStatus = Field(default=CaseStatus.NEW, index=True)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[int] = Field(default=None, foreign_key="users.id")
    final_esi: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class AssistanceEvent(SQLModel, table=True):
    """Anonymous 'patient needs a nurse at the kiosk' alert (e.g. consent refused)."""

    __tablename__ = "assistance_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: Optional[str] = Field(default=None, index=True, unique=True, max_length=64)
    reason: str = Field(default="unspecified", max_length=64)
    received_at: datetime = Field(default_factory=utcnow, index=True)
    status: EventStatus = Field(default=EventStatus.OPEN, index=True)
    attended_at: Optional[datetime] = None
    attended_by: Optional[int] = Field(default=None, foreign_key="users.id")


class Review(SQLModel, table=True):
    """One human-in-the-loop decision on a case (history is kept, never edited)."""

    __tablename__ = "reviews"

    id: Optional[int] = Field(default=None, primary_key=True)
    case_pk: int = Field(foreign_key="cases.id", index=True)
    reviewer_id: Optional[int] = Field(default=None, foreign_key="users.id")
    decision: ReviewDecision
    ai_esi: Optional[int] = None
    final_esi: int
    reason: Optional[str] = Field(default=None, sa_column=Column(Text))
    reviewed_at: datetime = Field(default_factory=utcnow)


class AuditLog(SQLModel, table=True):
    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    at: datetime = Field(default_factory=utcnow, index=True)
    actor: str = Field(max_length=64)           # username, "kiosk" or "system"
    action: str = Field(max_length=64, index=True)
    target_type: Optional[str] = Field(default=None, max_length=32)
    target_id: Optional[str] = Field(default=None, max_length=64)
    detail: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
