"""Audit-log helper: every important action is recorded (who, what, when)."""

from __future__ import annotations

from typing import Any, Optional

from sqlmodel import Session

from .models import AuditLog


def record(
    session: Session,
    actor: str,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    **detail: Any,
) -> None:
    """Add an audit entry to the session; the caller commits."""
    session.add(
        AuditLog(actor=actor, action=action, target_type=target_type, target_id=target_id, detail=detail)
    )
