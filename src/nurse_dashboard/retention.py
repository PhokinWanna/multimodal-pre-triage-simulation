"""PDPA data retention: delete old cases, their reviews, alerts and kiosk photos.

The audit log is kept (it holds actions and case numbers, never personal values),
so it remains possible to show who did what, even after the case itself is gone.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import delete
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from . import audit
from .ingest import safe_evidence_path
from .models import AssistanceEvent, Case, Review, utcnow
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class PurgeResult:
    cases: int = 0
    reviews: int = 0
    events: int = 0
    photos: int = 0


def purge_old_data(engine: Engine, settings: Settings, actor: str = "system") -> PurgeResult:
    result = PurgeResult()
    if settings.retention_days <= 0:
        return result
    cutoff = utcnow() - timedelta(days=settings.retention_days)

    with Session(engine) as session:
        old_cases = session.exec(select(Case).where(Case.received_at < cutoff)).all()
        ids = [c.id for c in old_cases]
        for case in old_cases:
            path = safe_evidence_path(case.evidence_path, settings.evidence_dir)
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    result.photos += 1
                except OSError as e:
                    logger.warning(f"Could not delete kiosk photo for {case.case_id}: {e}")
        if ids:
            result.reviews = session.exec(delete(Review).where(Review.case_pk.in_(ids))).rowcount or 0
            result.cases = session.exec(delete(Case).where(Case.id.in_(ids))).rowcount or 0
        result.events = session.exec(
            delete(AssistanceEvent).where(AssistanceEvent.received_at < cutoff)
        ).rowcount or 0
        if result.cases or result.events:
            audit.record(session, actor, "retention_purge", retention_days=settings.retention_days,
                         cases=result.cases, reviews=result.reviews, events=result.events, photos=result.photos)
        session.commit()

    if result.cases or result.events:
        logger.info(f"Retention purge: removed {result.cases} cases, {result.events} alerts, {result.photos} photos.")
    return result


async def purge_loop(engine: Engine, settings: Settings) -> None:
    """Run the purge at start-up and then every ``purge_interval_hours``."""
    if settings.retention_days <= 0:
        return
    while True:
        try:
            await asyncio.to_thread(purge_old_data, engine, settings)
        except Exception as e:  # never let a purge error stop the dashboard
            logger.error(f"Retention purge failed: {e}")
        await asyncio.sleep(max(60.0, settings.purge_interval_hours * 3600))
