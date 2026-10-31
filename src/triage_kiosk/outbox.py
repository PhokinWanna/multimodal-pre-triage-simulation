"""Outbox spooler ("conveyor belt") that delivers kiosk records to the nurse dashboard.

Every record is written to the spool directory *before* it is queued, so a record
survives a dashboard outage. On start-up the worker re-queues any records that are
still on disk from a previous run (for example, if the kiosk was closed while the
dashboard was offline). A record's spool file is removed only after the dashboard
answers 200/201.
"""

import json
import logging
import os
import queue
import threading
import time
import uuid

import requests

from .config import SystemConfig

logger = logging.getLogger(__name__)

NURSE_ASSISTANCE_PREFIX = "nurse-assistance-"


class ConveyorBeltWorker(threading.Thread):
    """Background Outbox Spooler (The Conveyor Belt)."""

    def __init__(self, spool_dir: str, endpoint: str, recover_existing: bool = True):
        super().__init__(daemon=True)
        self.spool_dir = spool_dir
        self.endpoint = endpoint
        self.queue = queue.Queue()
        self.running = True
        os.makedirs(self.spool_dir, exist_ok=True)
        if recover_existing:
            self.recover_spooled_items()

    # ------------------------------------------------------------------
    # Start-up recovery
    # ------------------------------------------------------------------
    def recover_spooled_items(self) -> int:
        """Re-queue records left in the spool directory by an earlier run.

        Files are queued oldest first. A file that cannot be parsed is renamed to
        ``*.json.corrupt`` so it never blocks the queue and can be inspected later.
        Returns the number of records re-queued.
        """
        try:
            names = [n for n in os.listdir(self.spool_dir) if n.endswith(".json")]
        except OSError as e:
            logger.error(f"[Conveyor Belt]: Cannot scan spool directory: {e}")
            return 0

        paths = sorted(
            (os.path.join(self.spool_dir, n) for n in names),
            key=lambda p: os.path.getmtime(p),
        )
        recovered = 0
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if not isinstance(payload, dict):
                    raise ValueError("spool record is not a JSON object")
            except (OSError, ValueError) as e:
                logger.error(f"[Conveyor Belt]: Unreadable spool file {os.path.basename(path)}: {e}")
                try:
                    os.replace(path, path + ".corrupt")
                except OSError:
                    pass
                continue

            if os.path.basename(path).startswith(NURSE_ASSISTANCE_PREFIX):
                label = "Recovered anonymous nurse-assistance event"
            else:
                label = f"Recovered case {payload.get('case_id', os.path.basename(path))}"
            self.queue.put((label, path, payload))
            recovered += 1

        if recovered:
            logger.info(f"[Conveyor Belt]: Re-queued {recovered} spooled record(s) from a previous run.")
        return recovered

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------
    def enqueue_case(self, case_container: dict):
        case_id = case_container.get("case_id", f"CASE-{int(time.time())}")
        disk_path = os.path.join(self.spool_dir, f"{case_id}.json")

        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(case_container, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to spool case to local disk: {e}")

        self.queue.put((f"Case {case_id}", disk_path, case_container))
        logger.info(f"[Conveyor Belt]: Case {case_id} queued for background transmission.")

    def enqueue_nurse_assistance(self, event: dict):
        """Persist and enqueue an anonymous, minimal nurse-assistance event.

        A random ``event_id`` is added so the dashboard can ignore a retried copy.
        It is generated here, is not linked to any person or case, and is safe
        under the PDPA data-minimisation rule.
        """
        event_id = uuid.uuid4().hex
        event = {**event, "event_id": event_id}
        disk_path = os.path.join(self.spool_dir, f"{NURSE_ASSISTANCE_PREFIX}{event_id}.json")

        try:
            with open(disk_path, "x", encoding="utf-8") as f:
                json.dump(event, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"Failed to persist consent-refusal nurse handoff: {e}")
            raise

        self.queue.put(("Anonymous nurse-assistance event", disk_path, event))
        logger.info("Anonymous nurse-assistance event queued for dashboard delivery.")

    # ------------------------------------------------------------------
    # Delivery loop
    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            try:
                item_label, disk_path, payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            transmitted = False
            while not transmitted and self.running:
                try:
                    res = requests.post(self.endpoint, json=payload, timeout=SystemConfig.HTTP_TIMEOUT)
                    if res.status_code in [200, 201]:
                        logger.info(f"[Conveyor Belt]: {item_label} delivered to Nurse Dashboard.")
                        transmitted = True
                        if os.path.exists(disk_path):
                            os.remove(disk_path)
                    else:
                        logger.warning(f"[Conveyor Belt]: {item_label}; dashboard status {res.status_code}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                        time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)
                except requests.exceptions.RequestException:
                    logger.warning(f"[Conveyor Belt]: Dashboard unreachable for {item_label}. Retrying in {SystemConfig.CONVEYOR_RETRY_INTERVAL}s...")
                    time.sleep(SystemConfig.CONVEYOR_RETRY_INTERVAL)

            self.queue.task_done()
