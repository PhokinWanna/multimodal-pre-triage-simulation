"""Tests for the kiosk outbox spooler (no network, camera or dashboard needed)."""

import json
import os
import tempfile
import time
import unittest

from triage_kiosk.outbox import ConveyorBeltWorker, NURSE_ASSISTANCE_PREFIX

ENDPOINT = "http://127.0.0.1:9/unused"


class OutboxRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spool = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, payload, mtime):
        path = os.path.join(self.spool, name)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(payload, str):
                f.write(payload)
            else:
                json.dump(payload, f)
        os.utime(path, (mtime, mtime))
        return path

    def _drain(self, worker):
        items = []
        while not worker.queue.empty():
            items.append(worker.queue.get_nowait())
        return items

    def test_requeues_spooled_records_oldest_first(self):
        now = time.time()
        self._write("CASE-2.json", {"case_id": "CASE-2"}, now - 10)
        self._write("CASE-1.json", {"case_id": "CASE-1"}, now - 20)
        self._write(f"{NURSE_ASSISTANCE_PREFIX}abc.json",
                    {"event_type": "nurse_assistance_requested", "reason": "consent_refused", "event_id": "abc"},
                    now - 5)

        worker = ConveyorBeltWorker(self.spool, ENDPOINT)
        items = self._drain(worker)

        self.assertEqual([p.get("case_id", p.get("event_id")) for _, _, p in items], ["CASE-1", "CASE-2", "abc"])
        self.assertIn("nurse-assistance", items[2][0])

    def test_corrupt_file_is_set_aside_not_queued(self):
        self._write("CASE-9.json", "{not json", time.time())
        worker = ConveyorBeltWorker(self.spool, ENDPOINT)

        self.assertTrue(worker.queue.empty())
        self.assertTrue(os.path.exists(os.path.join(self.spool, "CASE-9.json.corrupt")))
        self.assertFalse(os.path.exists(os.path.join(self.spool, "CASE-9.json")))

    def test_recovery_can_be_disabled(self):
        self._write("CASE-1.json", {"case_id": "CASE-1"}, time.time())
        worker = ConveyorBeltWorker(self.spool, ENDPOINT, recover_existing=False)
        self.assertTrue(worker.queue.empty())


class NurseAssistanceEventTests(unittest.TestCase):
    def test_event_gets_anonymous_event_id_and_stays_minimal(self):
        with tempfile.TemporaryDirectory() as spool:
            worker = ConveyorBeltWorker(spool, ENDPOINT)
            worker.enqueue_nurse_assistance(
                {"event_type": "nurse_assistance_requested", "reason": "consent_refused"}
            )
            _, path, payload = worker.queue.get_nowait()

            self.assertEqual(set(payload), {"event_type", "reason", "event_id"})
            self.assertRegex(payload["event_id"], r"^[0-9a-f]{32}$")
            self.assertTrue(os.path.basename(path).endswith(f"{payload['event_id']}.json"))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), payload)


if __name__ == "__main__":
    unittest.main()
