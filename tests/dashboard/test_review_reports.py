"""Tests for the nurse decision (HITL review), history, statistics and CSV export."""

import copy
import csv
import io

from sqlmodel import select

from conftest import ADMIN_PASSWORD, FULL_CASE, NURSE_PASSWORD, login
from nurse_dashboard.models import AuditLog, Case, CaseStatus, Review, ReviewDecision
from nurse_dashboard.reports import compute_stats
from nurse_dashboard.review import decision_type, triage_difference

URL = "/api/triage/submit_case"


def _send(client, case_id, esi=3, **changes):
    p = copy.deepcopy(FULL_CASE)
    p["case_id"] = case_id
    if esi is None:
        p["triage_verdict"] = None
    else:
        p["triage_verdict"]["esi_level"] = esi
    p.update(changes)
    assert client.post(URL, json=p).status_code == 201


def _case(session, case_id):
    session.expire_all()
    return session.exec(select(Case).where(Case.case_id == case_id)).one()


def _identify(client, case, hn="HN-1"):
    client.post(f"/cases/{case.id}/nurse-data", data={"hn": hn, "patient_name": "Test Patient"})


def _review(client, case, **data):
    return client.post(f"/cases/{case.id}/review", data=data, follow_redirects=False)


# ------------------------------------------------------------------ rules
def test_decision_type_and_difference():
    assert decision_type(3, 3) == ReviewDecision.CONFIRMED
    assert decision_type(3, 2) == ReviewDecision.REVISED
    assert decision_type(None, 2) == ReviewDecision.ASSIGNED
    assert triage_difference(3, 2) == "under"     # kiosk said 3, nurse said more urgent 2
    assert triage_difference(2, 4) == "over"
    assert triage_difference(2, 2) == "same"
    assert triage_difference(None, 2) is None


# ------------------------------------------------------------------ review
def test_confirm_suggestion(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R1", esi=3)
    case = _case(session, "CASE-R1")
    _identify(client, case)

    r = _review(client, case, final_esi="3", disposition="ER_TREATMENT")
    assert r.status_code == 303

    case = _case(session, "CASE-R1")
    assert case.status == CaseStatus.REVIEWED and case.final_esi == 3 and case.ai_esi == 3
    review = session.exec(select(Review)).one()
    assert review.decision == ReviewDecision.CONFIRMED and review.reviewer_id == nurse.id
    assert "Confirmed the kiosk suggestion" in client.get(f"/cases/{case.id}").text
    assert "CASE-R1" not in client.get("/queue/fragment").text   # left the waiting queue


def test_changing_level_requires_reason(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R2", esi=3)
    case = _case(session, "CASE-R2")
    _identify(client, case)

    r = _review(client, case, final_esi="2", disposition="ER_TREATMENT")
    assert r.status_code == 422 and "Give a reason" in r.text
    assert _case(session, "CASE-R2").status != CaseStatus.REVIEWED

    r = _review(client, case, final_esi="2", disposition="ER_TREATMENT", reason="HR 118, SpO2 90%")
    assert r.status_code == 303
    case = _case(session, "CASE-R2")
    assert case.final_esi == 2 and case.ai_esi == 3            # AI suggestion is never overwritten
    assert session.exec(select(Review)).one().decision == ReviewDecision.REVISED


def test_identity_required_or_explicitly_skipped(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R3", esi=1)
    case = _case(session, "CASE-R3")

    r = _review(client, case, final_esi="1", disposition="ER_RESUS")
    assert r.status_code == 422 and "Patient not yet identified" in r.text

    r = _review(client, case, final_esi="1", disposition="ER_RESUS", unidentified_ok="1")
    assert r.status_code == 303
    assert session.exec(select(Review)).one().patient_identified is False


def test_missing_level_and_disposition(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R4")
    case = _case(session, "CASE-R4")
    _identify(client, case)
    r = _review(client, case, final_esi="9")
    assert r.status_code == 422
    assert "Choose a triage level" in r.text and "Choose where the patient goes" in r.text


def test_no_suggestion_is_assigned_without_reason(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R5", esi=None)
    case = _case(session, "CASE-R5")
    _identify(client, case)
    assert _review(client, case, final_esi="4", disposition="FAST_TRACK").status_code == 303
    assert session.exec(select(Review)).one().decision == ReviewDecision.ASSIGNED


def test_correction_keeps_history(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R6", esi=3)
    case = _case(session, "CASE-R6")
    _identify(client, case)
    _review(client, case, final_esi="3", disposition="OPD_QUEUE")
    _review(client, case, final_esi="2", disposition="ER_TREATMENT", reason="Chest pain worsened")

    assert len(session.exec(select(Review)).all()) == 2
    case = _case(session, "CASE-R6")
    assert case.final_esi == 2 and case.disposition == "ER_TREATMENT"
    assert "Earlier decisions (1)" in client.get(f"/cases/{case.id}").text
    actions = [a.action for a in session.exec(select(AuditLog).order_by(AuditLog.id)).all()]
    assert "case_reviewed" in actions and "case_review_corrected" in actions


def test_other_disposition_needs_description(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-R7", esi=5)
    case = _case(session, "CASE-R7")
    _identify(client, case)
    assert _review(client, case, final_esi="5", disposition="OTHER").status_code == 422


# ------------------------------------------------------------------ history
def test_history_search_and_filters(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    _send(client, "CASE-H1", esi=2)
    _send(client, "CASE-H2", esi=4)
    _identify(client, _case(session, "CASE-H1"), hn="77-123")

    html = client.get("/history").text
    assert "CASE-H1" not in html                                  # case IDs are not shown, complaints are
    assert html.count("/cases/") == 2
    assert client.get("/history?q=77-123").text.count("/cases/") == 1
    assert client.get("/history?level=4").text.count("/cases/") == 1
    assert client.get("/history?date_from=2001-01-01&date_to=2001-01-02").text.count("/cases/") == 0


def test_history_requires_login(client):
    assert client.get("/history", follow_redirects=False).status_code == 303


# ------------------------------------------------------------------ statistics
def test_stats_numbers(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    for cid, ai, final in (("S1", 3, 3), ("S2", 3, 2), ("S3", 2, 4)):
        _send(client, cid, esi=ai)
        c = _case(session, cid)
        _identify(client, c)
        _review(client, c, final_esi=str(final), disposition="ER_TREATMENT", reason="test")
    _send(client, "S4", esi=5)                                     # still waiting

    session.expire_all()                                           # read fresh rows, not cached ones
    s = compute_stats(session.exec(select(Case)).all(), [])
    assert (s["total"], s["reviewed"], s["waiting"], s["compared"]) == (4, 3, 1, 3)
    assert (s["agree"], s["under"], s["over"]) == (1, 1, 1)
    assert s["agree_pct"] == 33.3
    assert s["matrix"][(3, 2)] == 1 and s["matrix"][(2, 4)] == 1

    html = client.get("/stats").text
    assert "Kiosk level vs nurse decision" in html and "33.3" in html
    assert "Download CSV" not in html                              # nurses cannot export


# ------------------------------------------------------------------ export
def test_export_is_admin_only(client, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    assert client.get("/export.csv").status_code == 403


def test_export_is_deidentified_and_audited(client, session, admin):
    login(client, "admin", ADMIN_PASSWORD)
    _send(client, "CASE-X1", esi=2)
    c = _case(session, "CASE-X1")
    client.post(f"/cases/{c.id}/nurse-data", data={"hn": "99-999", "patient_name": "Secret Name", "heart_rate": "110"})
    _review(client, c, final_esi="2", disposition="ER_TREATMENT")

    r = client.get("/export.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    text = r.content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 1 and rows[0]["final_esi"] == "2" and rows[0]["heart_rate"] == "110"
    assert "99-999" not in text and "Secret Name" not in text and "เจ็บหน้าอก" not in text
    assert session.exec(select(AuditLog).where(AuditLog.action == "data_exported")).first() is not None
