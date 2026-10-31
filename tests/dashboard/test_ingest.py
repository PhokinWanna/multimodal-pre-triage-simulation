"""Tests for POST /api/triage/submit_case (the kiosk → dashboard contract)."""

import os

from sqlmodel import select

from nurse_dashboard.models import AssistanceEvent, AuditLog, Case, CaseStatus

URL = "/api/triage/submit_case"


def _case(session, case_id):
    session.expire_all()
    return session.exec(select(Case).where(Case.case_id == case_id)).one()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_full_case_is_stored_with_201(client, session, settings, full_case):
    img = os.path.join(settings.evidence_dir, "CASE-1730361234.jpg")
    open(img, "wb").close()
    full_case["visual_evidence_path"] = img

    r = client.post(URL, json=full_case)

    assert r.status_code == 201
    assert r.json() == {"status": "created", "kind": "case", "id": "CASE-1730361234", "needs_check": False}
    case = _case(session, "CASE-1730361234")
    assert case.status == CaseStatus.NEW
    assert case.ai_esi == 2 and case.ai_urgency == "Emergent"
    assert case.scenario_type == "SCENARIO_A_CONFIRMED"
    assert case.chest_clutching and not case.abdominal_clutching
    assert case.skin_status == "FLUSHING"
    assert case.conversation[2] == {"role": "patient", "text": "เจ็บหน้าอกมากครับ"}  # Thai survives
    assert case.evidence_path == os.path.realpath(img)
    assert case.final_esi is None and case.check_notes == []


def test_retry_of_same_case_returns_200_and_is_not_duplicated(client, session, full_case):
    assert client.post(URL, json=full_case).status_code == 201
    r = client.post(URL, json=full_case)

    assert r.status_code == 200 and r.json()["status"] == "duplicate"
    rows = session.exec(select(Case)).all()
    assert len(rows) == 1 and rows[0].delivery_count == 2


def test_cancelled_case_without_verdict_is_accepted_and_not_flagged(client, session, full_case):
    full_case.update(case_id="CASE-2", exit_reason="USER_CANCELLED", triage_verdict=None)

    r = client.post(URL, json=full_case)

    assert r.status_code == 201
    case = _case(session, "CASE-2")
    assert case.ai_esi is None and case.needs_check is False


def test_missing_verdict_on_normal_case_is_flagged(client, session, full_case):
    full_case.update(case_id="CASE-3", triage_verdict=None)
    r = client.post(URL, json=full_case)

    assert r.status_code == 201 and r.json()["needs_check"] is True
    assert any("manually" in n for n in _case(session, "CASE-3").check_notes)


def test_malformed_case_is_still_accepted_so_the_kiosk_queue_never_blocks(client, session):
    r = client.post(URL, json={"case_id": "CASE-4", "pdpa_consented": True,
                               "triage_verdict": {"esi_level": 9}, "conversation": "oops"})

    assert r.status_code == 201
    case = _case(session, "CASE-4")
    assert case.needs_check is True
    assert case.ai_esi is None           # 9 is not a valid ESI level
    assert case.conversation == []
    assert case.raw_payload["triage_verdict"] == {"esi_level": 9}


def test_case_without_id_gets_temporary_id(client, session):
    r = client.post(URL, json={"pdpa_consented": True})
    assert r.status_code == 201 and r.json()["id"].startswith("UNKNOWN-")


def test_unconsented_case_keeps_no_personal_data(client, session, full_case):
    full_case.update(case_id="CASE-5", pdpa_consented=False)
    client.post(URL, json=full_case)

    case = _case(session, "CASE-5")
    assert case.needs_check is True
    assert case.conversation == [] and case.subjective is None and case.evidence_path is None
    assert "conversation" not in case.raw_payload and "triage_verdict" not in case.raw_payload


def test_evidence_outside_allowed_folder_is_not_linked(client, session, full_case, tmp_path):
    full_case.update(case_id="CASE-6", visual_evidence_path="/etc/passwd")
    client.post(URL, json=full_case)

    case = _case(session, "CASE-6")
    assert case.evidence_path is None
    assert any("outside the allowed folder" in n for n in case.check_notes)


def test_evidence_path_traversal_is_blocked(settings):
    from nurse_dashboard.ingest import safe_evidence_path

    sneaky = os.path.join(settings.evidence_dir, "..", "secret.jpg")
    assert safe_evidence_path(sneaky, settings.evidence_dir) is None


def test_assistance_event_is_stored_minimal_and_deduplicated(client, session):
    event = {"event_type": "nurse_assistance_requested", "reason": "consent_refused",
             "event_id": "a" * 32, "conversation": [{"role": "patient", "text": "should be ignored"}]}

    first = client.post(URL, json=event)
    second = client.post(URL, json=event)

    assert first.status_code == 201 and first.json()["kind"] == "assistance_event"
    assert second.status_code == 200
    rows = session.exec(select(AssistanceEvent)).all()
    assert len(rows) == 1 and rows[0].reason == "consent_refused"
    assert session.exec(select(Case)).all() == []   # never turned into a case


def test_legacy_assistance_event_without_id_is_accepted(client, session):
    r = client.post(URL, json={"event_type": "nurse_assistance_requested", "reason": "consent_refused"})
    assert r.status_code == 201


def test_non_object_body_is_rejected(client):
    assert client.post(URL, json=[1, 2, 3]).status_code == 400


def test_every_delivery_is_audited(client, session, full_case):
    client.post(URL, json=full_case)
    client.post(URL, json=full_case)

    actions = [a.action for a in session.exec(select(AuditLog).order_by(AuditLog.id)).all()]
    assert actions == ["case_received", "duplicate_delivery"]
