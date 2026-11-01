"""Tests for the queue, case page, nurse-completed data and live updates."""

import copy
import os

from sqlmodel import select

from conftest import FULL_CASE, NURSE_PASSWORD, login
from nurse_dashboard import labels
from nurse_dashboard.models import AssistanceEvent, AuditLog, Case, CaseStatus, EventStatus
from nurse_dashboard.pages import sort_key

URL = "/api/triage/submit_case"


def _send(client, case_id, esi=2, **changes):
    payload = copy.deepcopy(FULL_CASE)
    payload["case_id"] = case_id
    if esi is None:
        payload["triage_verdict"] = None
    else:
        payload["triage_verdict"]["esi_level"] = esi
    payload.update(changes)
    assert client.post(URL, json=payload).status_code == 201


def _case(session, case_id):
    session.expire_all()
    return session.exec(select(Case).where(Case.case_id == case_id)).one()


def _logged_in(client, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    return client


# ------------------------------------------------------------------ queue
def test_queue_requires_login(client):
    assert client.get("/queue/fragment", follow_redirects=False).status_code == 303
    assert client.get("/events", follow_redirects=False).status_code == 303


def test_empty_queue(client, nurse):
    r = _logged_in(client, nurse).get("/")
    assert r.status_code == 200 and "No patients waiting" in r.text


def test_queue_uses_nurse_words_not_system_terms(client, session, nurse):
    _send(client, "CASE-A", esi=2)
    html = _logged_in(client, nurse).get("/").text
    assert "เจ็บหน้าอกมากครับ" in html                 # chief complaint in the patient's words
    assert "Emergency" in html and "Holding chest" in html
    assert "Face appears flushed" in html and "RED_FLUSHING" not in html   # kiosk value shown in plain words
    for internal in ("chest_clutching", "LLM_INFERENCE_VALIDATED", "SCENARIO_A", "NORMAL_COMPLETE"):
        assert internal not in html


def test_queue_order_level_then_wait(client, session, nurse):
    _send(client, "CASE-L4", esi=4)
    _send(client, "CASE-L1", esi=1)
    _send(client, "CASE-NOLEVEL", esi=None)
    _send(client, "CASE-CANCEL", esi=None, exit_reason="USER_CANCELLED")
    _send(client, "CASE-L3", esi=3)
    order = [c.case_id for c in sorted(session.exec(select(Case)).all(), key=sort_key)]
    assert order == ["CASE-L1", "CASE-NOLEVEL", "CASE-L3", "CASE-L4", "CASE-CANCEL"]


def test_urgent_banner_until_seen(client, session, nurse):
    _send(client, "CASE-U", esi=1)
    c = _logged_in(client, nurse)
    assert "not yet seen" in c.get("/queue/fragment").text

    case = _case(session, "CASE-U")
    r = c.post(f"/cases/{case.id}/ack", follow_redirects=False)
    assert r.status_code == 303
    assert _case(session, "CASE-U").status == CaseStatus.ACKNOWLEDGED
    assert "not yet seen" not in c.get("/queue/fragment").text


def test_alert_keys_for_sound(client, session, nurse):
    _send(client, "CASE-U", esi=2)
    _send(client, "CASE-L5", esi=5)
    html = _logged_in(client, nurse).get("/queue/fragment").text
    urgent_id = _case(session, "CASE-U").id
    assert f'data-alert-keys="c{urgent_id} "' in html     # only the urgent one triggers sound


def test_assistance_alert_and_attended(client, session, nurse):
    client.post(URL, json={"event_type": "nurse_assistance_requested", "reason": "consent_refused", "event_id": "b" * 32})
    c = _logged_in(client, nurse)
    assert "needs a nurse" in c.get("/").text

    event = session.exec(select(AssistanceEvent)).one()
    c.post(f"/assistance/{event.id}/attend")
    session.expire_all()
    assert session.get(AssistanceEvent, event.id).status == EventStatus.ATTENDED
    assert "needs a nurse" not in c.get("/").text


def test_live_event_marker_changes(app, session, nurse):
    from nurse_dashboard.pages import change_marker
    before = change_marker(session)
    session.add(AuditLog(actor="kiosk", action="case_received"))
    session.commit()
    assert change_marker(session) != before


# ------------------------------------------------------------------ case page
def test_case_page_shows_triage_note_and_limitation(client, session, nurse):
    _send(client, "CASE-P", esi=2)
    case = _case(session, "CASE-P")
    html = _logged_in(client, nurse).get(f"/cases/{case.id}").text
    for heading in ("Situation", "Observed by the kiosk camera", "System assessment", "Completed by nurse",
                    "Kiosk conversation", "System details"):
        assert heading in html
    assert "The kiosk cannot measure vital signs" in html
    assert "ESI 2 · MOPH E" in html
    assert "Backup rules were used" not in html            # validated LLM path


def test_fallback_and_incomplete_warnings(client, session, nurse):
    _send(client, "CASE-F", esi=3)
    case = _case(session, "CASE-F")
    case.audit_mode = "DETERMINISTIC_FALLBACK"
    case.needs_check = True
    session.add(case); session.commit()
    html = _logged_in(client, nurse).get(f"/cases/{case.id}").text
    assert "Backup rules were used" in html and "Some kiosk data is incomplete" in html


def test_case_view_is_audited(client, session, nurse):
    _send(client, "CASE-V")
    case = _case(session, "CASE-V")
    _logged_in(client, nurse).get(f"/cases/{case.id}")
    session.expire_all()
    assert session.exec(select(AuditLog).where(AuditLog.action == "case_viewed")).first() is not None


def test_unknown_case_404(client, nurse):
    assert _logged_in(client, nurse).get("/cases/9999").status_code == 404


# ------------------------------------------------------------------ nurse-completed data
def test_nurse_saves_identification_and_vitals(client, session, nurse):
    _send(client, "CASE-N")
    case = _case(session, "CASE-N")
    c = _logged_in(client, nurse)
    r = c.post(f"/cases/{case.id}/nurse-data", data={
        "hn": "65-001234", "patient_name": "Somchai", "heart_rate": "118", "resp_rate": "24",
        "systolic_bp": "150", "diastolic_bp": "95", "temperature_c": "37.8", "spo2": "90", "pain_score": "8",
        "allergies": "Penicillin", "medications": "", "past_history": "HT", "nurse_note": "",
    }, follow_redirects=False)
    assert r.status_code == 303

    case = _case(session, "CASE-N")
    assert case.hn == "65-001234" and case.heart_rate == 118 and case.temperature_c == 37.8
    assert case.medications is None                         # empty field stays empty
    assert case.status == CaseStatus.ACKNOWLEDGED            # entering data = patient seen
    assert case.nurse_data_by == nurse.id

    html = c.get(f"/cases/{case.id}").text
    assert "ESI danger zone" in html and "Consider level 2" in html

    entry = session.exec(select(AuditLog).where(AuditLog.action == "nurse_data_saved")).one()
    assert "Somchai" not in str(entry.detail) and "patient_name" in entry.detail["fields"]  # no personal values in the log


def test_out_of_range_vitals_rejected(client, session, nurse):
    _send(client, "CASE-E")
    case = _case(session, "CASE-E")
    r = _logged_in(client, nurse).post(f"/cases/{case.id}/nurse-data",
                                       data={"heart_rate": "900", "spo2": "abc", "pain_score": "5"})
    assert r.status_code == 422
    assert "Must be between 20 and 250" in r.text and "Enter a number" in r.text
    assert _case(session, "CASE-E").pain_score is None      # nothing saved when any field is wrong


def test_vital_flag_thresholds():
    class V:  # minimal stand-in for a case
        heart_rate, resp_rate, spo2 = 100, 20, 92
    assert labels.vital_flags(V()) == []                     # exactly at the limit is not flagged
    V.heart_rate, V.resp_rate, V.spo2 = 101, 21, 91
    assert len(labels.vital_flags(V())) == 3


# ------------------------------------------------------------------ evidence photo
def test_evidence_photo_requires_login_and_is_served(client, session, settings, nurse):
    img = os.path.join(settings.evidence_dir, "CASE-IMG.jpg")
    with open(img, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")
    _send(client, "CASE-IMG", visual_evidence_path=img)
    case = _case(session, "CASE-IMG")

    assert client.get(f"/cases/{case.id}/evidence", follow_redirects=False).status_code == 303
    r = _logged_in(client, nurse).get(f"/cases/{case.id}/evidence")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"


def test_evidence_path_rechecked_when_served(client, session, nurse):
    _send(client, "CASE-X")
    case = _case(session, "CASE-X")
    case.evidence_path = "/etc/hosts"                       # tampered row
    session.add(case); session.commit()
    assert _logged_in(client, nurse).get(f"/cases/{case.id}/evidence").status_code == 404


# ------------------------------------------------------------------ labels
def test_labels_cover_all_kiosk_exit_reasons():
    from nurse_dashboard.ingest import KNOWN_EXIT_REASONS
    assert KNOWN_EXIT_REASONS <= set(labels.EXIT_REASONS)


def test_service_point_label_shown(app, client, nurse):
    html = _logged_in(client, nurse).get("/").text
    assert "Emergency triage" in html
