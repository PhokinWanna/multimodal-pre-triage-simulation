"""Tests for admin pages, data retention and the shared kiosk key."""

import copy
import os
from dataclasses import replace
from datetime import timedelta

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from conftest import ADMIN_PASSWORD, FULL_CASE, NURSE_PASSWORD, login
from nurse_dashboard.app import create_app
from nurse_dashboard.models import AssistanceEvent, AuditLog, Case, Review, Role, User, utcnow
from nurse_dashboard.retention import purge_old_data
from nurse_dashboard.security import create_user

URL = "/api/triage/submit_case"


def _send(client, case_id, **changes):
    p = copy.deepcopy(FULL_CASE)
    p["case_id"] = case_id
    p.update(changes)
    return client.post(URL, json=p)


# ------------------------------------------------------------------ admin: accounts
def test_admin_pages_are_admin_only(client, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    assert client.get("/admin/users").status_code == 403
    assert client.get("/admin/audit").status_code == 403
    assert "/admin/users" not in client.get("/").text            # no Admin link for nurses


def test_admin_adds_nurse_who_can_log_in(client, session, admin):
    login(client, "admin", ADMIN_PASSWORD)
    r = client.post("/admin/users", data={"display_name": "Nurse B", "username": "nurse.b",
                                          "password": "first-pass-1", "role": "nurse"}, follow_redirects=False)
    assert r.status_code == 303
    new = session.exec(select(User).where(User.username == "nurse.b")).one()
    assert new.role == Role.NURSE

    client.post("/logout")
    assert login(client, "nurse.b", "first-pass-1").status_code == 303


def test_add_user_errors_are_shown(client, admin):
    login(client, "admin", ADMIN_PASSWORD)
    r = client.post("/admin/users", data={"display_name": "X", "username": "x", "password": "short"})
    assert r.status_code == 422 and "at least 8" in r.text
    r = client.post("/admin/users", data={"display_name": "Y", "username": "admin", "password": "long-enough-1"})
    assert r.status_code == 422 and "already exists" in r.text


def test_switch_off_and_reset_password(client, session, admin, nurse):
    login(client, "admin", ADMIN_PASSWORD)
    client.post(f"/admin/users/{nurse.id}/active", data={"active": "0"})
    session.expire_all()
    assert session.get(User, nurse.id).is_active is False

    client.post(f"/admin/users/{nurse.id}/active", data={"active": "1"})
    client.post(f"/admin/users/{nurse.id}/password", data={"password": "reset-pass-99"})
    client.post("/logout")
    assert login(client, "nurse.a", NURSE_PASSWORD).status_code == 401
    assert login(client, "nurse.a", "reset-pass-99").status_code == 303


def test_admin_cannot_lock_themselves_out(client, session, admin):
    login(client, "admin", ADMIN_PASSWORD)
    r = client.post(f"/admin/users/{admin.id}/active", data={"active": "0"})
    assert r.status_code == 422 and "your own account" in r.text


def test_admin_can_switch_off_another_admin(client, session, admin):
    # The "at least one active admin" rule in admin.py is a safety net: through the web the
    # only remaining admin is always the person signed in, which the self-check already blocks.
    other = create_user(session, "admin2", "admin-pass-2", role=Role.ADMIN)
    login(client, "admin", ADMIN_PASSWORD)
    assert client.post(f"/admin/users/{other.id}/active", data={"active": "0"}, follow_redirects=False).status_code == 303
    session.expire_all()
    assert session.get(User, other.id).is_active is False


def test_audit_page_lists_actions_in_plain_words(client, admin):
    login(client, "admin", ADMIN_PASSWORD)
    _send(client, "CASE-AUD")
    html = client.get("/admin/audit").text
    assert "Case received from kiosk" in html and "Signed in" in html
    assert "Case received from kiosk" in client.get("/admin/audit?action=case_received").text


# ------------------------------------------------------------------ retention
def test_purge_deletes_old_cases_reviews_alerts_and_photos(app, client, session, settings, nurse):
    img = os.path.join(settings.evidence_dir, "CASE-OLD.jpg")
    open(img, "wb").close()
    _send(client, "CASE-OLD", visual_evidence_path=img)
    _send(client, "CASE-NEW")
    client.post(URL, json={"event_type": "nurse_assistance_requested", "reason": "consent_refused", "event_id": "c" * 32})

    old = session.exec(select(Case).where(Case.case_id == "CASE-OLD")).one()
    old.received_at = utcnow() - timedelta(days=31)
    session.add(Review(case_pk=old.id, decision="CONFIRMED", final_esi=2))
    for ev in session.exec(select(AssistanceEvent)).all():
        ev.received_at = utcnow() - timedelta(days=40)
    session.commit()

    r = purge_old_data(app.state.engine, replace(settings, retention_days=30))
    assert (r.cases, r.reviews, r.events, r.photos) == (1, 1, 1, 1)
    session.expire_all()
    assert [c.case_id for c in session.exec(select(Case)).all()] == ["CASE-NEW"]
    assert not os.path.exists(img)
    assert session.exec(select(AuditLog).where(AuditLog.action == "retention_purge")).first() is not None


def test_purge_disabled_when_zero(app, settings):
    assert purge_old_data(app.state.engine, replace(settings, retention_days=0)).cases == 0


def test_cli_purge(monkeypatch, tmp_path, capsys):
    from nurse_dashboard import __main__ as cli
    monkeypatch.setenv("DASHBOARD_DATABASE_URL", f"sqlite:///{tmp_path / 'p.db'}")
    monkeypatch.setenv("DASHBOARD_EVIDENCE_DIR", str(tmp_path))
    cli.main(["purge"])
    assert "Deleted data older than 30 days: 0 cases" in capsys.readouterr().out


# ------------------------------------------------------------------ kiosk key
def test_kiosk_key_required_when_configured(settings):
    keyed = create_app(replace(settings, kiosk_key="s3cret-key"))
    with TestClient(keyed) as c:
        p = copy.deepcopy(FULL_CASE)
        assert c.post(URL, json=p).status_code == 401
        assert c.post(URL, json=p, headers={"X-Kiosk-Key": "wrong"}).status_code == 401
        assert c.post(URL, json=p, headers={"X-Kiosk-Key": "s3cret-key"}).status_code == 201
        with Session(keyed.state.engine) as s:
            assert len(s.exec(select(AuditLog).where(AuditLog.action == "kiosk_rejected")).all()) == 2


def test_no_key_configured_keeps_local_mode(client):
    assert _send(client, "CASE-LOCAL").status_code == 201
