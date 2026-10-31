"""Tests for staff login, sessions, lockout, account tools and the audit trail."""

import pytest
from sqlmodel import select

from conftest import NURSE_PASSWORD, login
from nurse_dashboard.models import AuditLog, User
from nurse_dashboard.security import (
    LoginGuard, PasswordPolicyError, create_user, hash_password, set_password, verify_password,
)
from nurse_dashboard.settings import Settings, resolve_secret_key


def _actions(session):
    session.expire_all()
    return [a.action for a in session.exec(select(AuditLog).order_by(AuditLog.id)).all()]


# ------------------------------------------------------------------ passwords
def test_password_is_hashed_never_stored_plain(session, nurse):
    assert nurse.password_hash != NURSE_PASSWORD
    assert nurse.password_hash.startswith("$2")          # bcrypt format
    assert verify_password(NURSE_PASSWORD, nurse.password_hash)
    assert not verify_password("wrong-password", nurse.password_hash)


@pytest.mark.parametrize("bad", ["short", "x" * 73])
def test_password_policy(bad):
    with pytest.raises(PasswordPolicyError):
        hash_password(bad)


def test_usernames_are_unique_and_case_insensitive(session, nurse):
    with pytest.raises(ValueError):
        create_user(session, "NURSE.A", "another-pass-1")


def test_invalid_username_rejected(session):
    with pytest.raises(ValueError):
        create_user(session, "bad name!", "valid-pass-1")


# ------------------------------------------------------------------ pages and redirects
def test_pages_require_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=/"


def test_login_page_renders(client):
    r = client.get("/login")
    assert r.status_code == 200 and "Staff sign in" in r.text


def test_successful_login_then_logout(client, session, nurse):
    r = login(client, "Nurse.A ", NURSE_PASSWORD)           # case and spaces are forgiven
    assert r.status_code == 303 and r.headers["location"] == "/"

    home = client.get("/")
    assert home.status_code == 200 and "Nurse A" in home.text

    out = client.post("/logout", follow_redirects=False)
    assert out.headers["location"] == "/login"
    assert client.get("/", follow_redirects=False).status_code == 303
    assert _actions(session)[-2:] == ["login", "logout"]


def test_session_cookie_is_httponly_and_samesite_strict(client, nurse):
    r = login(client, "nurse.a", NURSE_PASSWORD)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_wrong_password_is_rejected_and_audited(client, session, nurse):
    r = login(client, "nurse.a", "wrong-password")
    assert r.status_code == 401 and "Wrong username or password" in r.text
    assert _actions(session)[-1] == "login_failed"


def test_unknown_user_gets_same_message(client):
    r = login(client, "nobody", "whatever-123")
    assert r.status_code == 401 and "Wrong username or password" in r.text


def test_lockout_after_five_failures(client, session, nurse):
    for _ in range(5):
        login(client, "nurse.a", "wrong-password")
    r = login(client, "nurse.a", NURSE_PASSWORD)            # even the right password is blocked now
    assert r.status_code == 429
    assert _actions(session)[-1] == "login_blocked"


def test_login_guard_expires():
    guard = LoginGuard(max_attempts=2, lockout_seconds=0.0)
    guard.failed("a"); guard.failed("a")
    assert not guard.is_locked("a")                          # window already passed


def test_deactivated_user_cannot_log_in(client, session, nurse):
    nurse.is_active = False
    session.add(nurse); session.commit()
    assert login(client, "nurse.a", NURSE_PASSWORD).status_code == 401


def test_deactivating_signs_out_existing_session(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    nurse.is_active = False
    session.add(nurse); session.commit()
    assert client.get("/", follow_redirects=False).status_code == 303


def test_password_change_signs_out_existing_session(client, session, nurse):
    login(client, "nurse.a", NURSE_PASSWORD)
    assert client.get("/").status_code == 200
    set_password(session, nurse, "brand-new-pass-1")
    assert client.get("/", follow_redirects=False).status_code == 303


@pytest.mark.parametrize("evil", ["https://evil.example", "//evil.example", "/\\evil.example"])
def test_next_parameter_cannot_redirect_off_site(client, nurse, evil):
    r = login(client, "nurse.a", NURSE_PASSWORD, next_url=evil)
    assert r.headers["location"] == "/"


def test_next_parameter_returns_to_requested_page(client, nurse):
    r = login(client, "nurse.a", NURSE_PASSWORD, next_url="/history?esi=2")
    assert r.headers["location"] == "/history?esi=2"


def test_kiosk_endpoint_still_works_without_login(client, full_case):
    assert client.post("/api/triage/submit_case", json=full_case).status_code == 201


def test_favicon_served(client):
    assert client.get("/favicon.ico").status_code == 200


# ------------------------------------------------------------------ secret key and CLI
def test_secret_key_is_created_once_and_reused(tmp_path):
    s = Settings(database_url="sqlite://", secret_key=None, secret_key_file=str(tmp_path / "k" / "secret_key"))
    first = resolve_secret_key(s)
    assert len(first) == 64 and resolve_secret_key(s) == first


def test_cli_create_admin_and_list(monkeypatch, tmp_path, capsys):
    from nurse_dashboard import __main__ as cli

    monkeypatch.setenv("DASHBOARD_DATABASE_URL", f"sqlite:///{tmp_path / 'cli.db'}")
    answers = iter(["Head.Nurse", "Head Nurse"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt="": "strong-pass-9")

    cli.main(["create-admin"])
    cli.main(["list-users"])
    out = capsys.readouterr().out
    assert "Created admin account 'head.nurse'" in out
    assert "head.nurse" in out and "admin" in out and "active" in out


def test_cli_deactivate(monkeypatch, tmp_path, capsys):
    from nurse_dashboard import __main__ as cli
    from nurse_dashboard.db import make_engine
    from sqlmodel import Session

    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DASHBOARD_DATABASE_URL", url)
    answers = iter(["n1", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt="": "strong-pass-9")
    cli.main(["create-user"])
    cli.main(["deactivate", "n1"])

    with Session(make_engine(Settings(database_url=url, secret_key="x"))) as s:
        assert s.exec(select(User).where(User.username == "n1")).one().is_active is False
