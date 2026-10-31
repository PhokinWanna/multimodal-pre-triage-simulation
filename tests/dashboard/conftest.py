"""Shared fixtures for dashboard tests.

Tests run on SQLite always. Set TEST_POSTGRES_URL (e.g. in CI) to run the same
tests on PostgreSQL too:
    TEST_POSTGRES_URL=postgresql+psycopg://triage:triage@localhost:5432/triage_test
"""

import copy
import os

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from nurse_dashboard.app import create_app
from nurse_dashboard.settings import Settings

BACKENDS = ["sqlite"] + (["postgresql"] if os.getenv("TEST_POSTGRES_URL") else [])


@pytest.fixture(params=BACKENDS)
def settings(request, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'test.db'}"
    else:
        url = os.environ["TEST_POSTGRES_URL"]
    return Settings(database_url=url, evidence_dir=str(evidence), secret_key="test")


@pytest.fixture
def app(settings):
    application = create_app(settings)
    if not settings.is_sqlite:  # start every PostgreSQL test from empty tables
        from nurse_dashboard.db import init_db
        init_db(application.state.engine)
        SQLModel.metadata.drop_all(application.state.engine)
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:  # runs the lifespan, so tables are created
        yield c


@pytest.fixture
def session(app, client):
    with Session(app.state.engine) as s:
        yield s


# A real payload shape, copied from what triage_kiosk/main.py builds.
FULL_CASE = {
    "case_id": "CASE-1730361234",
    "start_time": "2026-10-31 14:20:05",
    "language": "th",
    "pdpa_consented": True,
    "exit_reason": "NORMAL_COMPLETE",
    "conversation": [
        {"role": "pdpa_response", "text": "ยินยอมครับ"},
        {"role": "kiosk_proactive", "text": "วันนี้มีอาการอะไรครับ"},
        {"role": "patient", "text": "เจ็บหน้าอกมากครับ"},
    ],
    "cumulative_visual_signs": {"chest_clutching": True, "abdominal_clutching": False, "head_clutching": False},
    "facial_skin_status": "FLUSHING",
    "visual_evidence_path": None,  # filled per test
    "triage_verdict": {
        "esi_level": 2,
        "urgency": "Emergent",
        "scenario_type": "SCENARIO_A_CONFIRMED",
        "subjective": "เจ็บหน้าอกมากครับ",
        "objective": "Visual signs: chest clutching; Facial Skin Status (CNN): FLUSHING",
        "assessment": "Possible acute coronary syndrome.",
        "plan": "Immediate ECG and physician review.",
        "audit_mode": "LLM_INFERENCE_VALIDATED",
    },
}


@pytest.fixture
def full_case():
    return copy.deepcopy(FULL_CASE)


# ---------------------------------------------------------------- accounts
NURSE_PASSWORD = "nurse-pass-123"
ADMIN_PASSWORD = "admin-pass-123"


@pytest.fixture
def nurse(session):
    from nurse_dashboard.security import create_user
    return create_user(session, "nurse.a", NURSE_PASSWORD, display_name="Nurse A")


@pytest.fixture
def admin(session):
    from nurse_dashboard.models import Role
    from nurse_dashboard.security import create_user
    return create_user(session, "admin", ADMIN_PASSWORD, role=Role.ADMIN, display_name="Admin")


def login(client, username, password, next_url="/"):
    return client.post("/login", data={"username": username, "password": password, "next": next_url},
                       follow_redirects=False)
