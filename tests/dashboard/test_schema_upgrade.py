"""A database created by an older dashboard version must keep working after an update."""

from sqlalchemy import inspect, text
from sqlmodel import Session, select

from nurse_dashboard.db import init_db, upgrade_schema
from nurse_dashboard.models import Case, Review


def test_missing_columns_are_added_with_defaults(app, client):
    engine = app.state.engine
    with engine.begin() as conn:                      # simulate the step-4 database you had
        conn.execute(text("ALTER TABLE reviews DROP COLUMN patient_identified"))
        conn.execute(text("ALTER TABLE cases DROP COLUMN pain_score"))
        conn.execute(text("INSERT INTO cases (case_id, received_at, delivery_count, pdpa_consented, "
                          "chest_clutching, abdominal_clutching, head_clutching, needs_check, status) "
                          "VALUES ('OLD-1', CURRENT_TIMESTAMP, 1, true, false, false, false, false, 'NEW')"))
        case_pk = conn.execute(text("SELECT id FROM cases WHERE case_id = 'OLD-1'")).scalar_one()
        conn.execute(text(f"INSERT INTO reviews (case_pk, decision, final_esi, reviewed_at) "
                          f"VALUES ({case_pk}, 'CONFIRMED', 3, CURRENT_TIMESTAMP)"))

    assert "patient_identified" not in {c["name"] for c in inspect(engine).get_columns("reviews")}
    added = upgrade_schema(engine)
    assert set(added) == {"reviews.patient_identified", "cases.pain_score"}

    with Session(engine) as s:
        old = s.exec(select(Review)).one()
        assert old.patient_identified is True        # old rows get the model default
        assert s.exec(select(Case)).one().pain_score is None


def test_upgrade_is_a_no_op_on_a_current_database(app, client):
    assert upgrade_schema(app.state.engine) == []
    init_db(app.state.engine)                         # safe to run again


def test_server_error_shows_friendly_page(app, client):
    @app.get("/boom-test")
    def boom():
        raise RuntimeError("test failure")

    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom-test")
    assert r.status_code == 500 and "Something went wrong" in r.text and "test failure" not in r.text
