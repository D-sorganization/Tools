"""Contract and persistence tests for the append-only SCADA audit trail."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_log import (  # noqa: E402
    AuditEvent,
    AuditLog,
    AuditOutcome,
    append_audit_event,
    install_append_only_guards,
)
from identity import Principal, Role  # noqa: E402


@pytest.fixture
def audit_engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    SQLModel.metadata.create_all(engine)
    install_append_only_guards(engine)
    return engine


def _event(**overrides: object) -> AuditEvent:
    values: dict[str, object] = {
        "principal": Principal(
            subject="engineer.one",
            display_name="Engineer One",
            role=Role.ENGINEER,
        ),
        "action": "configuration.update",
        "target": "routing/default",
        "reason": "approved synthetic test",
        "outcome": AuditOutcome.SUCCEEDED,
        "before": {"limit": 10.0},
        "after": {"limit": 12.0},
        "source": "test-client",
        "configuration_revision": "cfg-0001",
        "correlation_id": "request-0001",
    }
    values.update(overrides)
    return AuditEvent(**values)  # type: ignore[arg-type]


def test_audit_event_rejects_missing_reason_and_identity() -> None:
    with pytest.raises(ValueError, match="reason"):
        _event(reason=" ")
    with pytest.raises(TypeError, match="principal"):
        _event(principal=None)


def test_append_audit_event_persists_attribution_and_redacts_secrets(
    audit_engine,
) -> None:
    event = _event(
        before={
            "setpoint": 10.0,
            "api_key": "must-not-persist",  # pragma: allowlist secret
            "nested": {"authorization": "Bearer must-not-persist"},
        },
        after={"setpoint": 12.0, "session_token": "must-not-persist"},
    )

    with Session(audit_engine) as session:
        row = append_audit_event(session, event)
        session.commit()
        stored = session.exec(select(AuditLog)).one()

    assert row.id is not None
    assert stored.actor_subject == "engineer.one"
    assert stored.actor_role == "engineer"
    assert stored.action == "configuration.update"
    assert stored.outcome == "succeeded"
    assert stored.reason == "approved synthetic test"
    assert stored.configuration_revision == "cfg-0001"
    assert stored.correlation_id == "request-0001"
    assert json.loads(stored.before_json) == {
        "setpoint": 10.0,
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]"},
    }
    assert json.loads(stored.after_json) == {
        "setpoint": 12.0,
        "session_token": "[REDACTED]",
    }
    assert "must-not-persist" not in stored.before_json
    assert "must-not-persist" not in stored.after_json


def test_append_audit_event_preserves_failed_attempt() -> None:
    event = _event(
        outcome=AuditOutcome.FAILED,
        error_code="permission_denied",
        after=None,
    )
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        stored = append_audit_event(session, event)
        session.commit()
        session.refresh(stored)
        assert stored.outcome == "failed"
        assert stored.error_code == "permission_denied"
        assert stored.after_json == "null"


def test_database_guards_reject_audit_update_and_delete(audit_engine) -> None:
    with Session(audit_engine) as session:
        row = append_audit_event(session, _event())
        session.commit()
        row_id = row.id

    with audit_engine.begin() as connection:
        with pytest.raises(Exception, match="append-only"):
            connection.execute(
                text("UPDATE auditlog SET reason='changed' WHERE id=:row_id"),
                {"row_id": row_id},
            )

    with audit_engine.begin() as connection:
        with pytest.raises(Exception, match="append-only"):
            connection.execute(
                text("DELETE FROM auditlog WHERE id=:row_id"),
                {"row_id": row_id},
            )


def test_append_audit_event_requires_session() -> None:
    with pytest.raises(TypeError, match="session"):
        append_audit_event(object(), _event())  # type: ignore[arg-type]
