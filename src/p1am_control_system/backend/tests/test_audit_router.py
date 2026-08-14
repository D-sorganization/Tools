"""Read-side API contracts for the immutable audit trail."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_log import AuditEvent, AuditOutcome, append_audit_event  # noqa: E402
from audit_router import create_audit_router  # noqa: E402
from identity import Principal, Role  # noqa: E402


def _client() -> TestClient:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    operator = Principal("operator.1", "Operator One", Role.OPERATOR)
    with Session(engine) as session:
        for index, outcome in enumerate(
            (AuditOutcome.SUCCEEDED, AuditOutcome.FAILED), start=1
        ):
            append_audit_event(
                session,
                AuditEvent(
                    principal=operator,
                    action="post /api/setpoint",
                    target="/api/setpoint",
                    reason=f"test {index}",
                    outcome=outcome,
                    before={},
                    after={"value": index},
                    source="test",
                    configuration_revision="rev-1",
                    correlation_id=f"corr-{index}",
                    error_code=None if index == 1 else "HTTP_403",
                ),
            )
        session.commit()

    def get_session() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app = FastAPI()
    app.include_router(create_audit_router(get_session, lambda: operator))
    return TestClient(app)


def test_audit_page_is_newest_first_and_structured() -> None:
    response = _client().get("/api/audit?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert len(payload["items"]) == 1
    assert payload["items"][0]["correlation_id"] == "corr-2"
    assert payload["items"][0]["outcome"] == "failed"


def test_audit_query_filters_by_actor_outcome_and_correlation() -> None:
    response = _client().get(
        "/api/audit",
        params={
            "actor_subject": "operator.1",
            "outcome": "succeeded",
            "correlation_id": "corr-1",
        },
    )

    assert response.status_code == 200
    assert [item["correlation_id"] for item in response.json()["items"]] == ["corr-1"]


def test_audit_query_rejects_invalid_outcome_contract() -> None:
    response = _client().get("/api/audit?outcome=maybe")

    assert response.status_code == 422
