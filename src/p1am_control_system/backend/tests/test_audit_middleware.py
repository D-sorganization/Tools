"""End-to-end contracts for automatic mutation-attempt auditing."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_log import AuditLog, install_append_only_guards  # noqa: E402
from audit_middleware import MutationAuditMiddleware  # noqa: E402
from identity import Principal, Role  # noqa: E402


@pytest.fixture
def audited_app() -> Iterator[tuple[TestClient, object]]:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    install_append_only_guards(engine)
    app = FastAPI()
    principal = Principal("operator.1", "Operator One", Role.OPERATOR)
    app.add_middleware(
        MutationAuditMiddleware,
        engine=engine,
        principal_resolver=lambda _request: principal,
        configuration_revision=lambda: "config-42",
    )

    @app.post("/api/setpoint")
    async def setpoint(payload: dict[str, object]) -> dict[str, object]:
        return payload

    @app.delete("/api/protected")
    async def denied() -> None:
        raise HTTPException(status_code=403, detail="denied")

    @app.patch("/api/broken")
    async def broken() -> None:
        raise RuntimeError("controller failed")

    @app.get("/api/status")
    async def status() -> dict[str, str]:
        return {"status": "ok"}

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, engine


def _rows(engine: object) -> list[AuditLog]:
    with Session(engine) as session:  # type: ignore[arg-type]
        return list(session.exec(select(AuditLog).order_by(AuditLog.id)))


def test_successful_mutation_is_attributed_and_secret_redacted(audited_app) -> None:
    client, engine = audited_app
    response = client.post(
        "/api/setpoint",
        json={
            "value": 12.5,
            "password": "never-store-this",  # pragma: allowlist secret
        },
        headers={
            "X-Change-Reason": "Commissioning check",
            "X-Correlation-ID": "work-order-17",
        },
    )

    assert response.status_code == 200
    row = _rows(engine)[0]
    assert row.actor_subject == "operator.1"
    assert row.reason == "Commissioning check"
    assert row.configuration_revision == "config-42"
    assert row.correlation_id == "work-order-17"
    assert row.outcome == "succeeded"
    payload = json.loads(row.after_json)
    assert payload["request"]["password"] == "[REDACTED]"
    assert "never-store-this" not in row.after_json


def test_denied_and_runtime_failed_mutations_are_both_audited(audited_app) -> None:
    client, engine = audited_app

    assert client.delete("/api/protected").status_code == 403
    assert client.patch("/api/broken").status_code == 500

    rows = _rows(engine)
    assert [row.outcome for row in rows] == ["failed", "failed"]
    assert [row.error_code for row in rows] == ["HTTP_403", "EXCEPTION"]


def test_read_only_request_is_not_written_to_mutation_audit(audited_app) -> None:
    client, engine = audited_app

    assert client.get("/api/status").status_code == 200

    assert _rows(engine) == []
