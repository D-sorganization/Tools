"""REST tests for advisory review without an authoritative write path."""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

try:
    from ._route_inventory import route_paths
except ImportError:
    from _route_inventory import route_paths
from advisory_router import create_advisory_router
from advisory_workspace import AdvisoryService
from fastapi import FastAPI
from fastapi.testclient import TestClient
from identity import Principal, Role


def _client() -> tuple[TestClient, FastAPI]:
    app = FastAPI()
    service = AdvisoryService(now=lambda: datetime(2026, 8, 3, 21, 0, tzinfo=UTC))
    app.include_router(
        create_advisory_router(
            service,
            operator_dependency=lambda: Principal(
                "operator.one", "Operator One", Role.OPERATOR
            ),
            read_dependency=lambda: None,
        )
    )
    return TestClient(app), app


def test_representative_advisory_and_disposition_are_review_only() -> None:
    client, app = _client()

    response = client.get("/api/operator/advisories/representative")
    assert response.status_code == 200
    advisory = response.json()
    assert advisory["authoritative_write_available"] is False
    assert advisory["replay"]["verified"] is True

    disposition = client.post(
        f"/api/operator/advisories/{advisory['advisory_id']}/dispositions",
        json={"decision": "accepted_for_review", "reason": "Use in synthetic study"},
    )
    assert disposition.status_code == 200
    assert disposition.json()["applied_to_control"] is False

    advisory_paths = {path for path in route_paths(app) if "/advisories" in path}
    # Guard the guard: an empty set would satisfy the `all(...)` below vacuously,
    # which is exactly what happened when this inventory was built by walking
    # `app.routes` and skipping FastAPI's `_IncludedRouter` marker. See
    # _route_inventory for why the schema is the authority here.
    assert advisory_paths, "advisory routes not discovered; next check is vacuous"
    assert all("command" not in path and "write" not in path for path in advisory_paths)


def test_unknown_advisory_cannot_receive_a_disposition() -> None:
    client, _ = _client()

    response = client.post(
        "/api/operator/advisories/unknown/dispositions",
        json={"decision": "rejected", "reason": "No matching result"},
    )

    assert response.status_code == 404
