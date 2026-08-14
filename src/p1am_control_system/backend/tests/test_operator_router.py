"""REST contracts for the synthetic operator workspace."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

from fastapi import FastAPI
from fastapi.testclient import TestClient
from identity import Principal, Role
from operator_router import create_operator_router
from protection_management import ProtectionService, representative_protections


def _client(role: Role = Role.ENGINEER) -> tuple[TestClient, ProtectionService]:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    service = ProtectionService(representative_protections(), now=lambda: now)
    app = FastAPI()
    app.include_router(
        create_operator_router(
            service,
            engineer_dependency=lambda: Principal("engineer", "Engineer", role),
        )
    )
    return TestClient(app), service


def test_overview_is_explicitly_synthetic_and_multi_area() -> None:
    client, _ = _client()

    response = client.get("/api/operator/overview")

    assert response.status_code == 200
    assert response.json()["data_classification"] == "synthetic"
    assert len(response.json()["areas"]) == 3


def test_trip_and_bypass_endpoints_return_operator_context() -> None:
    client, _ = _client()

    trip = client.post(
        "/api/operator/protections/SYNTHETIC.REACTOR.HIGH_PRESSURE/trips",
        json={"group_id": "fat-trip-1"},
    )
    bypass = client.post(
        "/api/operator/protections/SYNTHETIC.REACTOR.HIGH_PRESSURE/bypasses",
        json={
            "reason": "Synthetic FAT verification",
            "expires_at": (
                datetime(2026, 8, 3, 20, 0, tzinfo=UTC) + timedelta(hours=1)
            ).isoformat(),
        },
    )
    snapshot = client.get("/api/operator/protections")

    assert trip.status_code == 200
    assert trip.json()["first_out"] is True
    assert bypass.status_code == 200
    assert bypass.json()["banner_required"] is True
    assert (
        snapshot.json()["active_bypasses"][0]["reason"] == "Synthetic FAT verification"
    )


def test_non_bypassable_endpoint_maps_policy_conflict() -> None:
    client, _ = _client()

    response = client.post(
        "/api/operator/protections/SYNTHETIC.REACTOR.INDEPENDENT_TRIP/bypasses",
        json={
            "reason": "Attempted synthetic bypass",
            "expires_at": (
                datetime(2026, 8, 3, 20, 0, tzinfo=UTC) + timedelta(hours=1)
            ).isoformat(),
        },
    )

    assert response.status_code == 409
    assert "non-bypassable" in response.json()["detail"]
