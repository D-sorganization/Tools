"""API contracts for the supervisory professional alarm workspace."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from alarm_lifecycle import AlarmDefinition, AlarmManager, AlarmPriority  # noqa: E402
from alarm_router import create_alarm_router  # noqa: E402
from alarm_service import AlarmService  # noqa: E402
from identity import Principal, Role  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

OPERATOR = Principal("operator.1", "Operator One", Role.OPERATOR)
ENGINEER = Principal("engineer.1", "Engineer One", Role.ENGINEER)


def _client() -> tuple[TestClient, AlarmService]:
    manager = AlarmManager(
        [
            AlarmDefinition(
                tag="TAG_0",
                low_limit=10,
                high_limit=90,
                priority=AlarmPriority.HIGH,
                deadband=1,
                on_delay=timedelta(0),
                off_delay=timedelta(0),
                help_text="Synthetic alarm response guidance.",
                suppression_rules=frozenset({"synthetic.maintenance"}),
            )
        ]
    )
    service = AlarmService(manager)
    service.observe({"TAG_0": 95}, datetime(2026, 8, 3, tzinfo=UTC))
    app = FastAPI()
    app.include_router(
        create_alarm_router(
            service,
            operator_dependency=lambda: OPERATOR,
            engineer_dependency=lambda: ENGINEER,
        )
    )
    return TestClient(app), service


def test_active_alarm_surface_includes_lifecycle_help_and_first_out() -> None:
    client, _service = _client()

    response = client.get("/api/alarm-management/active")

    assert response.status_code == 200
    alarm = response.json()[0]
    assert alarm["lifecycle"] == "unacknowledged"
    assert alarm["priority"] == "high"
    assert alarm["first_out_sequence"] == 1
    assert alarm["help_text"].startswith("Synthetic")


def test_acknowledge_and_timed_shelving_mutations() -> None:
    client, _service = _client()

    shelf = client.post(
        "/api/alarm-management/TAG_0/shelf",
        json={"reason": "Synthetic maintenance", "duration_seconds": 300},
    )
    assert shelf.status_code == 200
    assert shelf.json()["lifecycle"] == "shelved"
    assert client.delete("/api/alarm-management/TAG_0/shelf").status_code == 200
    acknowledged = client.post("/api/alarm-management/TAG_0/acknowledge")
    assert acknowledged.status_code == 200
    assert acknowledged.json()["acknowledged_by"] == "operator.1"


def test_designed_suppression_and_performance_report() -> None:
    client, _service = _client()

    response = client.post(
        "/api/alarm-management/TAG_0/suppression",
        json={"rule": "synthetic.maintenance", "active": True},
    )
    assert response.status_code == 200
    assert response.json()["lifecycle"] == "suppressed"
    report = client.get("/api/alarm-management/performance")
    assert report.status_code == 200
    assert report.json()["activations"] == 1


def test_unknown_alarm_is_a_bounded_not_found_contract() -> None:
    client, _service = _client()

    response = client.post("/api/alarm-management/UNKNOWN/acknowledge")

    assert response.status_code == 404
