"""REST surface for reusable procedure, connector, notification, and HA contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

from availability import AvailabilityPolicy, AvailabilityService
from connector_plugins import ConnectorDescriptor, ConnectorManager
from fastapi import FastAPI
from fastapi.testclient import TestClient
from identity import Principal, Role
from notification_policy import NotificationPolicy, NotificationService
from product_router import create_product_router
from synthetic_procedure import SyntheticProcedure


class Connector:
    descriptor = ConnectorDescriptor(
        connector_id="SYNTHETIC.CONNECTOR.DEMO",
        version="1.0",
        tags=("SYNTHETIC.DEMO.PV",),
    )

    def read(self) -> dict[str, float]:
        return {"SYNTHETIC.DEMO.PV": 1.0}

    def write(self, tag: str, value: float) -> None:
        raise AssertionError("no writable tags")

    def diagnostics(self) -> dict[str, object]:
        return {"state": "online"}


class Channel:
    def send(self, recipient: str, message: str) -> None:
        return None


def _client() -> TestClient:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    procedure = SyntheticProcedure(now=lambda: now)
    connectors = ConnectorManager((Connector(),))
    notifications = NotificationService(
        NotificationPolicy(
            initial_delay=timedelta(minutes=1),
            escalation_delay=timedelta(minutes=5),
            primary_recipient="synthetic.primary",
            escalation_recipient="synthetic.escalation",
        ),
        Channel(),
        now=lambda: now,
    )
    availability = AvailabilityService(
        AvailabilityPolicy(
            recovery_time_objective=timedelta(minutes=5),
            recovery_point_objective=timedelta(seconds=30),
            max_clock_skew=timedelta(seconds=2),
            buffer_capacity=100,
        )
    )
    app = FastAPI()
    app.include_router(
        create_product_router(
            procedure,
            connectors,
            notifications,
            availability,
            operator_dependency=lambda: Principal(
                "operator.one", "Operator One", Role.OPERATOR
            ),
        )
    )
    return TestClient(app)


def test_product_status_exposes_all_reusable_contracts() -> None:
    response = _client().get("/api/operator/product-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["procedure_state"] == "idle"
    assert payload["connectors"][0]["connector_id"] == "SYNTHETIC.CONNECTOR.DEMO"
    assert payload["samples"]["SYNTHETIC.DEMO.PV"]["quality"] == "good"
    assert payload["notification_policy"]["primary_recipient"] == "synthetic.primary"
    assert payload["availability"]["recovery_time_objective_seconds"] == 300
    assert payload["data_classification"] == "synthetic"


def test_procedure_commands_are_role_gated_and_attributed() -> None:
    client = _client()

    response = client.post(
        "/api/operator/procedure/commands/start",
        json={"reason": "Begin representative procedure"},
    )

    assert response.status_code == 200
    assert response.json()["after"] == "starting"
    assert response.json()["actor"] == "operator.one"
