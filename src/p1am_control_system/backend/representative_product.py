"""Non-confidential product demonstration composition for the operator workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from advisory_workspace import AdvisoryService
from availability import AvailabilityPolicy, AvailabilityService
from connector_plugins import ConnectorDescriptor, ConnectorManager
from notification_policy import NotificationPolicy, NotificationService
from synthetic_procedure import SyntheticProcedure


class _HealthyConnector:
    descriptor = ConnectorDescriptor(
        connector_id="SYNTHETIC.CONNECTOR.HEALTHY",
        version="1.0.0",
        tags=("SYNTHETIC.CONNECTOR.HEALTHY.PV",),
        writable_tags=("SYNTHETIC.CONNECTOR.HEALTHY.SP",),
    )

    def read(self) -> dict[str, float]:
        return {"SYNTHETIC.CONNECTOR.HEALTHY.PV": 42.0}

    def write(self, tag: str, value: float) -> None:
        if tag != "SYNTHETIC.CONNECTOR.HEALTHY.SP":
            raise KeyError(tag)

    def diagnostics(self) -> dict[str, object]:
        return {"state": "online", "transport": "representative"}


class _UnavailableConnector:
    descriptor = ConnectorDescriptor(
        connector_id="SYNTHETIC.CONNECTOR.UNAVAILABLE",
        version="1.0.0",
        tags=("SYNTHETIC.CONNECTOR.UNAVAILABLE.PV",),
    )

    def read(self) -> dict[str, float]:
        raise ConnectionError("representative offline connector")

    def write(self, tag: str, value: float) -> None:
        raise ConnectionError("representative offline connector")

    def diagnostics(self) -> dict[str, object]:
        return {  # pragma: allowlist secret
            "state": "offline",
            "password": "demonstration-redaction-value",
        }


class _AuditOnlyChannel:
    def send(self, recipient: str, message: str) -> None:
        """No external side effect; the service retains delivery audit only."""


@dataclass(frozen=True)
class RepresentativeProduct:
    procedure: SyntheticProcedure
    connectors: ConnectorManager
    notifications: NotificationService
    availability: AvailabilityService
    advisories: AdvisoryService


def build_representative_product(now: Callable[[], datetime]) -> RepresentativeProduct:
    return RepresentativeProduct(
        procedure=SyntheticProcedure(now=now),
        connectors=ConnectorManager((_HealthyConnector(), _UnavailableConnector())),
        notifications=NotificationService(
            NotificationPolicy(
                initial_delay=timedelta(minutes=1),
                escalation_delay=timedelta(minutes=5),
                primary_recipient="synthetic.on-call.primary",
                escalation_recipient="synthetic.on-call.escalation",
                max_deliveries=10,
            ),
            _AuditOnlyChannel(),
            now=now,
        ),
        availability=AvailabilityService(
            AvailabilityPolicy(
                recovery_time_objective=timedelta(minutes=5),
                recovery_point_objective=timedelta(seconds=30),
                max_clock_skew=timedelta(seconds=2),
                buffer_capacity=1000,
            )
        ),
        advisories=AdvisoryService(now=now),
    )
