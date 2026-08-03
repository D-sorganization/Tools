"""Deployment identity and health-center contracts."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_workflow import (  # noqa: E402
    ConfigurationWorkflow,
    InMemoryRevisionRepository,
)
from models import RoutingConfig  # noqa: E402
from recovery_package import RecoveryPackageService  # noqa: E402
from system_health import HealthStatus, SystemHealthService  # noqa: E402

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _service(plc_connected: bool = False) -> SystemHealthService:
    async def deploy(_config: RoutingConfig) -> None:
        return None

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    recovery = RecoveryPackageService(workflow, "software-test-1")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return SystemHealthService(
        workflow=workflow,
        recovery=recovery,
        engine=engine,
        software_revision="software-test-1",
        plc_connected=lambda: plc_connected,
        simulator_available=lambda: True,
        clock=lambda: datetime(2026, 8, 3, tzinfo=UTC),
    )


def test_identity_is_observable_even_before_first_activation() -> None:
    identity = _service().identity()

    assert identity.software_revision == "software-test-1"
    assert identity.configuration_revision == "unversioned"
    assert identity.configuration_sha256 is None


def test_health_distinguishes_primary_transport_from_simulator_availability() -> None:
    report = _service(plc_connected=False).report()

    checks = {check.name: check for check in report.checks}
    assert report.overall is HealthStatus.DEGRADED
    assert checks["database"].status is HealthStatus.GOOD
    assert checks["primary_transport"].status is HealthStatus.DEGRADED
    assert checks["simulator"].status is HealthStatus.GOOD
    assert checks["configuration_identity"].status is HealthStatus.DEGRADED
