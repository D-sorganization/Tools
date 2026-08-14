"""Deployment identity and bounded system-health aggregation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from configuration_workflow import ConfigurationWorkflow
from enum_compat import StrEnum
from pydantic import BaseModel, ConfigDict, Field
from recovery_package import RecoveryPackageService
from sqlalchemy import Engine

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


class HealthStatus(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    BAD = "bad"


class DeploymentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    software_revision: str = Field(min_length=1)
    configuration_revision: str = Field(min_length=1)
    configuration_sha256: str | None
    configuration_state: str


class HealthCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    status: HealthStatus
    detail: str = Field(min_length=1, max_length=500)


class SystemHealthReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    overall: HealthStatus
    identity: DeploymentIdentity
    checks: tuple[HealthCheck, ...]


def _required_revision(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("software_revision must be a non-empty string")
    return value.strip()


class SystemHealthService:
    """Aggregate independent health providers without conflating their status."""

    def __init__(
        self,
        workflow: ConfigurationWorkflow,
        recovery: RecoveryPackageService,
        engine: Engine,
        software_revision: str,
        plc_connected: Callable[[], bool],
        simulator_available: Callable[[], bool],
        clock_synchronized: Callable[[], bool | None],
        storage_free_bytes: Callable[[], int],
        service_running: Callable[[], bool],
        driver_identity: Callable[[], str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workflow, ConfigurationWorkflow):
            raise TypeError("workflow must be a ConfigurationWorkflow")
        if not isinstance(recovery, RecoveryPackageService):
            raise TypeError("recovery must be a RecoveryPackageService")
        if not isinstance(engine, Engine):
            raise TypeError("engine must be an Engine")
        providers = (
            plc_connected,
            simulator_available,
            clock_synchronized,
            storage_free_bytes,
            service_running,
            driver_identity,
        )
        if not all(callable(provider) for provider in providers):
            raise TypeError("health providers must be callable")
        self._workflow = workflow
        self._recovery = recovery
        self._engine = engine
        self._software_revision = _required_revision(software_revision)
        self._plc_connected = plc_connected
        self._simulator_available = simulator_available
        self._clock_synchronized = clock_synchronized
        self._storage_free_bytes = storage_free_bytes
        self._service_running = service_running
        self._driver_identity = driver_identity
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    def identity(self) -> DeploymentIdentity:
        active = self._workflow.active()
        if active is None:
            return DeploymentIdentity(
                software_revision=self._software_revision,
                configuration_revision="unversioned",
                configuration_sha256=None,
                configuration_state="none",
            )
        return DeploymentIdentity(
            software_revision=self._software_revision,
            configuration_revision=active.activation_identity or active.revision_id,
            configuration_sha256=active.payload_sha256,
            configuration_state=active.state.value,
        )

    def _database_check(self) -> HealthCheck:
        try:
            with self._engine.connect() as connection:
                result = connection.exec_driver_sql("PRAGMA quick_check").scalar_one()
        except Exception as exc:  # noqa: BLE001 - report, do not obscure other checks
            return HealthCheck(
                name="database",
                status=HealthStatus.BAD,
                detail=f"Database check failed: {type(exc).__name__}",
            )
        status = HealthStatus.GOOD if str(result).lower() == "ok" else HealthStatus.BAD
        return HealthCheck(name="database", status=status, detail=str(result))

    def report(self) -> SystemHealthReport:
        identity = self.identity()
        primary_connected = bool(self._plc_connected())
        simulator_available = bool(self._simulator_available())
        clock_synchronized = self._clock_synchronized()
        free_bytes = self._storage_free_bytes()
        if not isinstance(free_bytes, int) or free_bytes < 0:
            raise ValueError(
                "storage_free_bytes provider must return a non-negative int"
            )
        service_running = bool(self._service_running())
        driver_identity = self._driver_identity()
        if not isinstance(driver_identity, str) or not driver_identity.strip():
            raise ValueError("driver_identity provider must return a non-empty string")
        storage_status = (
            HealthStatus.GOOD
            if free_bytes >= 1_000_000_000
            else (
                HealthStatus.DEGRADED if free_bytes >= 100_000_000 else HealthStatus.BAD
            )
        )
        clock_status = (
            HealthStatus.GOOD
            if clock_synchronized is True
            else (
                HealthStatus.BAD
                if clock_synchronized is False
                else HealthStatus.DEGRADED
            )
        )
        checks = (
            self._database_check(),
            HealthCheck(
                name="primary_transport",
                status=(
                    HealthStatus.GOOD if primary_connected else HealthStatus.DEGRADED
                ),
                detail=("Connected" if primary_connected else "Disconnected"),
            ),
            HealthCheck(
                name="simulator",
                status=(HealthStatus.GOOD if simulator_available else HealthStatus.BAD),
                detail=("Available" if simulator_available else "Unavailable"),
            ),
            HealthCheck(
                name="clock",
                status=clock_status,
                detail=(
                    "Synchronized"
                    if clock_synchronized is True
                    else (
                        "Not synchronized"
                        if clock_synchronized is False
                        else "Synchronization source not verified"
                    )
                ),
            ),
            HealthCheck(
                name="storage",
                status=storage_status,
                detail=f"{free_bytes} bytes free",
            ),
            HealthCheck(
                name="service",
                status=HealthStatus.GOOD if service_running else HealthStatus.BAD,
                detail="Running" if service_running else "Stopped",
            ),
            HealthCheck(
                name="driver",
                status=(
                    HealthStatus.GOOD if primary_connected else HealthStatus.DEGRADED
                ),
                detail=driver_identity.strip(),
            ),
            HealthCheck(
                name="configuration_identity",
                status=(
                    HealthStatus.GOOD
                    if identity.configuration_sha256
                    else HealthStatus.DEGRADED
                ),
                detail=identity.configuration_revision,
            ),
            HealthCheck(
                name="recovery_verification",
                status=(
                    HealthStatus.GOOD
                    if self._recovery.last_verified_at
                    else HealthStatus.DEGRADED
                ),
                detail=(
                    self._recovery.last_verified_at.isoformat()
                    if self._recovery.last_verified_at
                    else "No package verified in this process"
                ),
            ),
        )
        ranks = {
            HealthStatus.GOOD: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.BAD: 2,
        }
        overall = max((check.status for check in checks), key=ranks.__getitem__)
        return SystemHealthReport(
            generated_at=self._now(),
            overall=overall,
            identity=identity,
            checks=checks,
        )
