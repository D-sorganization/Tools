"""REST contracts for recovery packages and the system-health center."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_workflow import (  # noqa: E402
    ConfigurationWorkflow,
    InMemoryRevisionRepository,
)
from identity import Principal, Role  # noqa: E402
from models import InterlockConfig, RoutingConfig  # noqa: E402
from recovery_package import RecoveryPackageService  # noqa: E402
from system_health import SystemHealthService  # noqa: E402
from system_router import create_system_router  # noqa: E402


def _routing() -> RoutingConfig:
    return RoutingConfig(
        input_routing=["TAG_0"],
        output_routing=[],
        pids=[],
        interlocks={
            "TAG_0": InterlockConfig(
                lolo_limit=0,
                low_limit=10,
                high_limit=90,
                hihi_limit=100,
            )
        },
    )


async def _client() -> TestClient:
    async def deploy(_config: RoutingConfig) -> None:
        return None

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    engineer = Principal("engineer", "Engineer", Role.ENGINEER)
    admin = Principal("admin", "Admin", Role.ADMIN)
    draft = workflow.create_draft(_routing(), engineer, "Synthetic baseline")
    workflow.validate(draft.revision_id, engineer)
    workflow.submit_for_review(draft.revision_id, engineer)
    workflow.approve(draft.revision_id, engineer, "Synthetic approval")
    await workflow.activate(draft.revision_id, admin)
    recovery = RecoveryPackageService(workflow, "software-test-1")
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    health = SystemHealthService(
        workflow,
        recovery,
        engine,
        "software-test-1",
        plc_connected=lambda: False,
        simulator_available=lambda: True,
        clock_synchronized=lambda: None,
        storage_free_bytes=lambda: 2_000_000_000,
        service_running=lambda: True,
        driver_identity=lambda: "representative.test.driver",
    )
    app = FastAPI()
    app.include_router(
        create_system_router(
            recovery,
            health,
            engineer_dependency=lambda: engineer,
            admin_dependency=lambda: admin,
        )
    )
    return TestClient(app)


async def test_system_api_downloads_and_restores_verified_package() -> None:
    client = await _client()
    backup = client.post("/api/system/backups")

    assert backup.status_code == 200
    assert backup.headers["content-type"] == "application/zip"
    checksum = backup.headers["x-artifact-sha256"]
    restored = client.post(
        "/api/system/restores",
        content=backup.content,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Artifact-SHA256": checksum,
            "X-Change-Reason": "Synthetic recovery exercise",
        },
    )

    assert restored.status_code == 200
    assert restored.json()["state"] == "draft"


async def test_system_api_exposes_distinct_identity_and_health() -> None:
    client = await _client()

    identity = client.get("/api/system/identity")
    health = client.get("/api/system/health")

    assert identity.status_code == 200
    assert identity.json()["software_revision"] == "software-test-1"
    assert identity.json()["configuration_revision"].startswith("cfg-")
    assert health.status_code == 200
    assert health.json()["overall"] == "degraded"
