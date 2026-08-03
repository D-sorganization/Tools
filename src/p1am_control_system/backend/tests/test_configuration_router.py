"""REST contracts for the protected configuration workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from configuration_router import create_configuration_router  # noqa: E402
from configuration_workflow import (  # noqa: E402
    ConfigurationWorkflow,
    InMemoryRevisionRepository,
)
from identity import Principal, Role  # noqa: E402
from models import InterlockConfig, RoutingConfig  # noqa: E402


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


def _client() -> tuple[TestClient, list[RoutingConfig]]:
    deployed: list[RoutingConfig] = []

    async def deploy(config: RoutingConfig) -> None:
        deployed.append(config)

    workflow = ConfigurationWorkflow(InMemoryRevisionRepository(), deploy)
    engineer = Principal("engineer", "Engineer", Role.ENGINEER)
    admin = Principal("admin", "Admin", Role.ADMIN)
    app = FastAPI()
    app.include_router(
        create_configuration_router(
            workflow,
            engineer_dependency=lambda: engineer,
            admin_dependency=lambda: admin,
        )
    )
    return TestClient(app), deployed


def test_api_exposes_reviewed_activation_and_machine_readable_diff() -> None:
    client, deployed = _client()
    created = client.post(
        "/api/configurations/drafts",
        json={"payload": _routing().model_dump(), "reason": "Synthetic change"},
    )
    assert created.status_code == 200
    revision_id = created.json()["revision_id"]

    assert client.post(f"/api/configurations/{revision_id}/validate").status_code == 200
    diff = client.get(f"/api/configurations/{revision_id}/diff")
    assert diff.status_code == 200
    assert diff.json()
    assert client.post(f"/api/configurations/{revision_id}/review").status_code == 200
    approved = client.post(
        f"/api/configurations/{revision_id}/approve",
        json={"reason": "Synthetic review complete"},
    )
    assert approved.json()["state"] == "approved"
    activated = client.post(f"/api/configurations/{revision_id}/activate")

    assert activated.status_code == 200
    assert activated.json()["state"] == "active"
    assert activated.json()["activation_identity"] == revision_id
    assert len(deployed) == 1


def test_api_rejects_silent_direct_activation_and_bounds_unknown_ids() -> None:
    client, deployed = _client()
    response = client.post("/api/configurations/unknown/activate")

    assert response.status_code == 404
    assert deployed == []


def test_api_rollback_creates_new_revision_identity() -> None:
    client, _deployed = _client()
    created = client.post(
        "/api/configurations/drafts",
        json={"payload": _routing().model_dump(), "reason": "Synthetic baseline"},
    ).json()
    revision_id = created["revision_id"]
    client.post(f"/api/configurations/{revision_id}/validate")
    client.post(f"/api/configurations/{revision_id}/review")
    client.post(
        f"/api/configurations/{revision_id}/approve",
        json={"reason": "Synthetic approval"},
    )
    client.post(f"/api/configurations/{revision_id}/activate")

    rollback = client.post(
        f"/api/configurations/{revision_id}/rollback",
        json={"reason": "Synthetic recovery exercise"},
    )

    assert rollback.status_code == 200
    assert rollback.json()["source_revision_id"] == revision_id
    assert rollback.json()["revision_id"] != revision_id
