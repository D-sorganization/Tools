"""REST contracts for isolated scenario execution and evidence download."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence_package import EvidencePackageService  # noqa: E402
from identity import Principal, Role  # noqa: E402
from scenario_router import create_scenario_router  # noqa: E402


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(
        create_scenario_router(
            identity_provider=lambda: ("software-test-1", "cfg-000001-proof"),
            admin_dependency=lambda: Principal("admin", "Admin", Role.ADMIN),
        )
    )
    return TestClient(app)


def test_representative_scenario_is_machine_marked_synthetic() -> None:
    response = _client().get("/api/acceptance/scenarios/representative")

    assert response.status_code == 200
    assert response.json()["data_classification"] == "synthetic"
    assert response.json()["not_for_live_control"] is True
    assert all(
        step["target"].startswith("SYNTHETIC.") for step in response.json()["steps"]
    )


def test_scenario_run_returns_verified_self_contained_evidence_zip() -> None:
    client = _client()
    scenario = client.get("/api/acceptance/scenarios/representative").json()
    response = client.post("/api/acceptance/scenarios/run", json=scenario)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["x-evidence-passed"] == "true"
    verified = EvidencePackageService().verify(
        response.content, response.headers["x-artifact-sha256"]
    )
    assert verified.evidence.software_revision == "software-test-1"
    assert verified.evidence.configuration_revision == "cfg-000001-proof"
