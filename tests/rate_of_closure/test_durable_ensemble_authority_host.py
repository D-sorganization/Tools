"""Authenticated HTTP lifecycle for the durable ensemble authority."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from rate_of_closure.application.durable_ensemble import (
    DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
    DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
    DURABLE_ENSEMBLE_SCOPE,
    DurableEnsembleJobRegistry,
    RateDurableEnsembleService,
)
from rate_of_closure.application.morris.host import (
    API_PREFIX,
    DURABLE_ENSEMBLE_CAPABILITY_PATH,
    create_morris_authority_app,
)
from rate_of_closure.application.morris.router import MorrisJobRegistry
from rate_of_closure.application.morris.service import RateMorrisService

from .test_durable_ensemble_authority_service import _request


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret-token"}


def _wait(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 10.0
    path = f"{API_PREFIX}/durable-ensembles/jobs/{job_id}"
    while time.monotonic() < deadline:
        response = client.get(path, headers=_headers())
        document = response.json()
        if document["status"] in {"completed", "cancelled", "failed"}:
            return document
        time.sleep(0.01)
    raise AssertionError("authority job did not finish")


def test_authenticated_transport_exposes_capability_and_path_free_results(
    tmp_path: Path,
) -> None:
    morris = MorrisJobRegistry(RateMorrisService())
    durable = DurableEnsembleJobRegistry(RateDurableEnsembleService(tmp_path))
    app = create_morris_authority_app(
        "secret-token", morris, durable_ensemble_registry=durable
    )
    with TestClient(app) as client:
        unauthorized = client.get(DURABLE_ENSEMBLE_CAPABILITY_PATH)
        assert unauthorized.status_code == 401
        capability = client.get(DURABLE_ENSEMBLE_CAPABILITY_PATH, headers=_headers())
        assert capability.json() == {
            "schema_id": "rate-of-closure/durable-ensemble-authority-capability",
            "schema_version": 1,
            "available": True,
            "api_prefix": API_PREFIX,
            "scope": DURABLE_ENSEMBLE_SCOPE,
            "request_schema_id": DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
            "job_schema_id": DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
        }

        path = f"{API_PREFIX}/durable-ensembles/jobs"
        created = client.post(path, headers=_headers(), json=_request(2).to_json_dict())
        assert created.status_code == 202
        completed = _wait(client, created.json()["job_id"])

        assert completed["schema_id"] == DURABLE_ENSEMBLE_JOB_SCHEMA_ID
        assert completed["status"] == "completed"
        assert completed["completed_trials"] == 3
        assert completed["evidence"]["archive"]["analyzed_trial_count"] == 3
        assert "directory" not in str(completed)
        assert str(tmp_path) not in str(completed)


def test_transport_rejects_path_fields_and_unknown_jobs(tmp_path: Path) -> None:
    morris = MorrisJobRegistry(RateMorrisService())
    durable = DurableEnsembleJobRegistry(RateDurableEnsembleService(tmp_path))
    app = create_morris_authority_app(
        "secret-token", morris, durable_ensemble_registry=durable
    )
    with TestClient(app) as client:
        document = _request().to_json_dict()
        document["directory"] = "C:/private"
        response = client.post(
            f"{API_PREFIX}/durable-ensembles/jobs",
            headers=_headers(),
            json=document,
        )
        assert response.status_code == 422
        unknown = client.get(
            f"{API_PREFIX}/durable-ensembles/jobs/not-known",
            headers=_headers(),
        )
        assert unknown.status_code == 404
