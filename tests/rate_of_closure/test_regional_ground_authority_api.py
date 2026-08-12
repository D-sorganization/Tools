"""Fail-closed contract tests for the regional-ground browser authority."""

from __future__ import annotations

import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.api import create_authority_app
from rate_of_closure.web_authority.capability import (
    AUTHORITY_CAPABILITY_SCHEMA_VERSION,
    AuthorityCapability,
)
from rate_of_closure.web_authority.jobs import AuthorityJobManager
from rate_of_closure.web_authority.runtime import build_authority_process_spec
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job, _result


def test_capability_defaults_to_non_executable() -> None:
    capability = AuthorityCapability.unavailable(
        reason_code="execution_profile_unqualified",
        detail="Exact flight and ground execution profile is not qualified.",
    )

    assert capability.to_wire() == {
        "schema_version": AUTHORITY_CAPABILITY_SCHEMA_VERSION,
        "authority_id": "rate-of-closure-python-authority",
        "authority_version": "1",
        "available": False,
        "regional_ground_execution": False,
        "reason_code": "execution_profile_unqualified",
        "detail": "Exact flight and ground execution profile is not qualified.",
    }


def test_app_requires_nonempty_ephemeral_token() -> None:
    with pytest.raises(ValueError, match="token"):
        create_authority_app(token="")


def test_capability_endpoint_requires_exact_bearer_token() -> None:
    client = TestClient(create_authority_app(token="test-ephemeral-token"))

    missing = client.get("/api/rate-of-closure/v1/capabilities")
    wrong = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.headers["cache-control"] == "no-store"


def test_capability_endpoint_returns_injected_fail_closed_state() -> None:
    capability = AuthorityCapability.unavailable(
        reason_code="runner_not_started",
        detail="Qualified execution runner is not started.",
    )
    client = TestClient(
        create_authority_app(token="test-ephemeral-token", capability=capability)
    )

    response = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers={"Authorization": "Bearer test-ephemeral-token"},
    )

    assert response.status_code == 200
    assert response.json() == capability.to_wire()
    assert response.headers["cache-control"] == "no-store"


def test_authority_process_spec_keeps_token_out_of_command(tmp_path) -> None:
    spec = build_authority_process_spec(
        token="test-ephemeral-token",
        port=54321,
        source_root=tmp_path,
    )

    assert spec.command[-2:] == ("--no-access-log", "--log-level=warning")
    assert "test-ephemeral-token" not in " ".join(spec.command)
    assert spec.environment["ROC_AUTHORITY_TOKEN"] == "test-ephemeral-token"
    assert spec.environment["PYTHONPATH"].split(";")[0] == str(tmp_path)


def _headers(token: str = "test-ephemeral-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _executable_client() -> tuple[TestClient, AuthorityJobManager]:
    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        del hooks
        return _result(job)

    manager = AuthorityJobManager(runner=runner)
    client = TestClient(
        create_authority_app(token="test-ephemeral-token", job_manager=manager)
    )
    return client, manager


def test_all_job_routes_require_authentication_without_logging_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "test-ephemeral-token-never-log"
    client = TestClient(create_authority_app(token=token))
    routes = (
        ("post", "/api/rate-of-closure/v1/regional-ground/jobs"),
        ("get", "/api/rate-of-closure/v1/regional-ground/jobs/example"),
        ("post", "/api/rate-of-closure/v1/regional-ground/jobs/example/cancel"),
        ("get", "/api/rate-of-closure/v1/regional-ground/jobs/example/result"),
    )

    with caplog.at_level(logging.DEBUG):
        responses = [getattr(client, method)(path) for method, path in routes]

    assert [response.status_code for response in responses] == [401, 401, 401, 401]
    assert token not in caplog.text


def test_default_fail_closed_authority_rejects_job_submission() -> None:
    client = TestClient(create_authority_app(token="test-ephemeral-token"))

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/jobs",
        content=regional_ground_execution_job_to_json(_job()),
        headers=_headers(),
    )

    assert response.status_code == 503
    assert response.json()["code"] == "execution_unavailable"


def test_submit_status_and_complete_result_round_trip() -> None:
    client, manager = _executable_client()
    job = _job()
    submitted = client.post(
        "/api/rate-of-closure/v1/regional-ground/jobs",
        content=regional_ground_execution_job_to_json(job),
        headers=_headers(),
    )

    assert submitted.status_code == 202
    assert submitted.headers["cache-control"] == "no-store"
    manager.wait_for_terminal(job.job_id, timeout_s=2.0)
    status = client.get(
        f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}",
        headers=_headers(),
    )
    assert status.status_code == 200
    result = client.get(
        f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}/result",
        headers=_headers(),
    )
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert regional_ground_execution_result_from_json(
        result.text, expected_job=job
    ) == _result(job)


def test_injected_test_runner_does_not_promote_execution_capability() -> None:
    client, _manager = _executable_client()
    response = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["regional_ground_execution"] is False


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Authorization": "Bearer test-ephemeral-token"}, "{}", 415),
        (_headers(), '{"job_id":"one","job_id":"two"}', 400),
        (_headers(), '{"schema_version":true}', 400),
        (
            _headers(),
            "é" * 524_289,
            413,
        ),
    ],
    ids=("wrong-media", "duplicate", "typed-invalid", "oversized-utf8"),
)
def test_submit_rejects_wrong_media_duplicate_and_oversized_bodies(
    headers: dict[str, str], body: str, expected: int
) -> None:
    client, _manager = _executable_client()

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/jobs",
        content=body.encode("utf-8"),
        headers=headers,
    )

    assert response.status_code == expected


def test_result_is_unavailable_until_complete_and_cancel_is_idempotent() -> None:
    entered = threading.Event()
    release = threading.Event()

    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        entered.set()
        assert release.wait(timeout=2.0)
        return _result(job)

    manager = AuthorityJobManager(runner=runner)
    client = TestClient(
        create_authority_app(token="test-ephemeral-token", job_manager=manager)
    )
    job = _job()
    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/jobs",
        content=regional_ground_execution_job_to_json(job),
        headers=_headers(),
    )
    assert response.status_code == 202
    assert entered.wait(timeout=2.0)

    pending = client.get(
        f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}/result",
        headers=_headers(),
    )
    cancelled = client.post(
        f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}/cancel",
        headers=_headers(),
    )
    cancelled_again = client.post(
        f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}/cancel",
        headers=_headers(),
    )
    release.set()
    terminal = manager.wait_for_terminal(job.job_id, timeout_s=2.0)

    assert pending.status_code == 409
    assert cancelled.status_code == 202
    assert cancelled_again.status_code == 202
    assert terminal.status.value == "cancelled"
    assert json.loads(cancelled.text)["status"] == "cancel_requested"


def test_invalid_and_unknown_job_paths_are_indistinguishable() -> None:
    client = TestClient(create_authority_app(token="test-ephemeral-token"))

    invalid = client.get(
        "/api/rate-of-closure/v1/regional-ground/jobs/not%20stable",
        headers=_headers(),
    )
    unknown = client.get(
        "/api/rate-of-closure/v1/regional-ground/jobs/unknown-job",
        headers=_headers(),
    )

    assert invalid.status_code == 404
    assert unknown.status_code == 404
    assert invalid.json() == unknown.json()
