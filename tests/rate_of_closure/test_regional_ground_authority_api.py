"""Fail-closed contract tests for the regional-ground browser authority."""

from __future__ import annotations

import json
import logging
import os
import threading

import pytest
from fastapi.testclient import TestClient

from rate_of_closure.application._regional_ground_execution_job_values import (
    FlightLaunchInput,
)
from rate_of_closure.application.flight_execution_profiles import (
    FlightExecutionProfileQualificationError,
    FlightExecutionQualification,
    FlightExecutionQualificationReason,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    regional_ground_execution_job_from_json,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
)
from rate_of_closure.application.regional_ground_job_preparation_request import (
    RegionalGroundJobPreparationRequest,
    regional_ground_job_preparation_request_to_json,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.api import (
    CAPABILITY_PATH,
    JOB_COLLECTION_PATH,
    create_authority_app,
)
from rate_of_closure.web_authority.capability import (
    AUTHORITY_CAPABILITY_SCHEMA_VERSION,
    QUALIFIED_EXECUTION_CAPABILITY,
    AuthorityCapability,
)
from rate_of_closure.web_authority.job_store import AuthorityJobStore
from rate_of_closure.web_authority.jobs import (
    AuthorityExecutionUnavailable,
    AuthorityJobManager,
)
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


def test_qualified_capability_is_internally_consistent() -> None:
    assert QUALIFIED_EXECUTION_CAPABILITY.to_wire() == {
        "schema_version": AUTHORITY_CAPABILITY_SCHEMA_VERSION,
        "authority_id": "rate-of-closure-python-authority",
        "authority_version": "1",
        "available": True,
        "regional_ground_execution": True,
        "reason_code": "qualified_execution_profile",
        "detail": "Qualified Python regional-ground execution is available.",
    }

    with pytest.raises(ValueError, match="qualified"):
        AuthorityCapability(
            available=True,
            regional_ground_execution=True,
            reason_code="runner_not_started",
            detail="Runner is unavailable.",
        )


def test_direct_capability_construction_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        AuthorityCapability(
            available=False,
            regional_ground_execution=False,
            reason_code="unknown_reason",  # type: ignore[arg-type]
            detail="Authority is unavailable.",
        )


@pytest.mark.parametrize("reason", [17, None])
def test_direct_capability_construction_rejects_non_text_reason(
    reason: object,
) -> None:
    with pytest.raises(TypeError, match="reason"):
        AuthorityCapability(
            available=False,
            regional_ground_execution=False,
            reason_code=reason,  # type: ignore[arg-type]
            detail="Authority is unavailable.",
        )


def test_direct_capability_construction_rejects_non_text_detail() -> None:
    with pytest.raises(TypeError, match="detail"):
        AuthorityCapability(
            available=False,
            regional_ground_execution=False,
            reason_code="runner_not_started",
            detail=17,  # type: ignore[arg-type]
        )


def test_capability_json_parser_rejects_duplicate_and_split_brain_states() -> None:
    exact = json.dumps(QUALIFIED_EXECUTION_CAPABILITY.to_wire())
    assert AuthorityCapability.from_json(exact) == QUALIFIED_EXECUTION_CAPABILITY

    with pytest.raises(ValueError, match="duplicate"):
        AuthorityCapability.from_json(
            exact.replace('"available": true', '"available": true, "available": true')
        )
    with pytest.raises(ValueError, match="consistent"):
        AuthorityCapability.from_json(
            exact.replace(
                '"regional_ground_execution": true',
                '"regional_ground_execution": false',
            )
        )


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
    state_root = tmp_path / "state"
    state_root.mkdir()
    spec = build_authority_process_spec(
        token="test-ephemeral-token",
        port=54321,
        source_root=tmp_path,
        state_root=state_root,
    )

    assert spec.command[-2:] == ("-m", "rate_of_closure.web_authority.child")
    assert "test-ephemeral-token" not in " ".join(spec.command)
    assert "54321" not in " ".join(spec.command)
    assert spec.environment["ROC_AUTHORITY_TOKEN"] == "test-ephemeral-token"
    assert spec.environment["ROC_AUTHORITY_PORT"] == "54321"
    assert spec.environment["ROC_AUTHORITY_STATE_ROOT"] == str(state_root)
    assert spec.environment["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)


def test_authority_process_spec_accepts_only_bounded_import_factory(tmp_path) -> None:
    factory = "tests.rate_of_closure.test_module:create_app"
    spec = build_authority_process_spec(
        token="test-ephemeral-token",
        port=54321,
        source_root=tmp_path,
        app_factory=factory,
    )

    assert spec.environment["ROC_AUTHORITY_APP_FACTORY"] == factory
    for invalid in (
        "tests.module",
        "tests.module:create-app",
        " tests.module:create_app",
        "tests..module:create_app",
        f"tests.module:{'x' * 241}",
    ):
        with pytest.raises(ValueError, match="app_factory"):
            build_authority_process_spec(
                token="test-ephemeral-token",
                port=54321,
                source_root=tmp_path,
                app_factory=invalid,
            )


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
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
        )
    )
    return client, manager


@pytest.mark.asyncio
async def test_exceptional_lifespan_exit_still_closes_the_manager() -> None:
    manager = AuthorityJobManager(runner=lambda job, _hooks: _result(job))
    app = create_authority_app(
        token="test-ephemeral-token",
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=manager,
    )

    with pytest.raises(RuntimeError, match="lifespan failure"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("lifespan failure")

    assert manager.execution_available is False


def test_all_job_routes_require_authentication_without_logging_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "test-ephemeral-token-never-log"
    client = TestClient(create_authority_app(token=token))
    routes = (
        ("post", "/api/rate-of-closure/v1/regional-ground/job-preparations"),
        ("post", "/api/rate-of-closure/v1/regional-ground/jobs"),
        ("get", "/api/rate-of-closure/v1/regional-ground/jobs/example"),
        ("post", "/api/rate-of-closure/v1/regional-ground/jobs/example/cancel"),
        ("get", "/api/rate-of-closure/v1/regional-ground/jobs/example/result"),
    )

    with caplog.at_level(logging.DEBUG):
        responses = [getattr(client, method)(path) for method, path in routes]

    assert [response.status_code for response in responses] == [401, 401, 401, 401, 401]
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


def test_default_fail_closed_authority_rejects_job_preparation() -> None:
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "prepared-job-001", source.launch, source.variation_request
    )
    client = TestClient(create_authority_app(token="test-ephemeral-token"))

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/job-preparations",
        content=regional_ground_job_preparation_request_to_json(request),
        headers=_headers(),
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["code"] == "preparation_unavailable"


def test_prepare_returns_canonical_job_without_enqueueing_or_running() -> None:
    ran = threading.Event()

    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        del job, hooks
        ran.set()
        raise AssertionError("preparation must not run a study")

    manager = AuthorityJobManager(runner=runner)
    client = TestClient(
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
        )
    )
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "prepared-job-001",
        FlightLaunchInput(source.launch.launch),
        source.variation_request,
    )

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/job-preparations",
        content=regional_ground_job_preparation_request_to_json(request),
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    prepared = regional_ground_execution_job_from_json(response.text)
    assert prepared.job_id == "prepared-job-001"
    assert prepared.launch == request.launch
    assert prepared.variation_request == request.variation_request
    assert ran.is_set() is False
    with pytest.raises(KeyError):
        manager.status("prepared-job-001")


def test_prepare_reports_stable_qualification_failure_without_private_detail() -> None:
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "prepared-job-001", source.launch, source.variation_request
    )
    private = "C:/private/numerical-runtime.txt"

    def unavailable_profile(**_kwargs: object) -> object:
        error = FlightExecutionProfileQualificationError(
            FlightExecutionQualification(
                FlightExecutionQualificationReason.RECOMPUTATION_FAILED,
                "waterloo_penner",
                "tools-core/1.0.0",
            )
        )
        error.add_note(private)
        raise error

    manager = AuthorityJobManager(runner=lambda job, _hooks: _result(job))
    client = TestClient(
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
            job_preparer=unavailable_profile,
        )
    )

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/job-preparations",
        content=regional_ground_job_preparation_request_to_json(request),
        headers=_headers(),
    )

    assert response.status_code == 422
    assert response.json()["code"] == "preparation_failed"
    assert private not in response.text


def test_prepare_rejects_a_valid_but_substituted_preparer_response() -> None:
    source = _job()
    request = RegionalGroundJobPreparationRequest(
        "prepared-job-001", source.launch, source.variation_request
    )
    manager = AuthorityJobManager(runner=lambda job, _hooks: _result(job))
    client = TestClient(
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
            job_preparer=lambda **_kwargs: source,
        )
    )

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/job-preparations",
        content=regional_ground_job_preparation_request_to_json(request),
        headers=_headers(),
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_preparation"


@pytest.mark.parametrize(
    ("headers", "body", "expected"),
    [
        ({"Authorization": "Bearer test-ephemeral-token"}, "{}", 415),
        (_headers(), '{"job_id":"one","job_id":"two"}', 400),
        (_headers(), '{"schema_version":true}', 400),
        (_headers(), "é" * 524_289, 413),
    ],
    ids=("wrong-media", "duplicate", "typed-invalid", "oversized-utf8"),
)
def test_prepare_rejects_wrong_media_duplicate_and_oversized_bodies(
    headers: dict[str, str], body: str, expected: int
) -> None:
    client, _manager = _executable_client()

    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/job-preparations",
        content=body.encode("utf-8"),
        headers=headers,
    )

    assert response.status_code == expected


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


def test_attached_runner_and_capability_are_advertised_together() -> None:
    client, _manager = _executable_client()
    response = client.get(
        "/api/rate-of-closure/v1/capabilities",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["regional_ground_execution"] is True


def test_capability_fails_closed_after_durable_acceptance_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AuthorityJobStore(tmp_path / "authority.sqlite3", max_retained_jobs=4)
    manager = AuthorityJobManager(runner=lambda job, _hooks: _result(job), store=store)
    client = TestClient(
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
        )
    )

    def fail_replace(_records: object) -> None:
        raise RuntimeError("private persistence detail")

    monkeypatch.setattr(store, "replace", fail_replace)
    submitted = client.post(
        JOB_COLLECTION_PATH,
        content=regional_ground_execution_job_to_json(_job()),
        headers=_headers(),
    )
    capability = client.get(CAPABILITY_PATH, headers=_headers())

    assert submitted.status_code == 503
    assert "private persistence detail" not in submitted.text
    assert capability.json()["available"] is False
    assert capability.json()["reason_code"] == "runner_not_started"
    manager.close()


def test_executable_capability_requires_an_attached_runner() -> None:
    with pytest.raises(ValueError, match="runner"):
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
        )


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
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
        )
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


def test_cancel_reports_stable_unavailable_when_durable_state_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AuthorityJobManager(runner=lambda job, _hooks: _result(job))
    client = TestClient(
        create_authority_app(
            token="test-ephemeral-token",
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            job_manager=manager,
        )
    )

    def unavailable(_job_id: str) -> None:
        raise AuthorityExecutionUnavailable("C:/private/authority.sqlite3")

    monkeypatch.setattr(manager, "cancel", unavailable)
    response = client.post(
        "/api/rate-of-closure/v1/regional-ground/jobs/example/cancel",
        headers=_headers(),
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["code"] == "execution_unavailable"
    assert "private" not in response.text


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
