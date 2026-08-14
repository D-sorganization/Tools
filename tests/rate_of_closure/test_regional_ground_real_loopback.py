"""Real-process qualification for the PyQt regional-ground loopback adapter."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobStatus,
    regional_ground_authority_job_status_from_json,
)
from rate_of_closure.application.regional_ground_authority_transport import (
    LoopbackAuthorityHttpTransport,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
    RegionalGroundExecutionResult,
    regional_ground_execution_result_to_json,
)
from rate_of_closure.application.regional_ground_loopback_submitter import (
    LoopbackRegionalGroundSubmitter,
    RegionalGroundAuthorityPollPolicy,
    regional_ground_submitter_if_available,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.api import (
    CAPABILITY_PATH,
    JOB_COLLECTION_PATH,
    create_authority_app,
)
from rate_of_closure.web_authority.capability import (
    QUALIFIED_EXECUTION_CAPABILITY,
)
from rate_of_closure.web_authority.job_store import AuthorityJobStore
from rate_of_closure.web_authority.jobs import AuthorityJobManager, AuthorityJobRunner
from rate_of_closure.web_authority.production_runner import (
    run_regional_ground_production_job,
)
from rate_of_closure.web_authority.runtime import AuthorityRuntime, start_authority
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job, _result

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

_TOKEN_ENV = "ROC_AUTHORITY_TOKEN"
_STATE_ROOT_ENV = "ROC_AUTHORITY_STATE_ROOT"
_SOURCE_ROOT = Path(__file__).parents[2] / "src"
_POLICY = RegionalGroundAuthorityPollPolicy(
    poll_timeout_s=15.0,
    initial_interval_s=0.01,
    maximum_interval_s=0.05,
    backoff_multiplier=1.5,
    request_timeout_s=1.0,
    shutdown_timeout_s=3.0,
)


def create_preflight_authority_app() -> FastAPI:
    """Build the real API with the production fail-closed preflight runner."""
    return create_authority_app(
        token=os.environ[_TOKEN_ENV],
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=AuthorityJobManager(runner=run_regional_ground_production_job),
    )


def _wait_for_cancel(
    job: RegionalGroundExecutionJob, hooks: GroundRegionalVariationHooks
) -> RegionalGroundExecutionResult:
    callback = hooks.cancellation_requested
    if callback is None:
        raise AssertionError("authority manager must inject cancellation")
    while True:
        if callback():
            raise GroundRegionalVariationCancelled(0, job.execution_options.max_trials)
        time.sleep(0.005)


def create_cancellable_authority_app() -> FastAPI:
    """Build the real API with a non-physical cancellation-only test runner."""
    return create_authority_app(
        token=os.environ[_TOKEN_ENV],
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=AuthorityJobManager(runner=_wait_for_cancel),
    )


def _durable_manager(runner: AuthorityJobRunner) -> AuthorityJobManager:
    root = Path(os.environ[_STATE_ROOT_ENV])
    return AuthorityJobManager(
        runner=runner,
        store=AuthorityJobStore(root / "authority.v1.sqlite3", max_retained_jobs=4),
    )


def create_durable_test_authority_app() -> FastAPI:
    """Build a fast durable authority for process-restart qualification."""
    return create_authority_app(
        token=os.environ[_TOKEN_ENV],
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=_durable_manager(lambda job, _hooks: _result(job)),
    )


def create_durable_blocking_authority_app() -> FastAPI:
    """Build a durable authority whose worker survives until hard process loss."""
    return create_authority_app(
        token=os.environ[_TOKEN_ENV],
        capability=QUALIFIED_EXECUTION_CAPABILITY,
        job_manager=_durable_manager(_wait_for_cancel),
    )


@contextmanager
def _authority(factory: str) -> Iterator[AuthorityRuntime]:
    runtime = start_authority(source_root=_SOURCE_ROOT, app_factory=factory)
    try:
        yield runtime
    finally:
        runtime.close()
        assert runtime.process.poll() is not None


def _factory(name: str) -> str:
    return f"tests.rate_of_closure.test_regional_ground_real_loopback:{name}"


def test_default_environment_factory_is_qualified_and_executes(tmp_path: Path) -> None:
    runtime = start_authority(source_root=_SOURCE_ROOT, state_root=tmp_path)
    try:
        transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)
        capability_response = transport.request("GET", CAPABILITY_PATH, None, 4_096)
        assert json.loads(capability_response.body) == (
            QUALIFIED_EXECUTION_CAPABILITY.to_wire()
        )
        submitter = regional_ground_submitter_if_available(
            runtime=runtime,
            capability=QUALIFIED_EXECUTION_CAPABILITY,
            policy=_POLICY,
        )
        assert submitter is not None
        result = submitter(_job(), GroundRegionalVariationHooks())
        result.assert_matches_job(_job())
        submitter.close()
        transport.close()
    finally:
        runtime.close()


def test_real_qualified_authority_auth_status_cancel_and_result_contracts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = _job()
    with _authority(_factory("create_preflight_authority_app")) as runtime:
        assert runtime.token not in " ".join(runtime.process.args)
        submitter_from_capability = regional_ground_submitter_if_available(
            runtime=runtime, capability=QUALIFIED_EXECUTION_CAPABILITY
        )
        assert submitter_from_capability is not None
        submitter_from_capability.close()

        wrong_runtime = AuthorityRuntime(
            process=runtime.process,
            token=f"wrong-{runtime.token}",
            port=runtime.port,
        )
        wrong = LoopbackAuthorityHttpTransport(wrong_runtime, timeout_s=1.0)
        with caplog.at_level(logging.DEBUG):
            unauthorized = wrong.request("GET", CAPABILITY_PATH, None, 4_096)
        assert unauthorized.status == 401
        assert runtime.token not in caplog.text
        assert runtime.token.encode() not in unauthorized.body
        wrong.close()

        capability_transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)
        capability_response = capability_transport.request(
            "GET", CAPABILITY_PATH, None, 4_096
        )
        assert json.loads(capability_response.body) == (
            QUALIFIED_EXECUTION_CAPABILITY.to_wire()
        )
        capability_transport.close()

        submitter = LoopbackRegionalGroundSubmitter(
            LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0), policy=_POLICY
        )
        published = submitter(job, GroundRegionalVariationHooks())
        published.assert_matches_job(job)
        submitter.close()

        transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)
        job_path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
        status = transport.request("GET", job_path, None, 4_096)
        snapshot = regional_ground_authority_job_status_from_json(
            status.body.decode("utf-8"), job
        )
        assert snapshot.status is AuthorityJobStatus.SUCCEEDED
        assert snapshot.failure is None
        assert snapshot.result_available is True

        cancelled = transport.request("POST", f"{job_path}/cancel", None, 4_096)
        assert cancelled.status == 202
        assert json.loads(cancelled.body)["status"] == "succeeded"
        result = transport.request(
            "GET",
            f"{job_path}/result",
            None,
            MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
        )
        assert result.status == 200
        assert result.body.decode("utf-8") == regional_ground_execution_result_to_json(
            published
        )
        transport.close()


def test_real_loopback_submitter_close_posts_cancel_and_joins() -> None:
    with _authority(_factory("create_cancellable_authority_app")) as runtime:
        submitter = LoopbackRegionalGroundSubmitter(
            LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0), policy=_POLICY
        )
        progress_seen = threading.Event()
        terminals: list[BaseException] = []

        def run() -> None:
            try:
                submitter(
                    _job(),
                    GroundRegionalVariationHooks(
                        progress_callback=lambda _progress: progress_seen.set()
                    ),
                )
            except BaseException as error:
                terminals.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert progress_seen.wait(2.0)
        started_close = time.monotonic()
        submitter.close(timeout_s=3.0)
        assert time.monotonic() - started_close < 3.0
        worker.join(2.0)

        assert not worker.is_alive()
        assert len(terminals) == 1
        assert isinstance(terminals[0], GroundRegionalVariationCancelled)
        assert runtime.process.poll() is None


def test_complete_result_survives_hard_authority_process_restart(
    tmp_path: Path,
) -> None:
    factory = _factory("create_durable_test_authority_app")
    first = start_authority(
        source_root=_SOURCE_ROOT, app_factory=factory, state_root=tmp_path
    )
    job = _job()
    submitter = LoopbackRegionalGroundSubmitter(
        LoopbackAuthorityHttpTransport(first, timeout_s=1.0), policy=_POLICY
    )
    expected = submitter(job, GroundRegionalVariationHooks())
    submitter.close()
    first.process.kill()
    first.process.wait(timeout=5.0)

    second = start_authority(
        source_root=_SOURCE_ROOT, app_factory=factory, state_root=tmp_path
    )
    try:
        transport = LoopbackAuthorityHttpTransport(second, timeout_s=1.0)
        path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
        status = transport.request("GET", path, None, 4_096)
        result = transport.request(
            "GET", f"{path}/result", None, MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES
        )
        assert (
            regional_ground_authority_job_status_from_json(
                status.body.decode("utf-8"), job
            ).status
            is AuthorityJobStatus.SUCCEEDED
        )
        assert result.body.decode("utf-8") == regional_ground_execution_result_to_json(
            expected
        )
        transport.close()
    finally:
        second.close()


def test_hard_loss_marks_running_job_failed_without_automatic_replay(
    tmp_path: Path,
) -> None:
    factory = _factory("create_durable_blocking_authority_app")
    first = start_authority(
        source_root=_SOURCE_ROOT, app_factory=factory, state_root=tmp_path
    )
    transport = LoopbackAuthorityHttpTransport(first, timeout_s=1.0)
    job = _job()
    submitted = transport.request(
        "POST",
        JOB_COLLECTION_PATH,
        regional_ground_execution_job_to_json(job).encode("utf-8"),
        4_096,
    )
    assert submitted.status == 202
    path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = transport.request("GET", path, None, 4_096)
        snapshot = regional_ground_authority_job_status_from_json(
            status.body.decode("utf-8"), job
        )
        if snapshot.status is AuthorityJobStatus.RUNNING:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("durable job did not reach running")
    transport.close()
    first.process.kill()
    first.process.wait(timeout=5.0)

    second = start_authority(
        source_root=_SOURCE_ROOT, app_factory=factory, state_root=tmp_path
    )
    try:
        recovered_transport = LoopbackAuthorityHttpTransport(second, timeout_s=1.0)
        recovered = recovered_transport.request("GET", path, None, 4_096)
        snapshot = regional_ground_authority_job_status_from_json(
            recovered.body.decode("utf-8"), job
        )
        assert snapshot.status is AuthorityJobStatus.FAILED
        assert snapshot.failure is not None
        assert snapshot.failure.stage == "authority_restart"
        recovered_transport.close()
    finally:
        second.close()
