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
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.application.regional_ground_loopback_submitter import (
    LoopbackRegionalGroundSubmitter,
    RegionalGroundAuthorityPollPolicy,
    regional_ground_submitter_if_available,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
)
from rate_of_closure.web_authority.api import (
    CAPABILITY_PATH,
    JOB_COLLECTION_PATH,
    create_authority_app,
)
from rate_of_closure.web_authority.capability import DEFAULT_UNAVAILABLE_CAPABILITY
from rate_of_closure.web_authority.jobs import AuthorityJobManager
from rate_of_closure.web_authority.production_runner import (
    run_regional_ground_production_job,
)
from rate_of_closure.web_authority.runtime import AuthorityRuntime, start_authority
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

_TOKEN_ENV = "ROC_AUTHORITY_TOKEN"
_SOURCE_ROOT = Path(__file__).parents[2] / "src"
_POLICY = RegionalGroundAuthorityPollPolicy(
    poll_timeout_s=3.0,
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
        job_manager=AuthorityJobManager(runner=_wait_for_cancel),
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


def test_real_preflight_authority_auth_status_cancel_and_result_contracts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = _job()
    with _authority(_factory("create_preflight_authority_app")) as runtime:
        assert runtime.token not in " ".join(runtime.process.args)
        assert (
            regional_ground_submitter_if_available(
                runtime=runtime, capability=DEFAULT_UNAVAILABLE_CAPABILITY
            )
            is None
        )

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

        submitter = LoopbackRegionalGroundSubmitter(
            LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0), policy=_POLICY
        )
        with pytest.raises(GroundRegionalVariationFailed) as raised:
            submitter(job, GroundRegionalVariationHooks())
        assert raised.value.stage is GroundRegionalVariationFailureStage.PREFLIGHT
        assert (raised.value.completed, raised.value.total) == (0, 4)
        assert runtime.token not in str(raised.value)
        submitter.close()

        transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)
        job_path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
        status = transport.request("GET", job_path, None, 4_096)
        snapshot = regional_ground_authority_job_status_from_json(
            status.body.decode("utf-8"), job
        )
        assert snapshot.status is AuthorityJobStatus.FAILED
        assert snapshot.failure is not None
        assert snapshot.failure.stage == "preflight"

        cancelled = transport.request("POST", f"{job_path}/cancel", None, 4_096)
        assert cancelled.status == 202
        assert json.loads(cancelled.body)["status"] == "failed"
        result = transport.request("GET", f"{job_path}/result", None, 4_096)
        assert result.status == 409
        assert json.loads(result.body)["code"] == "result_unavailable"
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
