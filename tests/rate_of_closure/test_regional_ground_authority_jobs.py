"""In-memory lifecycle tests for regional-ground authority jobs."""

from __future__ import annotations

import threading

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    build_regional_ground_execution_job,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
    build_regional_ground_execution_result,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)
from rate_of_closure.variation.scalar_ensemble_wire import (
    scalar_ensemble_dataset_from_wire,
)
from rate_of_closure.web_authority.jobs import (
    AuthorityExecutionUnavailable,
    AuthorityJobConflict,
    AuthorityJobManager,
    AuthorityJobResultUnavailable,
    AuthorityJobStatus,
)
from tests.rate_of_closure.test_regional_ground_execution_result import (
    _dataset_payload,
)
from tests.rate_of_closure.test_regional_ground_execution_result import (
    _job as golden_job,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _job(job_id: str = "driver-ground-study-1729") -> RegionalGroundExecutionJob:
    source = golden_job()
    if job_id == source.job_id:
        return source
    return build_regional_ground_execution_job(
        job_id=job_id,
        launch=source.launch.launch,
        flight=source.flight,
        transfer=source.transfer,
        capture_speed_m_s=source.capture_speed_m_s,
        execution_options=source.execution_options,
        regional_execution_options=source.regional_execution_options,
        variation_request=source.variation_request,
        producer=source.provenance.producer,
        producer_version=source.provenance.producer_version,
        source_revision=source.provenance.source_revision,
    )


def _result(job: RegionalGroundExecutionJob) -> RegionalGroundExecutionResult:
    dataset = scalar_ensemble_dataset_from_wire(_dataset_payload())
    return build_regional_ground_execution_result(job, dataset)


def test_manager_rejects_submission_without_a_qualified_runner() -> None:
    manager = AuthorityJobManager()

    with pytest.raises(AuthorityExecutionUnavailable):
        manager.submit(_job())

    assert manager.retained_job_count == 0


def test_one_active_job_cancel_forwards_and_precludes_late_publication() -> None:
    entered = threading.Event()
    release = threading.Event()
    cancellation_seen = threading.Event()

    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        entered.set()
        assert release.wait(timeout=2.0)
        assert hooks.cancellation_requested is not None
        if hooks.cancellation_requested():
            cancellation_seen.set()
        return _result(job)

    manager = AuthorityJobManager(runner=runner)
    submitted = manager.submit(_job())
    assert submitted.status is AuthorityJobStatus.QUEUED
    assert entered.wait(timeout=2.0)

    with pytest.raises(AuthorityJobConflict):
        manager.submit(_job("second-ground-job"))
    with pytest.raises(AuthorityJobResultUnavailable):
        manager.result(submitted.job_id)

    requested = manager.cancel(submitted.job_id)
    assert requested.status is AuthorityJobStatus.CANCEL_REQUESTED
    release.set()
    terminal = manager.wait_for_terminal(submitted.job_id, timeout_s=2.0)

    assert cancellation_seen.is_set()
    assert terminal.status is AuthorityJobStatus.CANCELLED
    assert terminal.result_available is False
    assert terminal.failure is None
    with pytest.raises(AuthorityJobResultUnavailable):
        manager.result(submitted.job_id)


def test_success_publishes_only_a_complete_job_bound_result() -> None:
    progress_seen: list[tuple[int, int]] = []

    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        assert hooks.progress_callback is not None
        hooks.progress_callback(GroundRegionalVariationProgress(4, 4))
        return _result(job)

    manager = AuthorityJobManager(runner=runner)
    snapshot = manager.submit(_job())
    terminal = manager.wait_for_terminal(snapshot.job_id, timeout_s=2.0)
    progress_seen.append((terminal.completed, terminal.total))

    assert terminal.status is AuthorityJobStatus.SUCCEEDED
    assert terminal.result_available is True
    assert progress_seen == [(4, 4)]
    assert manager.result(snapshot.job_id) == _result(_job())


def test_typed_failure_never_publishes_partial_or_exception_detail() -> None:
    secret_detail = "Bearer must-not-leak-through-status"

    def runner(
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        del job, hooks
        raise GroundRegionalVariationFailed(
            GroundRegionalVariationFailureStage.EXECUTOR,
            completed=1,
            total=4,
            cause=RuntimeError(secret_detail),
        )

    manager = AuthorityJobManager(runner=runner)
    submitted = manager.submit(_job())
    terminal = manager.wait_for_terminal(submitted.job_id, timeout_s=2.0)
    wire = terminal.to_wire()

    assert terminal.status is AuthorityJobStatus.FAILED
    assert terminal.failure is not None
    assert terminal.failure.code == "execution_failed"
    assert terminal.failure.stage == "executor"
    assert terminal.completed == 1
    assert secret_detail not in repr(wire)
    with pytest.raises(AuthorityJobResultUnavailable):
        manager.result(submitted.job_id)


def test_wrong_job_result_is_rejected_before_publication() -> None:
    manager = AuthorityJobManager(runner=lambda _job, _hooks: _result(golden_job()))
    submitted = manager.submit(_job("different-ground-job"))
    terminal = manager.wait_for_terminal(submitted.job_id, timeout_s=2.0)

    assert terminal.status is AuthorityJobStatus.FAILED
    assert terminal.failure is not None
    assert terminal.failure.to_wire() == {
        "code": "result_rejected",
        "stage": "result_validation",
    }
    assert terminal.result_available is False


def test_terminal_retention_is_bounded_and_oldest_job_is_evicted() -> None:
    manager = AuthorityJobManager(
        runner=lambda job, _hooks: _result(job),
        max_retained_jobs=2,
    )
    job_ids = [
        "retained-ground-job-1",
        "retained-ground-job-2",
        "retained-ground-job-3",
    ]
    for job_id in job_ids:
        submitted = manager.submit(_job(job_id))
        manager.wait_for_terminal(submitted.job_id, timeout_s=2.0)

    assert manager.retained_job_count == 2
    with pytest.raises(KeyError):
        manager.status(job_ids[0])
    assert manager.status(job_ids[1]).status is AuthorityJobStatus.SUCCEEDED
    assert manager.status(job_ids[2]).status is AuthorityJobStatus.SUCCEEDED
