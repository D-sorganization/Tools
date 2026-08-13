"""Bounded in-memory lifecycle authority for regional-ground jobs."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from rate_of_closure.application._workspace_validation import stable_id
from rate_of_closure.application.regional_ground_authority_status import (
    AUTHORITY_JOB_STATUS_SCHEMA_VERSION,
    AuthorityFailureCode,
    AuthorityFailureStage,
    AuthorityJobFailure,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)

DEFAULT_MAX_RETAINED_JOBS: Final = 4
MAX_RETAINED_JOBS_LIMIT: Final = 16


class AuthorityExecutionUnavailable(RuntimeError):
    """Raised when no qualified physical runner is attached."""


class AuthorityJobConflict(RuntimeError):
    """Raised when submission would violate the one-active-job invariant."""


class AuthorityJobResultUnavailable(RuntimeError):
    """Raised unless a complete validated result has been published."""


AuthorityJobRunner = Callable[
    [RegionalGroundExecutionJob, GroundRegionalVariationHooks],
    RegionalGroundExecutionResult,
]


@dataclass(slots=True)
class _JobRecord:
    job: RegionalGroundExecutionJob
    cancellation: threading.Event
    status: AuthorityJobStatus = AuthorityJobStatus.QUEUED
    completed: int = 0
    result: RegionalGroundExecutionResult | None = None
    failure: AuthorityJobFailure | None = None


_TERMINAL = frozenset(
    {
        AuthorityJobStatus.SUCCEEDED,
        AuthorityJobStatus.FAILED,
        AuthorityJobStatus.CANCELLED,
    }
)


class AuthorityJobManager:
    """Own one active job and a bounded set of terminal in-memory records."""

    def __init__(
        self,
        *,
        runner: AuthorityJobRunner | None = None,
        max_retained_jobs: int = DEFAULT_MAX_RETAINED_JOBS,
    ) -> None:
        if runner is not None and not callable(runner):
            raise TypeError("runner must be callable or None")
        if (
            type(max_retained_jobs) is not int
            or max_retained_jobs < 1
            or max_retained_jobs > MAX_RETAINED_JOBS_LIMIT
        ):
            raise ValueError("max_retained_jobs lies outside the supported bound")
        self._runner = runner
        self._max_retained_jobs = max_retained_jobs
        self._records: dict[str, _JobRecord] = {}
        self._terminal_order: deque[str] = deque()
        self._active_job_id: str | None = None
        self._worker: threading.Thread | None = None
        self._closed = False
        self._condition = threading.Condition(threading.RLock())

    @property
    def retained_job_count(self) -> int:
        """Return the bounded number of active plus retained terminal jobs."""
        with self._condition:
            return len(self._records)

    @property
    def execution_available(self) -> bool:
        """Return whether this manager owns a callable execution runner."""
        with self._condition:
            return self._runner is not None and not self._closed

    def close(self, timeout_s: float = 5.0) -> None:
        """Cancel and join the owned worker before authority shutdown."""
        if type(timeout_s) not in {int, float} or timeout_s < 0:
            raise ValueError("timeout_s must be a nonnegative number")
        with self._condition:
            self._closed = True
            active_id = self._active_job_id
            if active_id is not None:
                self._records[active_id].cancellation.set()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(float(timeout_s))
            if worker.is_alive():
                raise TimeoutError("authority job worker did not stop before timeout")

    def submit(self, job: RegionalGroundExecutionJob) -> AuthorityJobSnapshot:
        """Accept one exact job and start its injected runner on a daemon thread."""
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("job must be an exact RegionalGroundExecutionJob")
        job.__post_init__()
        if self._runner is None:
            raise AuthorityExecutionUnavailable("qualified execution is unavailable")
        record = _JobRecord(job=job, cancellation=threading.Event())
        worker = threading.Thread(
            target=self._run,
            args=(record,),
            name="regional-ground-authority-job",
            daemon=True,
        )
        with self._condition:
            if self._closed:
                raise AuthorityExecutionUnavailable("authority manager is closed")
            if self._active_job_id is not None:
                raise AuthorityJobConflict("one regional-ground job is already active")
            if job.job_id in self._records:
                raise AuthorityJobConflict("job_id is already retained")
            self._records[job.job_id] = record
            self._active_job_id = job.job_id
            queued = self._snapshot(record)
            self._worker = worker
            try:
                worker.start()
            except Exception:
                self._worker = None
                self._records.pop(job.job_id, None)
                self._active_job_id = None
                raise
        return queued

    def status(self, job_id: str) -> AuthorityJobSnapshot:
        """Return an immutable snapshot for one retained job identity."""
        with self._condition:
            return self._snapshot(self._record(job_id))

    def cancel(self, job_id: str) -> AuthorityJobSnapshot:
        """Request cooperative cancellation and make publication ineligible."""
        with self._condition:
            record = self._record(job_id)
            if record.status not in _TERMINAL:
                record.cancellation.set()
                record.status = AuthorityJobStatus.CANCEL_REQUESTED
                self._condition.notify_all()
            return self._snapshot(record)

    def result(self, job_id: str) -> RegionalGroundExecutionResult:
        """Return only a complete validated result from a successful job."""
        with self._condition:
            record = self._record(job_id)
            if (
                record.status is not AuthorityJobStatus.SUCCEEDED
                or record.result is None
            ):
                raise AuthorityJobResultUnavailable(
                    "complete regional-ground result is unavailable"
                )
            return record.result

    def wait_for_terminal(
        self, job_id: str, *, timeout_s: float
    ) -> AuthorityJobSnapshot:
        """Wait a bounded duration for a retained job to become terminal."""
        if type(timeout_s) not in (int, float) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        with self._condition:
            reached = self._condition.wait_for(
                lambda: self._record(job_id).status in _TERMINAL,
                timeout=float(timeout_s),
            )
            if not reached:
                raise TimeoutError("regional-ground authority job did not terminate")
            return self._snapshot(self._record(job_id))

    def _run(self, record: _JobRecord) -> None:
        with self._condition:
            if record.cancellation.is_set():
                self._finish(record, AuthorityJobStatus.CANCELLED)
                return
            record.status = AuthorityJobStatus.RUNNING
            self._condition.notify_all()
        hooks = GroundRegionalVariationHooks(
            progress_callback=lambda progress: self._report(record, progress),
            cancellation_requested=record.cancellation.is_set,
        )
        runner = self._runner
        if runner is None:
            self._finish_failed(record, "execution_failed", "runner")
            return
        try:
            result = runner(record.job, hooks)
        except GroundRegionalVariationCancelled as error:
            if not self._terminal_counts_match(record, error.completed, error.total):
                self._finish_failed(record, "execution_failed", "validation")
                return
            self._finish_cancelled(record, error.completed)
            return
        except GroundRegionalVariationFailed as error:
            if not self._terminal_counts_match(record, error.completed, error.total):
                self._finish_failed(record, "execution_failed", "validation")
                return
            self._finish_variation_failure(record, error)
            return
        except Exception:
            self._finish_failed(record, "execution_failed", "runner")
            return
        self._accept_result(record, result)

    def _report(
        self, record: _JobRecord, progress: GroundRegionalVariationProgress
    ) -> None:
        if type(progress) is not GroundRegionalVariationProgress:
            raise TypeError("progress must be exact GroundRegionalVariationProgress")
        with self._condition:
            total = record.job.execution_options.max_trials
            if progress.total != total or progress.completed < record.completed:
                raise ValueError("progress must be monotonic and match job total")
            record.completed = progress.completed
            self._condition.notify_all()

    def _accept_result(
        self, record: _JobRecord, result: RegionalGroundExecutionResult
    ) -> None:
        try:
            if type(result) is not RegionalGroundExecutionResult:
                raise TypeError("runner result must be exact")
            result.assert_matches_job(record.job)
        except (TypeError, ValueError):
            self._finish_failed(record, "result_rejected", "result_validation")
            return
        with self._condition:
            if record.cancellation.is_set():
                self._finish(record, AuthorityJobStatus.CANCELLED)
                return
            record.result = result
            record.completed = record.job.execution_options.max_trials
            self._finish(record, AuthorityJobStatus.SUCCEEDED)

    def _finish_cancelled(self, record: _JobRecord, completed: int) -> None:
        with self._condition:
            record.completed = completed
            self._finish(record, AuthorityJobStatus.CANCELLED)

    def _terminal_counts_match(
        self,
        record: _JobRecord,
        completed: int,
        total: int,
    ) -> bool:
        """Require terminal progress to belong to this exact submitted job."""
        with self._condition:
            return bool(
                total == record.job.execution_options.max_trials
                and record.completed <= completed <= total
            )

    def _finish_variation_failure(
        self, record: _JobRecord, error: GroundRegionalVariationFailed
    ) -> None:
        with self._condition:
            record.completed = error.completed
            self._finish_failed(record, "execution_failed", error.stage.value)

    def _finish_failed(
        self,
        record: _JobRecord,
        code: AuthorityFailureCode,
        stage: AuthorityFailureStage,
    ) -> None:
        with self._condition:
            record.failure = AuthorityJobFailure(code, stage)
            self._finish(record, AuthorityJobStatus.FAILED)

    def _finish(self, record: _JobRecord, status: AuthorityJobStatus) -> None:
        record.status = status
        self._active_job_id = None
        self._terminal_order.append(record.job.job_id)
        while len(self._terminal_order) > self._max_retained_jobs:
            evicted = self._terminal_order.popleft()
            self._records.pop(evicted, None)
        self._condition.notify_all()

    def _record(self, job_id: str) -> _JobRecord:
        try:
            stable_id(job_id, "job_id")
        except (TypeError, ValueError):
            raise KeyError(job_id) from None
        try:
            return self._records[job_id]
        except KeyError:
            raise KeyError(job_id) from None

    @staticmethod
    def _snapshot(record: _JobRecord) -> AuthorityJobSnapshot:
        return AuthorityJobSnapshot(
            job_id=record.job.job_id,
            job_sha256=record.job.job_sha256,
            status=record.status,
            completed=record.completed,
            total=record.job.execution_options.max_trials,
            result_available=record.result is not None,
            failure=record.failure,
        )


__all__ = [
    "AUTHORITY_JOB_STATUS_SCHEMA_VERSION",
    "AuthorityExecutionUnavailable",
    "AuthorityJobConflict",
    "AuthorityJobFailure",
    "AuthorityJobManager",
    "AuthorityJobResultUnavailable",
    "AuthorityJobSnapshot",
    "AuthorityJobStatus",
]
