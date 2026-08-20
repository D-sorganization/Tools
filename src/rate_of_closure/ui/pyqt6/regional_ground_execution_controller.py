"""Widget-free Qt port for qualified regional-ground job submission."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol, cast

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)

_SHUTDOWN_WAIT_MS = 10_000


class RegionalGroundExecutionSubmitter(Protocol):
    """Injected application authority; no physical implementation is implied."""

    def __call__(
        self,
        job: RegionalGroundExecutionJob,
        hooks: GroundRegionalVariationHooks,
    ) -> RegionalGroundExecutionResult:
        """Return one complete job-bound result or raise a typed terminal."""


class RegionalGroundExecutionWorker(QThread):
    """Submit one immutable job without blocking the GUI event loop."""

    progressed = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    succeeded = pyqtSignal(object)  # noqa: N815
    cancelled = pyqtSignal(object)  # noqa: N815
    failed = pyqtSignal(object)  # noqa: N815

    def __init__(
        self,
        job: RegionalGroundExecutionJob,
        submitter: RegionalGroundExecutionSubmitter,
    ) -> None:
        super().__init__()
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("job must be an exact RegionalGroundExecutionJob")
        if not callable(submitter):
            raise TypeError("submitter must be callable")
        job.__post_init__()
        self._job = job
        self._submitter = submitter
        self._cancel_event = threading.Event()
        self._last_completed = 0

    @property
    def job(self) -> RegionalGroundExecutionJob:
        """Return the immutable submitted job authority."""
        return self._job

    def cancel(self) -> None:
        """Request cooperative cancellation through the injected hooks."""
        self._cancel_event.set()

    def _report_progress(self, report: GroundRegionalVariationProgress) -> None:
        """Validate and forward one immutable job-bound progress record."""
        if type(report) is not GroundRegionalVariationProgress:
            raise TypeError("progress must be exact")
        if report.total != self._job.execution_options.max_trials:
            raise ValueError("progress total must match the execution job")
        if report.completed < self._last_completed:
            raise ValueError("progress completed count must be monotonic")
        self._last_completed = report.completed
        self.progressed.emit(report)

    def _failure(
        self,
        stage: GroundRegionalVariationFailureStage,
        cause: Exception,
    ) -> GroundRegionalVariationFailed:
        """Create one typed terminal without attaching partial result state."""
        failure = GroundRegionalVariationFailed(
            stage,
            self._last_completed,
            self._job.execution_options.max_trials,
            cause,
        )
        failure.__cause__ = cause
        return failure

    def _terminal_matches_job(
        self,
        terminal: GroundRegionalVariationCancelled | GroundRegionalVariationFailed,
    ) -> bool:
        """Return whether terminal counts belong to this exact job."""
        return bool(terminal.total == self._job.execution_options.max_trials)

    def _publish_result(self, result: object) -> None:
        """Emit success only for one complete result bound to the exact job."""
        try:
            if type(result) is not RegionalGroundExecutionResult:
                raise TypeError("submitter must return an exact execution result")
            checked = cast(RegionalGroundExecutionResult, result)
            checked.assert_matches_job(self._job)
        except Exception as error:
            self.failed.emit(
                self._failure(GroundRegionalVariationFailureStage.VALIDATION, error)
            )
            return
        if self._cancel_event.is_set():
            self.cancelled.emit(
                GroundRegionalVariationCancelled(
                    self._last_completed, self._job.execution_options.max_trials
                )
            )
            return
        self.succeeded.emit(result)

    def run(self) -> None:  # pragma: no cover - exercised through Qt signals
        """Invoke only the injected authority and translate terminal signals."""
        hooks = GroundRegionalVariationHooks(
            progress_callback=self._report_progress,
            cancellation_requested=self._cancel_event.is_set,
        )
        try:
            result = self._submitter(self._job, hooks)
        except GroundRegionalVariationCancelled as terminal:
            if self._terminal_matches_job(terminal):
                self.cancelled.emit(terminal)
            else:
                self.failed.emit(
                    self._failure(
                        GroundRegionalVariationFailureStage.VALIDATION,
                        ValueError("cancellation total must match the execution job"),
                    )
                )
        except GroundRegionalVariationFailed as terminal:
            if self._terminal_matches_job(terminal):
                self.failed.emit(terminal)
            else:
                self.failed.emit(
                    self._failure(
                        GroundRegionalVariationFailureStage.VALIDATION,
                        ValueError("failure total must match the execution job"),
                    )
                )
        except Exception as error:
            self.failed.emit(
                self._failure(GroundRegionalVariationFailureStage.EXECUTOR, error)
            )
        else:
            self._publish_result(result)


RegionalGroundExecutionWorkerFactory = Callable[
    [RegionalGroundExecutionJob, RegionalGroundExecutionSubmitter],
    RegionalGroundExecutionWorker,
]


class RegionalGroundExecutionController(QObject):
    """Own one worker and expose typed signals to future UI adapters."""

    progressed = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    succeeded = pyqtSignal(object)  # noqa: N815
    cancelled = pyqtSignal(object)  # noqa: N815
    failed = pyqtSignal(object)  # noqa: N815
    finished = pyqtSignal()  # noqa: N815

    def __init__(
        self,
        submitter: RegionalGroundExecutionSubmitter,
        parent: QObject | None = None,
        *,
        worker_factory: RegionalGroundExecutionWorkerFactory = (
            RegionalGroundExecutionWorker
        ),
    ) -> None:
        super().__init__(parent)
        if not callable(submitter):
            raise TypeError("submitter must be callable")
        if not callable(worker_factory):
            raise TypeError("worker_factory must be callable")
        self._submitter = submitter
        self._worker_factory = worker_factory
        self._worker: RegionalGroundExecutionWorker | None = None

    @property
    def is_running(self) -> bool:
        """Return whether this controller owns an active submission."""
        return self._worker is not None

    @property
    def active_job_sha256(self) -> str | None:
        """Return the active job identity without exposing mutable state."""
        worker = self._worker
        return None if worker is None else worker.job.job_sha256

    def submit(self, job: RegionalGroundExecutionJob) -> None:
        """Start one injected submission and reject overlapping work."""
        if self._worker is not None:
            raise RuntimeError("regional-ground execution is already running")
        worker = self._worker_factory(job, self._submitter)
        if not isinstance(worker, RegionalGroundExecutionWorker):
            raise TypeError("worker_factory must return an execution worker")
        worker.progressed.connect(self._on_progressed)
        worker.succeeded.connect(self._on_succeeded)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        worker.start()

    def cancel(self) -> bool:
        """Request cancellation and return whether work was active."""
        worker = self._worker
        if worker is None:
            return False
        worker.cancel()
        return True

    def shutdown(self, timeout_ms: int = _SHUTDOWN_WAIT_MS) -> None:
        """Cancel and join the worker before controller teardown."""
        if type(timeout_ms) is not int or timeout_ms < 0:
            raise ValueError("timeout_ms must be a nonnegative integer")
        worker = self._worker
        if worker is None:
            return
        worker.cancel()
        if not worker.wait(timeout_ms):
            raise TimeoutError("regional-ground worker did not stop before timeout")
        self._worker = None

    def _is_active_sender(self) -> bool:
        """Reject stale queued signals from a prior submission."""
        return self.sender() is self._worker

    def _on_progressed(self, report: object) -> None:
        if self._is_active_sender():
            self.progressed.emit(report)

    def _on_succeeded(self, result: object) -> None:
        if self._is_active_sender():
            self.succeeded.emit(result)

    def _on_cancelled(self, terminal: object) -> None:
        if self._is_active_sender():
            self.cancelled.emit(terminal)

    def _on_failed(self, terminal: object) -> None:
        if self._is_active_sender():
            self.failed.emit(terminal)

    def _on_finished(self) -> None:
        if self._is_active_sender():
            self._worker = None
            self.finished.emit()


__all__ = [
    "RegionalGroundExecutionController",
    "RegionalGroundExecutionSubmitter",
    "RegionalGroundExecutionWorker",
    "RegionalGroundExecutionWorkerFactory",
]
