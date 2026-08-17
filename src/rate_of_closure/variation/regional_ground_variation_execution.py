"""Fail-closed execution loop for complete application-owned batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, NoReturn, TypeVar

from shared.python.contracts import require

from .regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)

_Trial = TypeVar("_Trial")
_Outcome = TypeVar("_Outcome")
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class CompleteBatchExecution(Generic[_Trial, _Outcome, _Result]):
    """Private-state batch plan whose publisher sees only complete outcomes."""

    trials: tuple[_Trial, ...]
    executor: Callable[[_Trial], object]
    validator: Callable[[_Trial, object], _Outcome]
    publisher: Callable[[tuple[_Outcome, ...]], _Result]

    def __post_init__(self) -> None:
        require(bool(self.trials), "trials must be nonempty")
        require(callable(self.executor), "executor must be callable")
        require(callable(self.validator), "validator must be callable")
        require(callable(self.publisher), "publisher must be callable")


class _ExecutionMonitor:
    """Isolate callbacks from trial outcomes and unpublished datasets."""

    def __init__(self, hooks: GroundRegionalVariationHooks, total: int) -> None:
        self._hooks = hooks
        self._total = total

    def raise_if_cancelled(self, completed: int) -> None:
        """Poll cancellation and convert callback defects to typed failure."""
        callback = self._hooks.cancellation_requested
        if callback is None:
            return
        try:
            requested = callback()
            if type(requested) is not bool:
                raise TypeError("cancellation callback must return an exact bool")
        except Exception as error:
            self.fail(
                GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK,
                completed,
                error,
            )
        if requested:
            raise GroundRegionalVariationCancelled(completed, self._total)

    def report(self, completed: int) -> None:
        """Send an immutable progress value or terminate without publication."""
        callback = self._hooks.progress_callback
        if callback is None:
            return
        try:
            callback(GroundRegionalVariationProgress(completed, self._total))
        except Exception as error:
            self.fail(
                GroundRegionalVariationFailureStage.PROGRESS_CALLBACK,
                completed,
                error,
            )

    def fail(
        self,
        stage: GroundRegionalVariationFailureStage,
        completed: int,
        cause: Exception,
    ) -> NoReturn:
        """Raise one typed terminal failure without exposing partial state."""
        raise GroundRegionalVariationFailed(
            stage, completed, self._total, cause
        ) from cause


def _execute_trial(
    job: CompleteBatchExecution[_Trial, _Outcome, _Result],
    trial: _Trial,
    monitor: _ExecutionMonitor,
    completed: int,
) -> _Outcome:
    """Execute and validate one trial, then honor in-flight cancellation."""
    try:
        raw_outcome = job.executor(trial)
    except Exception as error:
        monitor.fail(GroundRegionalVariationFailureStage.EXECUTOR, completed, error)
    try:
        outcome = job.validator(trial, raw_outcome)
    except Exception as error:
        monitor.fail(GroundRegionalVariationFailureStage.VALIDATION, completed, error)
    monitor.raise_if_cancelled(completed)
    return outcome


def _publish(
    job: CompleteBatchExecution[_Trial, _Outcome, _Result],
    outcomes: tuple[_Outcome, ...],
    monitor: _ExecutionMonitor,
) -> _Result:
    """Invoke the publisher only after every trial has been accepted."""
    try:
        return job.publisher(outcomes)
    except Exception as error:
        monitor.fail(
            GroundRegionalVariationFailureStage.PUBLICATION, len(outcomes), error
        )


def execute_complete_batch(
    job: CompleteBatchExecution[_Trial, _Outcome, _Result],
    hooks: GroundRegionalVariationHooks | None = None,
) -> _Result:
    """Execute all trials and publish once, or raise one typed terminal error."""
    require(
        type(job) is CompleteBatchExecution,
        "job must be an exact CompleteBatchExecution",
    )
    require(
        hooks is None or type(hooks) is GroundRegionalVariationHooks,
        "hooks must be an exact GroundRegionalVariationHooks",
    )
    monitor = _ExecutionMonitor(
        GroundRegionalVariationHooks() if hooks is None else hooks, len(job.trials)
    )
    outcomes: list[_Outcome] = []
    for trial in job.trials:
        completed = len(outcomes)
        monitor.raise_if_cancelled(completed)
        outcomes.append(_execute_trial(job, trial, monitor, completed))
        monitor.report(len(outcomes))
        monitor.raise_if_cancelled(len(outcomes))
    return _publish(job, tuple(outcomes), monitor)


__all__ = ["CompleteBatchExecution", "execute_complete_batch"]
