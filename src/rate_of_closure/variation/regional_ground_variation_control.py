"""UI-neutral execution controls for regional-ground variation batches."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from shared.python.contracts import require


@dataclass(frozen=True)
class GroundRegionalVariationProgress:
    """Immutable count reported only after one trial is accepted.

    Preconditions:
        ``total`` is positive and ``completed`` lies in ``[0, total]``.
    """

    completed: int
    total: int

    def __post_init__(self) -> None:
        require(type(self.total) is int and self.total > 0, "total must be positive")
        require(
            type(self.completed) is int and 0 <= self.completed <= self.total,
            "completed must be in [0, total]",
        )


GroundRegionalVariationProgressCallback = Callable[
    [GroundRegionalVariationProgress], None
]
GroundRegionalVariationCancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class GroundRegionalVariationHooks:
    """Optional application callbacks that cannot access mutable run state."""

    progress_callback: GroundRegionalVariationProgressCallback | None = None
    cancellation_requested: GroundRegionalVariationCancellationCheck | None = None

    def __post_init__(self) -> None:
        require(
            self.progress_callback is None or callable(self.progress_callback),
            "progress_callback must be callable",
        )
        require(
            self.cancellation_requested is None
            or callable(self.cancellation_requested),
            "cancellation_requested must be callable",
        )


class GroundRegionalVariationFailureStage(StrEnum):
    """Stable terminal stage for a failed batch with no published dataset."""

    CANCELLATION_CALLBACK = "cancellation_callback"
    PREFLIGHT = "preflight"
    EXECUTOR = "executor"
    VALIDATION = "validation"
    PROGRESS_CALLBACK = "progress_callback"
    PUBLICATION = "publication"


class GroundRegionalVariationTerminalError(RuntimeError):
    """Base terminal signal carrying counts but never partial result rows."""

    def __init__(self, message: str, completed: int, total: int) -> None:
        progress = GroundRegionalVariationProgress(completed, total)
        self.completed = progress.completed
        self.total = progress.total
        super().__init__(message)


class GroundRegionalVariationCancelled(GroundRegionalVariationTerminalError):
    """Cooperative terminal cancellation with accepted-trial count."""

    def __init__(self, completed: int, total: int) -> None:
        super().__init__(
            f"regional-ground variation cancelled after {completed} of {total} trials",
            completed,
            total,
        )


class GroundRegionalVariationFailed(GroundRegionalVariationTerminalError):
    """Typed terminal failure that intentionally carries no partial dataset."""

    def __init__(
        self,
        stage: GroundRegionalVariationFailureStage,
        completed: int,
        total: int,
        cause: Exception,
    ) -> None:
        require(
            type(stage) is GroundRegionalVariationFailureStage,
            "stage must be an exact GroundRegionalVariationFailureStage",
        )
        require(isinstance(cause, Exception), "cause must be an Exception")
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)
        super().__init__(
            f"regional-ground variation failed in {stage.value}: "
            f"{self.cause_type}: {self.cause_message}",
            completed,
            total,
        )


__all__ = [
    "GroundRegionalVariationCancelled",
    "GroundRegionalVariationCancellationCheck",
    "GroundRegionalVariationFailed",
    "GroundRegionalVariationFailureStage",
    "GroundRegionalVariationHooks",
    "GroundRegionalVariationProgress",
    "GroundRegionalVariationProgressCallback",
    "GroundRegionalVariationTerminalError",
]
