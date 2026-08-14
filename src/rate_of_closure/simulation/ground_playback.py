"""Phase-aware playback adapter for strict ground simulation results."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import cast

from shared.python.swing_sim.ground import (
    GroundPhase,
    GroundResultStatus,
    GroundSimulationResult,
    result_from_json,
)

DEFAULT_IMPORT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_IMPORT_MAX_POINTS = 100_000


@dataclass(frozen=True)
class GroundPlaybackFrame:
    """One immutable display frame on the absolute ground timeline."""

    time_s: float
    elapsed_s: float
    position_m: tuple[float, float, float]
    phase: str
    lower_index: int
    interpolation_fraction: float
    is_terminal: bool


def load_ground_result_json(
    text: str,
    *,
    max_bytes: int = DEFAULT_IMPORT_MAX_BYTES,
    max_points: int = DEFAULT_IMPORT_MAX_POINTS,
) -> GroundSimulationResult:
    """Parse one bounded, exact ``flight-to-ground-result/v1`` document.

    Args:
        text: Complete JSON document containing only the result record.
        max_bytes: Maximum UTF-8 encoded document size.
        max_points: Maximum accepted trajectory sample count.

    Raises:
        TypeError: If an argument has the wrong exact type.
        ValueError: If a bound or the strict result contract is violated.
    """
    if type(text) is not str:
        raise TypeError("ground result JSON must be text")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_points) is not int or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("ground result JSON exceeds the import size limit")
    result = result_from_json(text)
    if len(result.trajectory) > max_points:
        raise ValueError("ground result trajectory exceeds the import point limit")
    return result


class GroundPlaybackTimeline:
    """Read-only ground trajectory with phase-safe interpolation semantics."""

    def __init__(self, result: GroundSimulationResult) -> None:
        if type(result) is not GroundSimulationResult:
            raise TypeError("result must use the exact GroundSimulationResult type")
        if result.status not in {
            GroundResultStatus.COMPLETE,
            GroundResultStatus.PARTIAL,
        }:
            raise ValueError("playback requires a complete or partial ground result")
        if not result.trajectory or result.summary is None:
            raise ValueError("playback requires trajectory and summary output")
        self._result = result
        self._points = result.trajectory
        self._times: tuple[float, ...] = tuple(
            float(point.time_s) for point in self._points
        )

    @property
    def result(self) -> GroundSimulationResult:
        """Return the immutable strict result backing this timeline."""
        return self._result

    @property
    def start_time_s(self) -> float:
        """Return the absolute first-contact time."""
        return self._times[0]

    @property
    def end_time_s(self) -> float:
        """Return the absolute observed termination time."""
        return self._times[-1]

    @property
    def duration_s(self) -> float:
        """Return elapsed time between first contact and observed end."""
        return self.end_time_s - self.start_time_s

    @property
    def is_complete(self) -> bool:
        """Return whether the result reached a qualified terminal condition."""
        return self._result.status is GroundResultStatus.COMPLETE

    @property
    def end_label(self) -> str:
        """Return honest terminal marker language for the result status."""
        if not self.is_complete:
            return "Observed end"
        if self._result.termination.reason.value == "rest":
            return "Rest"
        return "End / left surface"

    @property
    def carry_position_m(self) -> tuple[float, float, float]:
        """Return the first-contact marker position."""
        return cast(tuple[float, float, float], self._points[0].position_m)

    @property
    def endpoint_position_m(self) -> tuple[float, float, float]:
        """Return the final observed marker position."""
        return cast(tuple[float, float, float], self._points[-1].position_m)

    def phase_time(self, phase: str) -> float | None:
        """Return the first exact sample time for ``phase``, when present."""
        try:
            requested = GroundPhase(phase)
        except ValueError as exc:
            raise ValueError(f"unknown ground phase: {phase}") from exc
        return next(
            (point.time_s for point in self._points if point.phase is requested),
            None,
        )

    def step_time(self, current_time_s: float, direction: int) -> float:
        """Return the adjacent exact sample time in ``direction``."""
        self._validate_time(current_time_s)
        if direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        if direction > 0:
            index = bisect.bisect_right(self._times, current_time_s + 1e-12)
            return self._times[min(index, len(self._times) - 1)]
        index = bisect.bisect_left(self._times, current_time_s - 1e-12) - 1
        return self._times[max(index, 0)]

    def frame_at(self, time_s: float) -> GroundPlaybackFrame:
        """Return a clamped frame, holding position across phase transitions."""
        self._validate_time(time_s)
        clamped = min(max(time_s, self.start_time_s), self.end_time_s)
        lower_index = max(0, bisect.bisect_right(self._times, clamped) - 1)
        if lower_index >= len(self._points) - 1:
            return self._frame(lower_index, clamped, self._points[-1].position_m, 0.0)
        lower = self._points[lower_index]
        upper = self._points[lower_index + 1]
        if lower.phase is not upper.phase:
            return self._frame(lower_index, clamped, lower.position_m, 0.0)
        fraction = (clamped - lower.time_s) / (upper.time_s - lower.time_s)
        position = tuple(
            lower.position_m[index]
            + fraction * (upper.position_m[index] - lower.position_m[index])
            for index in range(3)
        )
        return self._frame(lower_index, clamped, position, fraction)

    @staticmethod
    def _validate_time(value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("playback time must be a number")
        if not math.isfinite(float(value)):
            raise ValueError("playback time must be finite")

    def _frame(
        self,
        lower_index: int,
        time_s: float,
        position_m: tuple[float, float, float],
        fraction: float,
    ) -> GroundPlaybackFrame:
        point = self._points[lower_index]
        return GroundPlaybackFrame(
            time_s=time_s,
            elapsed_s=time_s - self.start_time_s,
            position_m=position_m,
            phase=point.phase.value,
            lower_index=lower_index,
            interpolation_fraction=fraction,
            is_terminal=time_s >= self.end_time_s,
        )


__all__ = [
    "DEFAULT_IMPORT_MAX_BYTES",
    "DEFAULT_IMPORT_MAX_POINTS",
    "GroundPlaybackFrame",
    "GroundPlaybackTimeline",
    "load_ground_result_json",
]
