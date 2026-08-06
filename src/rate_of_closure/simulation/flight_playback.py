"""Deterministic interpolation contract for ball-flight playback (#4200).

The physics engine owns the sampled timestamps. Playback interpolates those
samples in physical SI time without altering or re-integrating the trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlaybackFrame:
    """One interpolated app-frame ball position at a physical time."""

    time_s: float
    position_m: np.ndarray = field(repr=False)
    lower_index: int
    fraction: float
    is_impact: bool


@dataclass(frozen=True)
class TimedTrajectory:
    """Validated immutable trajectory used by every playback presentation.

    Preconditions:
        ``times_s`` is finite, one-dimensional, non-negative, and strictly
        increasing. ``positions_m`` is finite with shape ``(N, 3)``.
    """

    times_s: np.ndarray = field(repr=False)
    positions_m: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Copy and validate samples so later caller mutation is impossible."""
        times = np.array(self.times_s, dtype=float, copy=True)
        positions = np.array(self.positions_m, dtype=float, copy=True)
        if times.ndim != 1:
            raise ValueError("times_s must be a one-dimensional array")
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("positions_m must have shape (N, 3)")
        if len(times) != len(positions):
            raise ValueError("times_s and positions_m must have the same sample count")
        if not len(times):
            raise ValueError("trajectory must contain at least one sample")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("trajectory times and positions must be finite")
        if times[0] < 0.0:
            raise ValueError("trajectory times must be non-negative")
        if len(times) > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("trajectory times must be strictly increasing")
        times.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "positions_m", positions)

    @property
    def duration_s(self) -> float:
        """Physical timestamp of impact/landing [s]."""
        return float(self.times_s[-1])

    def frame_at(self, requested_time_s: float) -> PlaybackFrame:
        """Linearly interpolate position at a finite, endpoint-clamped time."""
        if not math.isfinite(requested_time_s):
            raise ValueError("requested_time_s must be finite")
        time_s = min(max(float(requested_time_s), 0.0), self.duration_s)
        if time_s <= self.times_s[0] or len(self.times_s) == 1:
            return self._frame(0, 0.0, time_s)
        if time_s >= self.duration_s:
            return self._frame(len(self.times_s) - 1, 0.0, time_s)
        upper_index = int(np.searchsorted(self.times_s, time_s, side="right"))
        lower_index = upper_index - 1
        span = float(self.times_s[upper_index] - self.times_s[lower_index])
        fraction = (time_s - float(self.times_s[lower_index])) / span
        position = (
            self.positions_m[lower_index] * (1.0 - fraction)
            + self.positions_m[upper_index] * fraction
        )
        return PlaybackFrame(time_s, position, lower_index, fraction, False)

    def _frame(self, index: int, fraction: float, time_s: float) -> PlaybackFrame:
        """Create an endpoint frame with a defensive position copy."""
        return PlaybackFrame(
            time_s=time_s,
            position_m=self.positions_m[index].copy(),
            lower_index=index,
            fraction=fraction,
            is_impact=index == len(self.times_s) - 1,
        )


__all__ = ["PlaybackFrame", "TimedTrajectory"]
