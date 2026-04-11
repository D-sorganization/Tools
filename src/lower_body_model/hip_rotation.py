"""Inclined-plane hip rotation target profiles for the lower-body model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HipRotationSample:
    """A deterministic point on the requested hip rotation target."""

    time_sec: float
    rotation_deg: float
    plane_point: np.ndarray


@dataclass(frozen=True)
class InclinedPlaneHipRotationTarget:
    """Two-phase golf hip rotation target viewed from above."""

    duration_sec: float
    backswing_degrees: float = 45.0
    counterclockwise_degrees: float = 90.0
    incline_degrees: float = 12.0
    sample_count: int = 181

    def __post_init__(self) -> None:
        assert self.duration_sec > 0.0, "DbC PRE: duration_sec must be positive"
        assert 0.0 < self.backswing_degrees <= 90.0, (
            "DbC PRE: backswing_degrees must be in (0, 90]"
        )
        assert 0.0 < self.counterclockwise_degrees <= 180.0, (
            "DbC PRE: counterclockwise_degrees must be in (0, 180]"
        )
        assert -60.0 <= self.incline_degrees <= 60.0, (
            "DbC PRE: incline_degrees must be in [-60, 60]"
        )
        assert self.sample_count >= 2, "DbC PRE: sample_count must be at least 2"

    @property
    def reversal_time_sec(self) -> float:
        """Time where the clockwise phase reverses."""
        return self.duration_sec / 2.0

    @property
    def finish_rotation_deg(self) -> float:
        """Final rotation angle in degrees."""
        return -self.backswing_degrees + self.counterclockwise_degrees

    def rotation_degrees_at(self, time_sec: float) -> float:
        """Return the target rotation in degrees at ``time_sec``."""
        assert time_sec >= 0.0, "DbC PRE: time_sec must be non-negative"
        clamped_time = min(time_sec, self.duration_sec)

        if clamped_time <= self.reversal_time_sec:
            phase = clamped_time / self.reversal_time_sec
            return -self.backswing_degrees * phase

        phase = (clamped_time - self.reversal_time_sec) / self.reversal_time_sec
        return -self.backswing_degrees + self.counterclockwise_degrees * phase

    def plane_point_at(self, time_sec: float) -> np.ndarray:
        """Return a unit-radius point on the inclined target plane."""
        rotation_rad = np.radians(self.rotation_degrees_at(time_sec))
        incline_rad = np.radians(self.incline_degrees)
        return np.array(
            [
                np.cos(rotation_rad),
                np.sin(rotation_rad) * np.cos(incline_rad),
                np.sin(rotation_rad) * np.sin(incline_rad),
            ],
            dtype=float,
        )

    def sample(self) -> list[HipRotationSample]:
        """Sample the full target trajectory deterministically."""
        times = np.linspace(0.0, self.duration_sec, self.sample_count)
        return [
            HipRotationSample(
                time_sec=float(time_sec),
                rotation_deg=self.rotation_degrees_at(float(time_sec)),
                plane_point=self.plane_point_at(float(time_sec)),
            )
            for time_sec in times
        ]
