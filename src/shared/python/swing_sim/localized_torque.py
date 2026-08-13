"""Topological joint/time contracts for additive torque commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import cast

from shared.python.contracts import require

SHOULDER_JOINT_ID = "joint.shoulder"
WRIST_JOINT_ID = "joint.wrist"
DOUBLE_PENDULUM_JOINT_IDS = (SHOULDER_JOINT_ID, WRIST_JOINT_ID)


def _finite_real(value: object, message: str) -> float:
    """Normalize one strict non-boolean finite real scalar."""
    require(
        isinstance(value, Real) and not isinstance(value, bool),
        message,
        value,
    )
    normalized = float(cast(Real, value))
    require(math.isfinite(normalized), message, value)
    return normalized


@dataclass(frozen=True)
class LocalizedTorqueOffset:
    """One additive commanded torque over a half-open joint/time locus.

    Topological joint IDs are deliberately distinct from spatial trace IDs.
    The command is active on ``[start_s, end_s)`` so adjacent windows cannot
    double-apply at their shared boundary.
    """

    joint_id: str
    time_window_s: tuple[float, float]
    torque_nm: float

    def __post_init__(self) -> None:
        require(
            self.joint_id in DOUBLE_PENDULUM_JOINT_IDS,
            "localized torque joint_id must belong to the double-pendulum model",
            self.joint_id,
        )
        raw_window = cast(object, self.time_window_s)
        require(
            isinstance(raw_window, (tuple, list)) and len(raw_window) == 2,
            "localized torque time_window_s must contain two real values",
            raw_window,
        )
        window = cast(tuple[object, object] | list[object], raw_window)
        start_s = _finite_real(
            window[0], "localized torque time_window_s must contain two real values"
        )
        end_s = _finite_real(
            window[1], "localized torque time_window_s must contain two real values"
        )
        require(
            0.0 <= start_s < end_s,
            "localized torque time_window_s must satisfy 0 <= start < end",
            raw_window,
        )
        torque_nm = _finite_real(
            cast(object, self.torque_nm),
            "localized torque offset must be a finite real scalar",
        )
        object.__setattr__(self, "time_window_s", (start_s, end_s))
        object.__setattr__(self, "torque_nm", torque_nm)

    def is_active(self, time_s: float) -> bool:
        """Return whether ``time_s`` lies in the declared half-open window."""
        sample_time_s = _finite_real(
            cast(object, time_s),
            "localized torque sample time must be a finite real scalar",
        )
        start_s, end_s = self.time_window_s
        return start_s <= sample_time_s < end_s


def add_localized_offsets(
    base_torques_nm: tuple[float, float],
    offsets: tuple[LocalizedTorqueOffset, ...],
    time_s: float,
) -> tuple[float, float]:
    """Add all active offsets to one shoulder/wrist command pair."""
    torques = [float(base_torques_nm[0]), float(base_torques_nm[1])]
    for offset in offsets:
        if offset.is_active(time_s):
            torques[DOUBLE_PENDULUM_JOINT_IDS.index(offset.joint_id)] += (
                offset.torque_nm
            )
    return torques[0], torques[1]


def require_offsets_within_duration(
    offsets: tuple[LocalizedTorqueOffset, ...], duration_s: float
) -> None:
    """Require every half-open command window to lie inside one run."""
    for offset in offsets:
        require(
            offset.time_window_s[1] <= duration_s,
            "localized torque time window must lie within the run duration",
            (offset.time_window_s, duration_s),
        )


__all__ = [
    "DOUBLE_PENDULUM_JOINT_IDS",
    "SHOULDER_JOINT_ID",
    "WRIST_JOINT_ID",
    "LocalizedTorqueOffset",
    "add_localized_offsets",
    "require_offsets_within_duration",
]
