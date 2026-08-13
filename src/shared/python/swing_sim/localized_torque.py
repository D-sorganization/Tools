"""Topological joint/time contracts for additive torque commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from shared.python.contracts import require

from ._numeric_contracts import finite_real

SHOULDER_JOINT_ID = "joint.shoulder"
WRIST_JOINT_ID = "joint.wrist"
DOUBLE_PENDULUM_JOINT_IDS = (SHOULDER_JOINT_ID, WRIST_JOINT_ID)


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
        start_s = finite_real(window[0], "localized torque time_window_s start")
        end_s = finite_real(window[1], "localized torque time_window_s end")
        require(
            0.0 <= start_s < end_s,
            "localized torque time_window_s must satisfy 0 <= start < end",
            raw_window,
        )
        torque_nm = finite_real(
            cast(object, self.torque_nm),
            "localized torque offset",
        )
        object.__setattr__(self, "time_window_s", (start_s, end_s))
        object.__setattr__(self, "torque_nm", torque_nm)

    def is_active(self, time_s: float) -> bool:
        """Return whether ``time_s`` lies in the declared half-open window."""
        sample_time_s = finite_real(
            cast(object, time_s),
            "localized torque sample time",
        )
        start_s, end_s = self.time_window_s
        return bool(start_s <= sample_time_s < end_s)


def add_localized_offsets(
    base_torques_nm: tuple[float, float],
    offsets: tuple[LocalizedTorqueOffset, ...],
    time_s: float,
) -> tuple[float, float]:
    """Add all active offsets to one shoulder/wrist command pair."""
    raw_base = cast(object, base_torques_nm)
    require(
        isinstance(raw_base, (tuple, list)) and len(raw_base) == 2,
        "base_torques_nm must contain two finite real values",
        raw_base,
    )
    base = cast(tuple[object, object] | list[object], raw_base)
    torques = [
        finite_real(base[0], "base_torques_nm shoulder value"),
        finite_real(base[1], "base_torques_nm wrist value"),
    ]
    commands = _validated_offsets(offsets)
    sample_time_s = finite_real(cast(object, time_s), "localized torque sample time")
    for offset in commands:
        if offset.is_active(sample_time_s):
            torques[DOUBLE_PENDULUM_JOINT_IDS.index(offset.joint_id)] += (
                offset.torque_nm
            )
    return torques[0], torques[1]


def require_offsets_within_duration(
    offsets: tuple[LocalizedTorqueOffset, ...], duration_s: float
) -> None:
    """Require every half-open command window to lie inside one run."""
    commands = _validated_offsets(offsets)
    duration = finite_real(cast(object, duration_s), "run duration")
    require(duration > 0.0, "run duration must be > 0", duration_s)
    for offset in commands:
        require(
            offset.time_window_s[1] <= duration,
            "localized torque time window must lie within the run duration",
            (offset.time_window_s, duration),
        )


def _validated_offsets(offsets: object) -> tuple[LocalizedTorqueOffset, ...]:
    """Return one strictly typed command collection."""
    require(
        isinstance(offsets, (tuple, list)),
        "offsets must be a tuple or list of LocalizedTorqueOffset values",
        offsets,
    )
    raw_commands = cast(tuple[object, ...] | list[object], offsets)
    commands = tuple(raw_commands)
    require(
        all(isinstance(offset, LocalizedTorqueOffset) for offset in commands),
        "offsets must contain only LocalizedTorqueOffset values",
        commands,
    )
    return cast(tuple[LocalizedTorqueOffset, ...], commands)


__all__ = [
    "DOUBLE_PENDULUM_JOINT_IDS",
    "SHOULDER_JOINT_ID",
    "WRIST_JOINT_ID",
    "LocalizedTorqueOffset",
    "add_localized_offsets",
    "require_offsets_within_duration",
]
