"""Shared result-state validation helpers for flight-to-ground v1."""

from __future__ import annotations

import math

from .contract_types import (
    GroundEvent,
    GroundEventType,
    GroundPhase,
    GroundResultStatus,
    GroundTerminationReason,
    GroundTrajectoryPoint,
)
from .result_types import GroundTermination

_STATUS_TERMINATIONS = {
    GroundResultStatus.COMPLETE: frozenset(
        {GroundTerminationReason.REST, GroundTerminationReason.LEFT_SURFACE}
    ),
    GroundResultStatus.PARTIAL: frozenset(
        {GroundTerminationReason.TIME_LIMIT, GroundTerminationReason.EVENT_LIMIT}
    ),
    GroundResultStatus.FAILED: frozenset({GroundTerminationReason.NUMERICAL_FAILURE}),
    GroundResultStatus.UNAVAILABLE: frozenset(
        {GroundTerminationReason.UNAVAILABLE_INPUT}
    ),
}


def close(left: float, right: float) -> bool:
    """Return whether two canonical contract numbers agree within tolerance."""
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def vector_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    """Return whether equal-length vectors agree componentwise."""
    return all(close(a, b) for a, b in zip(left, right, strict=True))


def validate_status_termination(
    status: GroundResultStatus,
    termination: GroundTermination,
) -> None:
    """Require one valid status and termination-reason pairing."""
    validate_status_reason(status, termination.reason)


def validate_status_reason(
    status: GroundResultStatus,
    reason: GroundTerminationReason,
) -> None:
    """Require a canonical result-status and termination-reason pairing."""
    if reason not in _STATUS_TERMINATIONS[status]:
        raise ValueError(
            f"result status {status} is incompatible with {reason} termination"
        )


def validate_terminal_state(
    status: GroundResultStatus,
    points: tuple[GroundTrajectoryPoint, ...],
    events: tuple[GroundEvent, ...],
    termination: GroundTermination,
) -> None:
    """Bind terminal ledger data to the declared final state."""
    final_point = points[-1]
    final_event = events[-1]
    terminal_events = {GroundEventType.REST, GroundEventType.LEFT_SURFACE}
    if status is GroundResultStatus.PARTIAL:
        if (
            final_point.phase is GroundPhase.REST
            or final_event.event_type in terminal_events
        ):
            raise ValueError("partial result cannot contain a terminal state")
        return
    expected_event = {
        GroundTerminationReason.REST: GroundEventType.REST,
        GroundTerminationReason.LEFT_SURFACE: GroundEventType.LEFT_SURFACE,
    }[termination.reason]
    if final_event.event_type is not expected_event:
        raise ValueError("completed termination must match the terminal event")
    if not close(final_event.time_s, termination.time_s):
        raise ValueError("terminal event time must match termination time")
    if not vector_close(final_event.position_m, final_point.position_m):
        raise ValueError(
            "terminal event position must match the final trajectory point"
        )
    if not vector_close(final_event.velocity_after_m_s, final_point.velocity_m_s):
        raise ValueError(
            "terminal event output velocity must match the final trajectory point"
        )
    if not vector_close(
        final_event.angular_velocity_after_rad_s,
        final_point.angular_velocity_rad_s,
    ):
        raise ValueError(
            "terminal event output spin must match the final trajectory point"
        )
    if termination.reason is GroundTerminationReason.REST:
        if final_point.phase is not GroundPhase.REST:
            raise ValueError("rest termination requires a final rest-phase point")
    elif final_point.phase is GroundPhase.REST:
        raise ValueError("left-surface termination cannot end in the rest phase")


__all__ = [
    "close",
    "validate_status_reason",
    "validate_status_termination",
    "validate_terminal_state",
    "vector_close",
]
