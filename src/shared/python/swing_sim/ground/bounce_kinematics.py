"""Exact contact interpolation and ballistic hop kinematics."""

from __future__ import annotations

import math
from dataclasses import replace

from ._vector_math import add, dot, interpolate, scale
from .bounce_types import BounceModelSettings
from .contract_records import GroundSimulationRequest
from .contract_types import GroundContactState, GroundPhase, GroundTrajectoryPoint


def _project_to_contact(
    state: GroundContactState,
    request: GroundSimulationRequest,
) -> GroundContactState:
    gap = request.surface.signed_gap_m(state, request.ball_radius_m)
    position = add(state.position_m, scale(request.surface.normal_unit, -gap))
    return replace(state, position_m=position)


def interpolate_first_contact(request: GroundSimulationRequest) -> GroundContactState:
    """Interpolate the physical sphere-plane contact from a valid bracket."""
    if type(request) is not GroundSimulationRequest:
        raise ValueError("contact interpolation requires an exact ground request")
    left = request.last_separated_state
    right = request.first_penetrating_state
    left_gap = request.surface.signed_gap_m(left, request.ball_radius_m)
    right_gap = request.surface.signed_gap_m(right, request.ball_radius_m)
    fraction = left_gap / (left_gap - right_gap)
    state = GroundContactState(
        time_s=left.time_s + fraction * (right.time_s - left.time_s),
        frame=left.frame,
        position_m=interpolate(left.position_m, right.position_m, fraction),
        velocity_m_s=interpolate(left.velocity_m_s, right.velocity_m_s, fraction),
        angular_velocity_rad_s=interpolate(
            left.angular_velocity_rad_s,
            right.angular_velocity_rad_s,
            fraction,
        ),
    )
    return _project_to_contact(state, request)


def ballistic_state(
    initial: GroundContactState,
    elapsed_s: float,
    settings: BounceModelSettings,
) -> GroundContactState:
    """Propagate a separated sphere under constant gravity and constant spin."""
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("ballistic elapsed_s must be finite and nonnegative")
    gravity = settings.gravity_m_s2
    displacement = add(
        scale(initial.velocity_m_s, elapsed_s),
        scale(gravity, 0.5 * elapsed_s * elapsed_s),
    )
    return replace(
        initial,
        time_s=initial.time_s + elapsed_s,
        position_m=add(initial.position_m, displacement),
        velocity_m_s=add(initial.velocity_m_s, scale(gravity, elapsed_s)),
    )


def next_contact_elapsed_s(
    outgoing: GroundContactState,
    request: GroundSimulationRequest,
    settings: BounceModelSettings,
) -> float | None:
    """Return the exact positive ballistic contact root, or ``None``."""
    normal = request.surface.normal_unit
    outgoing_speed = request.surface.relative_normal_speed_m_s(outgoing)
    gravity_normal = dot(settings.gravity_m_s2, normal)
    if outgoing_speed <= settings.velocity_tolerance_m_s:
        return None
    if gravity_normal >= -settings.velocity_tolerance_m_s:
        return None
    elapsed = -2.0 * outgoing_speed / gravity_normal
    return elapsed if elapsed > settings.time_tolerance_s else None


def contact_state_after_hop(
    outgoing: GroundContactState,
    request: GroundSimulationRequest,
    settings: BounceModelSettings,
) -> GroundContactState | None:
    """Return the exact next contact state after a ballistic hop."""
    elapsed = next_contact_elapsed_s(outgoing, request, settings)
    if elapsed is None:
        return None
    return _project_to_contact(ballistic_state(outgoing, elapsed, settings), request)


def trajectory_point(
    state: GroundContactState, phase: GroundPhase
) -> GroundTrajectoryPoint:
    """Convert a contact or airborne state to an existing trajectory record."""
    return GroundTrajectoryPoint(
        time_s=state.time_s,
        frame=state.frame,
        position_m=state.position_m,
        velocity_m_s=state.velocity_m_s,
        angular_velocity_rad_s=state.angular_velocity_rad_s,
        phase=phase,
    )


__all__ = [
    "ballistic_state",
    "contact_state_after_hop",
    "interpolate_first_contact",
    "next_contact_elapsed_s",
    "trajectory_point",
]
