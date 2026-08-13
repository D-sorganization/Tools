"""Fail-closed adapter from Python flight output to ground request v1."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from shared.python.swing_sim.ball_setup import BallSetup
from shared.python.swing_sim.ground.contract_records import GroundSimulationRequest
from shared.python.swing_sim.ground.contract_types import (
    GroundCalibration,
    GroundContactState,
    GroundProvenance,
    GroundSurfaceProfile,
)
from shared.python.swing_sim.ground.unavailable_types import (
    GroundUnavailableFieldId,
    GroundUnavailableReason,
)

from .state import FlightStatePoint
from .surface_simulation import flight_point_to_ground_state
from .types import FlightResult, LaunchConditions, TrajectoryPoint

_INCOMING_SPEED_TOLERANCE_M_S = 1e-12


@dataclass(frozen=True)
class FlightGroundTransferSettings:
    """Ground-request identities, solver limits, and evidence records."""

    request_id: str
    surface: GroundSurfaceProfile
    calibration: GroundCalibration
    provenance: GroundProvenance
    max_time_s: float
    output_interval_s: float
    max_events: int
    rotational_inertia_factor: float = 0.4


class FlightGroundTransferError(ValueError):
    """Typed failure to construct a physically qualified ground request."""

    def __init__(
        self,
        message: str,
        field_id: GroundUnavailableFieldId,
        reason: GroundUnavailableReason,
    ) -> None:
        super().__init__(message)
        self.field_id = field_id
        self.reason = reason


def _failure(
    message: str,
    field_id: GroundUnavailableFieldId,
    reason: GroundUnavailableReason,
) -> FlightGroundTransferError:
    return FlightGroundTransferError(message, field_id, reason)


def launch_relative_surface(
    surface: GroundSurfaceProfile,
    ball_radius_m: float,
    ball_setup: BallSetup,
) -> GroundSurfaceProfile:
    """Translate terrain elevation into launch-ball-center coordinates."""
    if not isinstance(surface, GroundSurfaceProfile):
        raise ValueError("surface must be a GroundSurfaceProfile")
    if not isinstance(ball_setup, BallSetup):
        raise ValueError("ball_setup must be a BallSetup")
    if not np.isfinite(ball_radius_m) or ball_radius_m <= 0.0:
        raise ValueError("ball_radius_m must be finite and > 0")
    return replace(
        surface,
        height_m=surface.height_m - float(ball_radius_m) - ball_setup.tee_height_m,
    )


def _require_trajectory_qualification(points: tuple[TrajectoryPoint, ...]) -> None:
    if not points:
        raise _failure(
            "flight trajectory is empty",
            GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
            GroundUnavailableReason.NO_PHYSICAL_CONTACT,
        )
    initial = points[0]
    exact_origin = initial.time == 0.0 and np.array_equal(initial.position, np.zeros(3))
    if not exact_origin:
        raise _failure(
            "flight trajectory must begin at the exact launch origin "
            "in time and position",
            GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
            GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS,
        )
    previous_time = -1.0
    for point in points:
        if not math.isfinite(point.time) or point.time < 0.0:
            raise _failure(
                "flight sample times must be finite and nonnegative",
                GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
                GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS,
            )
        if point.time <= previous_time:
            raise _failure(
                "flight sample times must be strictly increasing",
                GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
                GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS,
            )
        previous_time = point.time


def _require_angular_states(
    points: tuple[TrajectoryPoint, ...],
) -> tuple[FlightStatePoint, ...]:
    if not all(isinstance(point, FlightStatePoint) for point in points):
        raise _failure(
            "flight trajectory does not propagate terminal angular velocity",
            GroundUnavailableFieldId.TERMINAL_ANGULAR_VELOCITY,
            GroundUnavailableReason.SOURCE_DOES_NOT_PROPAGATE,
        )
    return tuple(point for point in points if isinstance(point, FlightStatePoint))


def _contact_bracket(
    points: tuple[FlightStatePoint, ...],
    surface: GroundSurfaceProfile,
    radius_m: float,
) -> tuple[GroundContactState, GroundContactState]:
    separated: GroundContactState | None = None
    for point in points:
        state = flight_point_to_ground_state(point)
        if surface.signed_gap_m(state, radius_m) > 0.0:
            separated = state
            continue
        if separated is None:
            continue
        speeds = (
            surface.relative_normal_speed_m_s(separated),
            surface.relative_normal_speed_m_s(state),
        )
        if any(speed >= -_INCOMING_SPEED_TOLERANCE_M_S for speed in speeds):
            raise _failure(
                "physical contact bracket must be strictly incoming, not grazing",
                GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
                GroundUnavailableReason.SOURCE_OUT_OF_BOUNDS,
            )
        return separated, state
    raise _failure(
        "flight trajectory has no descending physical contact crossing",
        GroundUnavailableFieldId.PHYSICAL_CONTACT_BRACKET,
        GroundUnavailableReason.NO_PHYSICAL_CONTACT,
    )


def build_ground_simulation_request(
    result: FlightResult,
    launch: LaunchConditions,
    settings: FlightGroundTransferSettings,
) -> GroundSimulationRequest:
    """Build v1 request from launch-origin flight samples or fail closed."""
    if not isinstance(result, FlightResult) or not isinstance(launch, LaunchConditions):
        raise ValueError("result and launch must be flight contract records")
    if not isinstance(settings, FlightGroundTransferSettings):
        raise ValueError("settings must be FlightGroundTransferSettings")
    points = result.trajectory
    _require_trajectory_qualification(points)
    angular_points = _require_angular_states(points)
    surface = launch_relative_surface(
        settings.surface, launch.ball_radius, launch.ball_setup
    )
    separated, penetrating = _contact_bracket(
        angular_points, surface, launch.ball_radius
    )
    return GroundSimulationRequest(
        request_id=settings.request_id,
        surface=surface,
        last_separated_state=separated,
        first_penetrating_state=penetrating,
        ball_radius_m=launch.ball_radius,
        ball_mass_kg=launch.ball_mass,
        rotational_inertia_factor=settings.rotational_inertia_factor,
        max_time_s=settings.max_time_s,
        output_interval_s=settings.output_interval_s,
        max_events=settings.max_events,
        calibration=settings.calibration,
        provenance=settings.provenance,
    )


__all__ = [
    "FlightGroundTransferError",
    "FlightGroundTransferSettings",
    "build_ground_simulation_request",
    "launch_relative_surface",
]
