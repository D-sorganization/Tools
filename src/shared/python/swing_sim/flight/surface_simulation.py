"""Surface-aware native flight integration contracts and frame adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shared.python.swing_sim.ground import GroundContactState, GroundSurfaceProfile

from .frames import from_flight_frame
from .state import FlightStatePoint

GROUND_STATE_FRAME = "target_frame:x_downrange,y_up,z_right"


@dataclass(frozen=True)
class SurfaceFlightSimulationSettings:
    """Validated integration controls for an already launch-relative plane."""

    launch_relative_surface: GroundSurfaceProfile
    max_time_s: float = 10.0
    output_interval_s: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.launch_relative_surface, GroundSurfaceProfile):
            raise ValueError("launch_relative_surface must be a GroundSurfaceProfile")
        if not math.isfinite(self.max_time_s) or self.max_time_s <= 0.0:
            raise ValueError("max_time_s must be finite and > 0")
        if not math.isfinite(self.output_interval_s) or self.output_interval_s <= 0.0:
            raise ValueError("output_interval_s must be finite and > 0")


def flight_point_to_ground_state(point: FlightStatePoint) -> GroundContactState:
    """Rotate a full flight sample into the canonical target frame."""
    position = from_flight_frame(point.position)
    velocity = from_flight_frame(point.velocity)
    omega = from_flight_frame(point.angular_velocity_rad_s)
    return GroundContactState(
        time_s=point.time,
        frame=GROUND_STATE_FRAME,
        position_m=tuple(float(value) for value in position),
        velocity_m_s=tuple(float(value) for value in velocity),
        angular_velocity_rad_s=tuple(float(value) for value in omega),
    )


def flight_ode_signed_gap_m(
    surface: GroundSurfaceProfile,
    radius_m: float,
    state: np.ndarray,
) -> float:
    """Evaluate the ground contract's signed sphere gap for an ODE state."""
    values = np.asarray(state, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("flight ODE state must be a finite six-vector")
    point = FlightStatePoint(0.0, values[:3], values[3:], np.zeros(3))
    ground_state = flight_point_to_ground_state(point)
    return float(surface.signed_gap_m(ground_state, radius_m))


__all__ = ["SurfaceFlightSimulationSettings"]
