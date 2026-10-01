"""Ball-flight value types (DbC-validated dataclasses).

Ported from UpstreamDrift ``src/shared/python/physics/flight_models.py``
(``UnifiedLaunchConditions``, ``TrajectoryPoint``, ``FlightResult``) for
epic #4103 / flight port #4107, rewritten self-contained.

Frame convention (UpstreamDrift flight frame): x forward, y left, z up.
Use :mod:`shared.python.swing_sim.flight.frames` to convert to the app
frame (x target, y up, z right).

Units: SI throughout; angular fields are radians; ``spin_rate`` is RPM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from shared.python.swing_sim.ball_setup import BallSetup

from ._constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
    MIN_SPEED_THRESHOLD_M_S,
    MPH_TO_MPS,
    RPM_TO_RAD_S,
)
from .wind import WindScenario

DEFAULT_BACKSPIN_AXIS = (0.0, -1.0, 0.0)
"""Pure-backspin unit axis in the flight frame (x fwd / y left / z up)."""


@dataclass(frozen=True)
class LaunchConditions:
    """Launch conditions with explicit units.

    ``ball_speed`` is m/s; angular fields are radians; ``spin_rate`` is RPM;
    ``wind_speed`` is m/s; mass, radius, density, and gravity are SI units.

    ``spin_axis`` (optional) is a unit 3-vector in the flight frame. When
    provided it overrides the legacy ``spin_axis_angle`` decomposition in
    :meth:`get_spin_vector`, so post-impact spin vectors derived by
    :func:`shared.python.swing_sim.flight.launch.derive_launch_conditions`
    round-trip exactly.
    """

    ball_speed: float
    launch_angle: float
    azimuth_angle: float = 0.0
    spin_rate: float = 2500.0
    spin_axis_angle: float = 0.0
    spin_axis: tuple[float, float, float] | None = None
    ball_mass: float = GOLF_BALL_MASS_KG
    ball_radius: float = GOLF_BALL_RADIUS_M
    air_density: float = AIR_DENSITY_SEA_LEVEL_KG_M3
    gravity: float = GRAVITY_M_S2
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    wind_scenario: WindScenario | None = None
    ball_setup: BallSetup = field(default_factory=BallSetup)

    def __post_init__(self) -> None:
        """Validate finiteness, signs, and angle ranges (DbC preconditions)."""
        scalars = {
            "ball_speed": self.ball_speed,
            "launch_angle": self.launch_angle,
            "azimuth_angle": self.azimuth_angle,
            "spin_rate": self.spin_rate,
            "spin_axis_angle": self.spin_axis_angle,
            "ball_mass": self.ball_mass,
            "ball_radius": self.ball_radius,
            "air_density": self.air_density,
            "gravity": self.gravity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
        }
        for name, value in scalars.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value!r}")
        if self.ball_speed < 0.0:
            raise ValueError(f"ball_speed must be >= 0; got {self.ball_speed!r}")
        if self.spin_rate < 0.0:
            raise ValueError(f"spin_rate must be RPM and >= 0; got {self.spin_rate!r}")
        for name in ("ball_mass", "ball_radius", "air_density", "gravity"):
            if scalars[name] <= 0.0:
                raise ValueError(f"{name} must be > 0; got {scalars[name]!r}")
        if not abs(self.launch_angle) <= math.pi / 2.0:
            raise ValueError(
                "launch_angle is radians and must be within [-pi/2, pi/2] — "
                "did you pass degrees? Use from_imperial()."
            )
        if self.spin_axis is not None:
            axis = np.asarray(self.spin_axis, dtype=float)
            if axis.shape != (3,) or not np.all(np.isfinite(axis)):
                raise ValueError(f"spin_axis must be a finite 3-vector; got {axis!r}")
            norm = float(np.linalg.norm(axis))
            if abs(norm - 1.0) > 1e-6:
                raise ValueError(f"spin_axis must be a unit vector; |axis|={norm!r}")
            object.__setattr__(
                self, "spin_axis", (float(axis[0]), float(axis[1]), float(axis[2]))
            )
        if self.wind_scenario is not None and self.wind_speed != 0.0:
            raise ValueError(
                "provide either wind_scenario or legacy wind_speed, not both"
            )
        if not isinstance(self.ball_setup, BallSetup):
            raise ValueError("ball_setup must be a BallSetup")

    @classmethod
    def from_imperial(
        cls,
        ball_speed_mph: float,
        launch_angle_deg: float,
        spin_rate_rpm: float,
        azimuth_angle_deg: float = 0.0,
        spin_axis_angle_deg: float = 0.0,
        wind_speed_mph: float = 0.0,
        wind_direction_deg: float = 0.0,
    ) -> LaunchConditions:
        """Create launch conditions from imperial units."""
        if ball_speed_mph is None:
            raise ValueError("ball_speed_mph must be provided")
        return cls(
            ball_speed=ball_speed_mph * MPH_TO_MPS,
            launch_angle=math.radians(launch_angle_deg),
            azimuth_angle=math.radians(azimuth_angle_deg),
            spin_rate=spin_rate_rpm,
            spin_axis_angle=math.radians(spin_axis_angle_deg),
            wind_speed=wind_speed_mph * MPH_TO_MPS,
            wind_direction=math.radians(wind_direction_deg),
        )

    def get_initial_velocity(self) -> np.ndarray:
        """Compute the 3D initial velocity vector from launch angles and speed."""
        ca, sa = math.cos(self.azimuth_angle), math.sin(self.azimuth_angle)
        cv, sv = math.cos(self.launch_angle), math.sin(self.launch_angle)
        velocity: np.ndarray = np.array(
            [self.ball_speed * cv * ca, self.ball_speed * cv * sa, self.ball_speed * sv]
        )
        return velocity

    def get_spin_vector(self) -> np.ndarray:
        """Compute the 3D spin vector [rad/s] in the flight frame.

        When ``spin_axis`` is set, returns ``omega * spin_axis``; otherwise
        uses the legacy backspin/sidespin decomposition from
        ``spin_axis_angle`` (UpstreamDrift behaviour).
        """
        omega = self.spin_rate * RPM_TO_RAD_S
        if self.spin_axis is not None:
            spin_vector: np.ndarray = omega * np.asarray(self.spin_axis, dtype=float)
            return spin_vector
        backspin = omega * math.cos(self.spin_axis_angle)
        sidespin = omega * math.sin(self.spin_axis_angle)
        spin_vector = np.array(
            [
                sidespin * math.sin(self.azimuth_angle),
                -backspin,
                sidespin * math.cos(self.azimuth_angle),
            ]
        )
        return spin_vector

    def get_wind_vector(
        self, time_s: float = 0.0, position_m: object = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """Return wind-to velocity at physical time and flight-frame position."""
        if self.wind_scenario is not None:
            wind_vector: np.ndarray = np.asarray(
                self.wind_scenario.velocity_at(time_s, position_m), dtype=float
            )
            return wind_vector
        wind_vector = np.array(
            [
                -self.wind_speed * math.cos(self.wind_direction),
                -self.wind_speed * math.sin(self.wind_direction),
                0.0,
            ]
        )
        return wind_vector


@dataclass(frozen=True)
class TrajectoryPoint:
    """Single point in a flight trajectory (flight frame, SI units)."""

    time: float
    position: np.ndarray = field(repr=False)
    velocity: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Coerce vectors to float arrays and validate shapes/finiteness."""
        if not math.isfinite(self.time) or self.time < 0.0:
            raise ValueError(f"time must be finite and >= 0; got {self.time!r}")
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        for name, vec in (("position", position), ("velocity", velocity)):
            if vec.shape != (3,):
                raise ValueError(f"{name} must be shape (3,); got {vec.shape}")
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"{name} must be finite; got {vec!r}")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)


class FlightTermination(Enum):
    """Reason the flight simulation stopped.

    Distinguishes physical ground landing from truncation at the time horizon,
    numerical solver failure, or external cancellation. Landing-derived metrics
    (carry, landing angle, lateral deviation at landing) are only well-defined
    for LANDED trajectories.
    """

    LANDED = "landed"
    TIME_LIMIT = "time_limit"
    SOLVER_FAILED = "solver_failed"
    CANCELLED = "cancelled"

    @property
    def flight_completed(self) -> bool:
        """Whether the flight ran to a terminal ground landing event."""
        return self is FlightTermination.LANDED

    @property
    def landed(self) -> bool:
        """Alias for flight_completed."""
        return self is FlightTermination.LANDED


class IncompleteFlightError(RuntimeError):
    """Raised when landing metrics or terminal state are requested prematurely.

    Carries the partial result so a caller can still inspect the trace -- the
    trace is evidence, and refusing to export a conclusion is not a reason to
    discard it.
    """

    def __init__(self, result: FlightResult) -> None:
        self.result = result
        self.termination = result.termination
        final_z = float(result.trajectory[-1].position[2]) if result.trajectory else 0.0
        final_t = float(result.trajectory[-1].time) if result.trajectory else 0.0
        super().__init__(
            f"cannot export landing metrics from flight that did not land: "
            f"termination={result.termination.value}, "
            f"final time={final_t:.4f} s, "
            f"final height={final_z:.4f} m. "
            f"Inspect .result for the partial trace."
        )


@dataclass(frozen=True)
class FlightResult:
    """Result of a ball-flight simulation.

    ``trajectory`` is time-ordered; scalar metrics are derived from it.
    Landing-derived metrics (carry [m], landing angle [deg, positive downward],
    and lateral deviation [m, +left in the flight frame]) are only populated
    when ``termination`` is ``FlightTermination.LANDED``; otherwise they are
    ``None``.
    """

    trajectory: tuple[TrajectoryPoint, ...]
    model_name: str
    carry_distance: float | None = None
    max_height: float = 0.0
    flight_time: float = 0.0
    landing_angle: float | None = None
    lateral_deviation: float | None = None
    termination: FlightTermination = FlightTermination.LANDED
    terminal_event: bool = True
    actual_horizon: float = 0.0

    def __post_init__(self) -> None:
        """Normalise trajectory to a tuple and validate landing metrics."""
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        if not isinstance(self.termination, FlightTermination):
            raise TypeError(
                f"termination must be a FlightTermination; got {self.termination!r}"
            )
        if self.termination.landed:
            if self.carry_distance is None:
                object.__setattr__(self, "carry_distance", 0.0)
            if self.landing_angle is None:
                object.__setattr__(self, "landing_angle", 0.0)
            if self.lateral_deviation is None:
                object.__setattr__(self, "lateral_deviation", 0.0)
        else:
            if (
                self.carry_distance is not None
                or self.landing_angle is not None
                or self.lateral_deviation is not None
            ):
                raise ValueError(
                    "Landing metrics (carry_distance, landing_angle, "
                    "lateral_deviation) must be None when flight did not land "
                    f"(termination={self.termination.value})"
                )

    @property
    def flight_completed(self) -> bool:
        """Whether the flight ran to a terminal ground landing event."""
        return self.termination.flight_completed

    @property
    def landed(self) -> bool:
        """Whether the flight reached the ground."""
        return self.termination.landed

    def require_landing(self) -> FlightResult:
        """Assert that the flight reached the ground, or raise IncompleteFlightError."""
        if not self.landed:
            raise IncompleteFlightError(self)
        return self

    def to_position_array(self) -> np.ndarray:
        """Convert trajectory to an Nx3 position array."""
        if not self.trajectory:
            return np.zeros((0, 3))
        positions: np.ndarray = np.array([p.position for p in self.trajectory])
        return positions


def compute_flight_metrics(
    trajectory: list[TrajectoryPoint] | tuple[TrajectoryPoint, ...],
    model_name: str,
    *,
    termination: FlightTermination = FlightTermination.LANDED,
    terminal_event: bool = True,
    actual_horizon: float | None = None,
) -> FlightResult:
    """Standardised metrics computation shared by all backends.

    Ported from ``BallFlightModel._compute_metrics`` in UpstreamDrift's
    ``flight_models.py`` (consolidated for DRY there; module-level here so
    the Rust facade reuses it).
    """
    if trajectory is None:
        raise ValueError("trajectory must be provided")
    points = tuple(trajectory)
    horizon = (
        actual_horizon
        if actual_horizon is not None
        else (points[-1].time if points else 0.0)
    )
    if not points:
        return FlightResult(
            (),
            model_name,
            carry_distance=0.0 if termination.landed else None,
            max_height=0.0,
            flight_time=0.0,
            landing_angle=0.0 if termination.landed else None,
            lateral_deviation=0.0 if termination.landed else None,
            termination=termination,
            terminal_event=terminal_event,
            actual_horizon=horizon,
        )

    pos = np.array([p.position for p in points])
    max_h = float(np.max(pos[:, 2]))
    time = points[-1].time

    if not termination.landed:
        return FlightResult(
            points,
            model_name,
            carry_distance=None,
            max_height=max_h,
            flight_time=time,
            landing_angle=None,
            lateral_deviation=None,
            termination=termination,
            terminal_event=terminal_event,
            actual_horizon=horizon,
        )

    carry = math.hypot(float(pos[-1, 0]), float(pos[-1, 1]))
    lateral = float(pos[-1, 1])

    angle = 0.0
    if len(points) >= 1:
        v = points[-1].velocity
        v_horiz = math.hypot(float(v[0]), float(v[1]))
        angle = (
            math.degrees(math.atan2(-float(v[2]), v_horiz))
            if v_horiz > MIN_SPEED_THRESHOLD_M_S
            else 90.0
        )

    return FlightResult(
        points,
        model_name,
        carry_distance=carry,
        max_height=max_h,
        flight_time=time,
        landing_angle=angle,
        lateral_deviation=lateral,
        termination=termination,
        terminal_event=terminal_event,
        actual_horizon=horizon,
    )


__all__ = [
    "DEFAULT_BACKSPIN_AXIS",
    "FlightResult",
    "FlightTermination",
    "IncompleteFlightError",
    "LaunchConditions",
    "TrajectoryPoint",
    "compute_flight_metrics",
]
