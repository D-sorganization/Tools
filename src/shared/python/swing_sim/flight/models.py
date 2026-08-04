"""Literature ball-flight models (scipy ``solve_ivp`` RK45 integration).

Ported from UpstreamDrift ``src/shared/python/physics/flight_models.py``
(epic #4103, flight port #4107), rewritten self-contained: the
:class:`BallFlightModel` base with the unified ODE loop and terminal ground
event, the Waterloo/Penner quadratic-coefficient model, the
MacDonald-Hanzely spin-decay model, and the constant-coefficient model
family parameterised by :class:`ConstantCoefficientSpec` (name /
description / reference metadata preserved as the citation trail).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.integrate import solve_ivp

from ._constants import (
    MAX_GOLF_BALL_LIFT_COEFFICIENT,
    MIN_SPEED_THRESHOLD_M_S,
    NUMERICAL_EPSILON,
    RPM_TO_RAD_S,
)
from .types import (
    FlightResult,
    LaunchConditions,
    TrajectoryPoint,
    compute_flight_metrics,
)


def _capped_lift_coefficient(value: float) -> float:
    """Return a physically bounded golf-ball lift coefficient."""
    if value <= 0.0:
        return 0.0
    return min(MAX_GOLF_BALL_LIFT_COEFFICIENT, value)


def _spin_ratio_lift_coefficient(spin_ratio: float, max_coefficient: float) -> float:
    """Calibrate low-spin lift without letting high-spin shots balloon."""
    if spin_ratio <= 0.0 or max_coefficient <= 0.0:
        return 0.0
    return _capped_lift_coefficient(min(max_coefficient, 1.7 * spin_ratio))


class BallFlightModel(ABC):
    """Base class for flight models (shared ODE loop and metrics)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of the flight model."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a short description of the model approach."""
        ...

    @property
    @abstractmethod
    def reference(self) -> str:
        """Return the citation or reference for the model."""
        ...

    @abstractmethod
    def simulate(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        """Simulate ball flight and return the trajectory result."""
        ...

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Delegate to the shared module-level metrics computation."""
        return compute_flight_metrics(trajectory, self.name)

    def _run_ode_simulation(
        self,
        launch: LaunchConditions,
        deriv_func: Callable[[float, np.ndarray], np.ndarray],
        max_time: float,
        dt: float,
    ) -> FlightResult:
        """Unified ODE integration loop with a terminal ground event.

        Preconditions: ``launch`` provided, ``max_time > 0``, ``0 < dt``.
        """
        if launch is None:
            raise ValueError("launch must be provided")
        if not (math.isfinite(max_time) and max_time > 0.0):
            raise ValueError(f"max_time must be finite and > 0; got {max_time!r}")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0; got {dt!r}")
        v0 = launch.get_initial_velocity()
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

        def ground_ev(t: float, y: np.ndarray) -> float:
            """Return the ball height for ground-contact event detection."""
            return float(y[2])

        # Type-safe attribute assignment for solve_ivp
        setattr(ground_ev, "terminal", True)  # noqa: B010
        setattr(ground_ev, "direction", -1)  # noqa: B010

        sol = solve_ivp(
            deriv_func,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_ev,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        points = [
            TrajectoryPoint(float(t), sol.sol(t)[:3], sol.sol(t)[3:]) for t in t_eval
        ]
        if sol.t[-1] not in t_eval:
            points.append(
                TrajectoryPoint(float(sol.t[-1]), sol.y[:3, -1], sol.y[3:, -1])
            )

        return self._compute_metrics(points)


class WaterlooPennerModel(BallFlightModel):
    """Waterloo/Penner model implementation."""

    def __init__(
        self,
        cd0: float = 0.21,
        cd1: float = 0.05,
        cd2: float = 0.02,
        cl0: float = 0.00,
        cl1: float = 0.70,
        cl2: float = 0.645,
        cl_max: float = MAX_GOLF_BALL_LIFT_COEFFICIENT,
    ) -> None:
        self.params = (cd0, cd1, cd2, cl0, cl1, cl2, cl_max)

    @property
    def name(self) -> str:
        """Return the Waterloo/Penner model name."""
        return "Waterloo/Penner"

    @property
    def description(self) -> str:
        """Return the Waterloo/Penner model description."""
        return "Waterloo quadratic Cd with Penner spin-ratio lift fit"

    @property
    def reference(self) -> str:
        """Return the Waterloo/Penner model citation."""
        return "Penner (2003); McPhee et al. (Waterloo)"

    def simulate(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        """Simulate flight using the quadratic-Cd / power-law-Cl model."""
        if launch is None:
            raise ValueError("launch must be provided")
        cd0, cd1, cd2, cl0, cl1, cl2, cl_max = self.params
        omega_v = launch.get_spin_vector()
        omega_m = math.hypot(omega_v[0], omega_v[1], omega_v[2])
        wind_v = launch.get_wind_vector()
        area = math.pi * launch.ball_radius**2

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            """Compute state derivatives using quadratic Cd/Cl aerodynamics."""
            if t is None:
                raise ValueError("t must be provided")
            v_val = cast(np.ndarray, y[3:])
            v_rel = v_val - wind_v
            speed = math.hypot(v_rel[0], v_rel[1], v_rel[2])
            if speed < MIN_SPEED_THRESHOLD_M_S:
                return np.array(
                    [v_val[0], v_val[1], v_val[2], 0.0, 0.0, -launch.gravity]
                )

            vu = v_rel / speed
            s = (omega_m * launch.ball_radius) / speed
            cd = cd0 + cd1 * s + cd2 * s**2
            cl_val = cl0 + cl1 * s**cl2 if s > 0.0 else cl0
            cl = min(cl_max, _capped_lift_coefficient(cl_val))

            acc = (
                -(0.5 * launch.air_density * speed**2 * cd * area / launch.ball_mass)
                * vu
            )
            if omega_m > 0:
                cross = np.cross(omega_v / omega_m, vu)
                cross_norm = math.hypot(cross[0], cross[1], cross[2])
                if cross_norm > NUMERICAL_EPSILON:
                    acc += (
                        0.5
                        * launch.air_density
                        * speed**2
                        * cl
                        * area
                        / launch.ball_mass
                    ) * (cross / cross_norm)

            acc[2] -= launch.gravity
            return np.array([v_val[0], v_val[1], v_val[2], acc[0], acc[1], acc[2]])

        return self._run_ode_simulation(launch, derivatives, max_time, dt)


class MacDonaldHanzelyModel(BallFlightModel):
    """MacDonald-Hanzely model implementation."""

    def __init__(
        self, cd: float = 0.225, cl: float = 0.20, decay: float = 0.05
    ) -> None:
        self.cd, self.cl, self.decay = cd, cl, decay

    @property
    def name(self) -> str:
        """Return the MacDonald-Hanzely model name."""
        return "MacDonald-Hanzely"

    @property
    def description(self) -> str:
        """Return the MacDonald-Hanzely model description."""
        return "ODE model with exponential spin decay"

    @property
    def reference(self) -> str:
        """Return the MacDonald-Hanzely model citation."""
        return "MacDonald & Hanzely (1991)"

    def simulate(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        """Simulate flight using the MacDonald-Hanzely spin-decay model."""
        if launch is None:
            raise ValueError("launch must be provided")
        omega_0 = launch.spin_rate * RPM_TO_RAD_S
        spin_axis = launch.get_spin_vector()
        spin_norm = math.hypot(spin_axis[0], spin_axis[1], spin_axis[2])
        if spin_norm > 0:
            spin_axis = spin_axis / spin_norm
        wind_v = launch.get_wind_vector()
        area = math.pi * launch.ball_radius**2
        k_drag = 0.5 * launch.air_density * area * self.cd / launch.ball_mass

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            """Compute state derivatives with exponential spin decay."""
            if t is None:
                raise ValueError("t must be provided")
            v_val = cast(np.ndarray, y[3:])
            v_rel = v_val - wind_v
            speed = math.hypot(v_rel[0], v_rel[1], v_rel[2])
            if speed < MIN_SPEED_THRESHOLD_M_S:
                return np.array(
                    [v_val[0], v_val[1], v_val[2], 0.0, 0.0, -launch.gravity]
                )

            omega = omega_0 * math.exp(-self.decay * t)
            vu = v_rel / speed
            acc = -k_drag * speed**2 * vu

            if omega > 0:
                spin_ratio = omega * launch.ball_radius / speed
                cl_eff = _spin_ratio_lift_coefficient(spin_ratio, self.cl)
                cross = np.cross(spin_axis, vu)
                cross_norm = math.hypot(cross[0], cross[1], cross[2])
                if cross_norm > NUMERICAL_EPSILON:
                    acc += (
                        0.5
                        * launch.air_density
                        * area
                        * cl_eff
                        * speed**2
                        / launch.ball_mass
                    ) * (cross / cross_norm)

            acc[2] -= launch.gravity
            return np.array([v_val[0], v_val[1], v_val[2], acc[0], acc[1], acc[2]])

        return self._run_ode_simulation(launch, derivatives, max_time, dt)


@dataclass(frozen=True)
class ConstantCoefficientSpec:
    """Specification for constant coefficient flight models.

    Attributes:
        name: Display name for the model.
        description: Short description of the model.
        reference: Citation or reference for the model.
        cd: Drag coefficient [unitless].
        cl: Lift coefficient [unitless].
        spin_decay: Spin decay rate [1/s]. Use 0.0 to disable decay.
    """

    name: str
    description: str
    reference: str
    cd: float
    cl: float
    spin_decay: float


class ConstantCoefficientModel(BallFlightModel):
    """Flight model using constant Cd/Cl with optional spin decay."""

    def __init__(self, spec: ConstantCoefficientSpec) -> None:
        self._spec = spec

    @property
    def name(self) -> str:
        """Return the model name from the specification."""
        return self._spec.name

    @property
    def description(self) -> str:
        """Return the model description from the specification."""
        return self._spec.description

    @property
    def reference(self) -> str:
        """Return the model citation from the specification."""
        return self._spec.reference

    def simulate(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        """Simulate flight using constant drag and lift coefficients."""
        if launch is None:
            raise ValueError("launch must be provided")
        omega_0 = launch.spin_rate * RPM_TO_RAD_S
        spin_axis = launch.get_spin_vector()
        spin_norm = math.hypot(spin_axis[0], spin_axis[1], spin_axis[2])
        if spin_norm > 0:
            spin_axis = spin_axis / spin_norm
        wind_v = launch.get_wind_vector()
        area = math.pi * launch.ball_radius**2
        k_drag = 0.5 * launch.air_density * area * self._spec.cd / launch.ball_mass

        def derivatives(t: float, y: np.ndarray) -> np.ndarray:
            """Compute state derivatives with constant coefficients and decay."""
            if t is None:
                raise ValueError("t must be provided")
            v_val = cast(np.ndarray, y[3:])
            v_rel = v_val - wind_v
            speed = math.hypot(v_rel[0], v_rel[1], v_rel[2])
            if speed < MIN_SPEED_THRESHOLD_M_S:
                return np.array(
                    [v_val[0], v_val[1], v_val[2], 0.0, 0.0, -launch.gravity]
                )

            omega = omega_0 * math.exp(-self._spec.spin_decay * t)
            vu = v_rel / speed
            acc = -k_drag * speed**2 * vu

            if omega > 0:
                spin_ratio = omega * launch.ball_radius / speed
                cl_eff = _spin_ratio_lift_coefficient(spin_ratio, self._spec.cl)
                cross = np.cross(spin_axis, vu)
                cross_norm = math.hypot(cross[0], cross[1], cross[2])
                if cross_norm > NUMERICAL_EPSILON:
                    acc += (
                        0.5
                        * launch.air_density
                        * area
                        * cl_eff
                        * speed**2
                        / launch.ball_mass
                    ) * (cross / cross_norm)

            acc[2] -= launch.gravity
            return np.array([v_val[0], v_val[1], v_val[2], acc[0], acc[1], acc[2]])

        return self._run_ode_simulation(launch, derivatives, max_time, dt)


__all__ = [
    "BallFlightModel",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "MacDonaldHanzelyModel",
    "WaterlooPennerModel",
]
