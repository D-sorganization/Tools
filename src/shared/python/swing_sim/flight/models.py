"""Literature ball-flight models with adaptive physical-contact integration."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from shared.python.swing_sim.ground import GroundSurfaceProfile

from ._constants import MAX_GOLF_BALL_LIFT_COEFFICIENT
from .dynamics import ConstantCoefficientDynamics, WaterlooDynamics
from .state import FlightStatePoint
from .surface_simulation import (
    SurfaceFlightSimulationSettings,
    flight_ode_signed_gap_m,
)
from .types import (
    FlightResult,
    LaunchConditions,
    TrajectoryPoint,
    compute_flight_metrics,
)


@dataclass(frozen=True)
class _OdeRun:
    launch: LaunchConditions
    derivatives: Callable[[float, np.ndarray], np.ndarray]
    max_time: float
    dt: float
    launch_relative_surface: GroundSurfaceProfile | None = None


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

    def simulate(
        self, launch: LaunchConditions, max_time: float = 10.0, dt: float = 0.01
    ) -> FlightResult:
        """Simulate ball flight and return the trajectory result."""
        if launch is None:
            raise ValueError("launch must be provided")
        run = _OdeRun(launch, self._build_dynamics(launch), max_time, dt)
        return self._run_ode_simulation(run)

    def simulate_to_surface(
        self,
        launch: LaunchConditions,
        settings: SurfaceFlightSimulationSettings,
    ) -> FlightResult:
        """Simulate until sphere contact with the configured planar surface."""
        if launch is None:
            raise ValueError("launch must be provided")
        if not isinstance(settings, SurfaceFlightSimulationSettings):
            raise ValueError("settings must be SurfaceFlightSimulationSettings")
        run = _OdeRun(
            launch,
            self._build_dynamics(launch),
            settings.max_time_s,
            settings.output_interval_s,
            settings.launch_relative_surface,
        )
        return self._run_ode_simulation(run)

    def _build_dynamics(
        self, launch: LaunchConditions
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Build this model's state derivative for one launch."""
        raise NotImplementedError("surface-aware simulation is unavailable")

    def _compute_metrics(self, trajectory: list[TrajectoryPoint]) -> FlightResult:
        """Delegate to the shared module-level metrics computation."""
        return compute_flight_metrics(trajectory, self.name)

    def _spin_decay_rate(self) -> float:
        """Return exponential spin-decay rate for trajectory state output."""
        return 0.0

    def _state_point(
        self, launch: LaunchConditions, time_s: float, state: np.ndarray
    ) -> FlightStatePoint:
        decay = math.exp(-self._spin_decay_rate() * time_s)
        omega = launch.get_spin_vector() * decay
        return FlightStatePoint(time_s, state[:3], state[3:], omega)

    def _run_ode_simulation(self, run: _OdeRun) -> FlightResult:
        """Unified ODE integration loop with a terminal ground event.

        Preconditions: ``launch`` provided, ``max_time > 0``, ``0 < dt``.
        """
        launch = run.launch
        max_time = run.max_time
        dt = run.dt
        if launch is None:
            raise ValueError("launch must be provided")
        if not (math.isfinite(max_time) and max_time > 0.0):
            raise ValueError(f"max_time must be finite and > 0; got {max_time!r}")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(f"dt must be finite and > 0; got {dt!r}")
        v0 = launch.get_initial_velocity()
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])

        def ground_ev(t: float, y: np.ndarray) -> float:
            """Return the configured signed sphere gap for contact detection."""
            if run.launch_relative_surface is not None:
                return flight_ode_signed_gap_m(
                    run.launch_relative_surface,
                    launch.ball_radius,
                    y,
                )
            return float(y[2] + launch.ball_setup.tee_height_m)

        # Type-safe attribute assignment for solve_ivp
        setattr(ground_ev, "terminal", True)  # noqa: B010
        setattr(ground_ev, "direction", -1)  # noqa: B010

        sol = solve_ivp(
            run.derivatives,
            (0, max_time),
            y0,
            method="RK45",
            events=ground_ev,
            dense_output=True,
            max_step=0.1,
        )

        t_eval = np.arange(0, sol.t[-1], dt)
        points: list[TrajectoryPoint] = [
            self._state_point(launch, float(t), sol.sol(t)) for t in t_eval
        ]
        if sol.t[-1] not in t_eval:
            points.append(self._state_point(launch, float(sol.t[-1]), sol.y[:, -1]))
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

    def _build_dynamics(self, launch: LaunchConditions) -> WaterlooDynamics:
        return WaterlooDynamics(launch, self.params)


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

    def _spin_decay_rate(self) -> float:
        return self.decay

    def _build_dynamics(self, launch: LaunchConditions) -> ConstantCoefficientDynamics:
        return ConstantCoefficientDynamics(launch, self.cd, self.cl, self.decay)


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

    def _spin_decay_rate(self) -> float:
        return self._spec.spin_decay

    def _build_dynamics(self, launch: LaunchConditions) -> ConstantCoefficientDynamics:
        return ConstantCoefficientDynamics(
            launch,
            self._spec.cd,
            self._spec.cl,
            self._spec.spin_decay,
        )


__all__ = [
    "BallFlightModel",
    "ConstantCoefficientModel",
    "ConstantCoefficientSpec",
    "MacDonaldHanzelyModel",
    "WaterlooPennerModel",
]
