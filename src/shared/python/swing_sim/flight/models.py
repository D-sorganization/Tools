"""Literature ball-flight models with adaptive physical-contact integration."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from shared.python.swing_sim.ground.contract_types import GroundSurfaceProfile

from ._cancellation import (
    FlightCancellationCallbackError,
    FlightSimulationCancelled,
    raise_if_flight_cancelled,
)
from ._constants import MAX_GOLF_BALL_LIFT_COEFFICIENT
from .capability_observation import CancellationCheck
from .dynamics import ConstantCoefficientDynamics, WaterlooDynamics
from .state import FlightStatePoint
from .surface_simulation import (
    SurfaceFlightSimulationSettings,
    flight_ode_signed_gap_m,
)
from .types import (
    FlightResult,
    FlightTermination,
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
    cancellation_requested: CancellationCheck | None = None


class _OdeSolution(Protocol):
    """Structural subset of SciPy's integration result used here."""

    t: np.ndarray
    y: np.ndarray
    sol: Callable[[float], np.ndarray]
    status: int
    success: bool
    t_events: list[np.ndarray] | None


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
        *,
        cancellation_requested: CancellationCheck | None = None,
    ) -> FlightResult:
        """Simulate to planar contact or raise typed cooperative cancellation.

        Args:
            launch: Exact physical launch conditions.
            settings: Exact planar-contact integration settings.
            cancellation_requested: Optional synchronous exact-bool callback.

        Returns:
            A complete flight result; partial results are never returned.

        Raises:
            FlightSimulationCancelled: Cancellation was requested.
            FlightCancellationCallbackError: The callback raised or returned a
                non-boolean value.
        """
        if launch is None:
            raise ValueError("launch must be provided")
        if not isinstance(settings, SurfaceFlightSimulationSettings):
            raise ValueError("settings must be SurfaceFlightSimulationSettings")
        if cancellation_requested is not None and not callable(cancellation_requested):
            raise TypeError("cancellation_requested must be callable or None")
        run = _OdeRun(
            launch,
            self._build_dynamics(launch),
            settings.max_time_s,
            settings.output_interval_s,
            settings.launch_relative_surface,
            cancellation_requested,
        )
        return self._run_ode_simulation(run)

    def _build_dynamics(
        self, launch: LaunchConditions
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """Build this model's state derivative for one launch."""
        raise NotImplementedError("surface-aware simulation is unavailable")

    def _compute_metrics(
        self,
        trajectory: list[TrajectoryPoint],
        *,
        termination: FlightTermination = FlightTermination.LANDED,
        terminal_event: bool = True,
        actual_horizon: float | None = None,
    ) -> FlightResult:
        """Delegate to the shared module-level metrics computation."""
        return compute_flight_metrics(
            trajectory,
            self.name,
            termination=termination,
            terminal_event=terminal_event,
            actual_horizon=actual_horizon,
        )

    def _safe_compute_metrics(
        self,
        trajectory: list[TrajectoryPoint],
        *,
        termination: FlightTermination = FlightTermination.LANDED,
        terminal_event: bool = True,
        actual_horizon: float | None = None,
    ) -> FlightResult:
        try:
            return self._compute_metrics(
                trajectory,
                termination=termination,
                terminal_event=terminal_event,
                actual_horizon=actual_horizon,
            )
        except TypeError:
            result = self._compute_metrics(trajectory)
            if result.termination != termination:
                return compute_flight_metrics(
                    trajectory,
                    result.model_name,
                    termination=termination,
                    terminal_event=terminal_event,
                    actual_horizon=actual_horizon,
                )
            return result

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
        self._validate_ode_run(run)
        v0 = run.launch.get_initial_velocity()
        y0 = np.array([0.0, 0.0, 0.0, v0[0], v0[1], v0[2]])
        last_state = [(0.0, y0.copy())]
        derivatives = self._controlled_derivatives(run, last_state)
        ground_event = self._ground_event(run)

        try:
            raise_if_flight_cancelled(run.cancellation_requested)
            sol = solve_ivp(
                derivatives,
                (0, run.max_time),
                y0,
                method="RK45",
                events=ground_event,
                dense_output=True,
                max_step=0.1,
            )
            raise_if_flight_cancelled(run.cancellation_requested)
        except FlightSimulationCancelled as exc:
            last_t, last_y = last_state[0]
            partial_points: list[TrajectoryPoint] = [
                self._state_point(run.launch, last_t, last_y)
            ]
            partial_result = self._safe_compute_metrics(
                partial_points,
                termination=FlightTermination.CANCELLED,
                terminal_event=False,
                actual_horizon=last_t,
            )
            raise FlightSimulationCancelled(
                str(exc) or "flight simulation cancelled",
                result=partial_result,
            ) from exc

        success = getattr(sol, "success", True)
        status = getattr(sol, "status", 0)
        t_events = getattr(sol, "t_events", None)

        if not success or status < 0:
            termination = FlightTermination.SOLVER_FAILED
            terminal_event = False
            actual_horizon = float(sol.t[-1]) if len(sol.t) > 0 else 0.0
        elif status == 1 and t_events and len(t_events[0]) > 0:
            termination = FlightTermination.LANDED
            terminal_event = True
            actual_horizon = float(t_events[0][0])
        elif status == 1:
            termination = FlightTermination.LANDED
            terminal_event = True
            actual_horizon = float(sol.t[-1])
        else:
            termination = FlightTermination.TIME_LIMIT
            terminal_event = False
            actual_horizon = float(sol.t[-1])

        try:
            points = self._sample_solution(run, sol)
            raise_if_flight_cancelled(run.cancellation_requested)
            result = self._safe_compute_metrics(
                points,
                termination=termination,
                terminal_event=terminal_event,
                actual_horizon=actual_horizon,
            )
            raise_if_flight_cancelled(run.cancellation_requested)
            return result
        except FlightSimulationCancelled as exc:
            partial_points = self._sample_solution(run, sol, ignore_cancellation=True)
            partial_result = self._safe_compute_metrics(
                partial_points,
                termination=FlightTermination.CANCELLED,
                terminal_event=False,
                actual_horizon=float(sol.t[-1]),
            )
            raise FlightSimulationCancelled(
                str(exc) or "flight simulation cancelled",
                result=partial_result,
            ) from exc

    @staticmethod
    def _validate_ode_run(run: _OdeRun) -> None:
        if run.launch is None:
            raise ValueError("launch must be provided")
        if not (math.isfinite(run.max_time) and run.max_time > 0.0):
            raise ValueError(f"max_time must be finite and > 0; got {run.max_time!r}")
        if not (math.isfinite(run.dt) and run.dt > 0.0):
            raise ValueError(f"dt must be finite and > 0; got {run.dt!r}")

    @staticmethod
    def _controlled_derivatives(
        run: _OdeRun,
        last_state: list[tuple[float, np.ndarray]] | None = None,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        if run.cancellation_requested is None:
            if last_state is not None:

                def tracking(time_s: float, state: np.ndarray) -> np.ndarray:
                    last_state[0] = (time_s, state.copy())
                    return run.derivatives(time_s, state)

                return tracking
            return run.derivatives

        def controlled(time_s: float, state: np.ndarray) -> np.ndarray:
            raise_if_flight_cancelled(run.cancellation_requested)
            if last_state is not None:
                last_state[0] = (time_s, state.copy())
            return run.derivatives(time_s, state)

        return controlled

    @staticmethod
    def _ground_event(run: _OdeRun) -> Callable[[float, np.ndarray], float]:
        def ground_event(_time_s: float, state: np.ndarray) -> float:
            if run.launch_relative_surface is not None:
                return float(
                    flight_ode_signed_gap_m(
                        run.launch_relative_surface,
                        run.launch.ball_radius,
                        state,
                    )
                )
            return float(state[2] + run.launch.ball_setup.tee_height_m)

        setattr(ground_event, "terminal", True)  # noqa: B010
        setattr(ground_event, "direction", -1)  # noqa: B010
        return ground_event

    def _sample_solution(
        self,
        run: _OdeRun,
        solution: _OdeSolution,
        *,
        ignore_cancellation: bool = False,
    ) -> list[TrajectoryPoint]:
        def controlled_state_point(
            time_s: float, state: np.ndarray
        ) -> FlightStatePoint:
            if not ignore_cancellation:
                raise_if_flight_cancelled(run.cancellation_requested)
            return self._state_point(run.launch, time_s, state)

        t_eval = np.arange(0, solution.t[-1], run.dt)
        points: list[TrajectoryPoint] = [
            controlled_state_point(
                float(time_s),
                solution.sol(float(time_s)),
            )
            for time_s in t_eval
        ]
        if solution.t[-1] not in t_eval:
            points.append(
                controlled_state_point(
                    float(solution.t[-1]),
                    solution.y[:, -1],
                )
            )
        if not ignore_cancellation:
            raise_if_flight_cancelled(run.cancellation_requested)
        return points


class WaterlooPennerModel(BallFlightModel):
    """Waterloo/Penner model implementation (canonical baseline)."""

    def __init__(
        self,
        cd0: float = 0.21,
        cd1: float = 0.05,
        cd2: float = 0.02,
        cl0: float = 0.00,
        cl1: float = 0.70,
        cl2: float = 0.645,
        cl_max: float = MAX_GOLF_BALL_LIFT_COEFFICIENT,
        spin_decay: float = 0.0,
    ) -> None:
        self.params = (cd0, cd1, cd2, cl0, cl1, cl2, cl_max)
        self.spin_decay = spin_decay

    @property
    def name(self) -> str:
        """Return the Waterloo/Penner model name."""
        return "Waterloo/Penner"

    @property
    def description(self) -> str:
        """Return the Waterloo/Penner model description."""
        return "Waterloo quadratic Cd with Penner spin-ratio lift fit (canonical)"

    @property
    def reference(self) -> str:
        """Return the Waterloo/Penner model citation."""
        return "Penner (2003); McPhee et al. (Waterloo)"

    def _spin_decay_rate(self) -> float:
        """Return exponential spin-decay rate for trajectory state output."""
        return self.spin_decay

    def _build_dynamics(self, launch: LaunchConditions) -> WaterlooDynamics:
        return WaterlooDynamics(launch, self.params, self.spin_decay)


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
    "FlightCancellationCallbackError",
    "FlightSimulationCancelled",
    "MacDonaldHanzelyModel",
    "WaterlooPennerModel",
]
