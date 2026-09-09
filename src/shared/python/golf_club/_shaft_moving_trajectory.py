"""Explicit Lie midpoint shaft trajectories with separate inertial work ports."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_moving_chain import MovingChainResponse, moving_chain_response
from ._shaft_moving_contracts import MovingChainState
from ._shaft_se3 import exp_twist
from ._shaft_trajectory_contracts import (
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
)


@dataclass(frozen=True)
class MovingTrajectorySample:
    """Owned endpoint state, force/energy response and accumulated SI work.

    Anchor work is positive out of the shaft/grip system. Energy-balance error
    is E(t)-E(0)-W_applied+W_anchor+D. It is a discretization diagnostic, never
    removed by rescaling velocities, changing loads or injecting correction work.
    """

    time_s: float
    state: MovingChainState
    response: MovingChainResponse
    applied_work_j: float
    anchor_work_j: float
    dissipated_energy_j: float
    energy_balance_error_j: float


@dataclass(frozen=True)
class MovingTrajectory:
    """Complete fixed-step trajectory; convergence and physical data are separate.

    Constitutive/pose domains are checked at stages and endpoints only. No
    continuous strain enclosure, event resolution, adaptivity, stability or
    symplectic/energy-conserving claim follows from a successful evaluation.
    """

    controls: MovingTrajectoryControls
    samples: tuple[MovingTrajectorySample, ...]
    evaluation_count: int

    @property
    def evidence_status(self) -> str:
        return "time-discrete-unqualified"

    @property
    def stability_status(self) -> str:
        return "unqualified"


@dataclass(frozen=True)
class _WorkLedger:
    initial_energy_j: float
    work_j: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def advance(self, response: MovingChainResponse, step_s: float) -> _WorkLedger:
        powers = np.array(
            [
                response.applied_power_w,
                response.anchor_power_w,
                response.dissipated_power_w,
            ]
        )
        increments = finite_array(step_s * powers, (3,), "work increments")
        work = tuple(
            math.fsum((old, change))
            for old, change in zip(self.work_j, increments, strict=True)
        )
        finite_array(work, (3,), "accumulated work")
        return _WorkLedger(self.initial_energy_j, work)  # type: ignore[arg-type]


def _sample(
    time_s: float,
    state: MovingChainState,
    response: MovingChainResponse,
    ledger: _WorkLedger,
) -> MovingTrajectorySample:
    applied, anchor, dissipated = ledger.work_j
    defect = math.fsum(
        (
            response.total_energy_j,
            -ledger.initial_energy_j,
            -applied,
            anchor,
            dissipated,
        )
    )
    finite_array(defect, (), "trajectory energy balance")
    return MovingTrajectorySample(
        time_s, state, response, applied, anchor, dissipated, defect
    )


def _chart_state(
    state: MovingChainState, increments: np.ndarray, velocities: np.ndarray
) -> MovingChainState:
    poses = np.array(
        [
            pose @ exp_twist(increment)
            for pose, increment in zip(np.asarray(state.poses), increments, strict=True)
        ]
    )
    return MovingChainState(poses, velocities, state.observer_id)


def _advance(
    state: MovingChainState, motion: tuple[np.ndarray, np.ndarray], step_s: float
) -> MovingChainState:
    velocity, rates = motion
    return _chart_state(
        state, step_s * velocity, np.asarray(state.twists) + step_s * rates
    )


def _response(
    problem: MovingTrajectoryProblem, state: MovingChainState, time_s: float
) -> MovingChainResponse:
    return moving_chain_response(problem.chain_at(time_s), state, problem.controls)


def _step(
    problem: MovingTrajectoryProblem,
    state: MovingChainState,
    response: MovingChainResponse,
    cell: tuple[float, float, float],
) -> _StepResult:
    start, middle, end = cell
    midpoint = _advance(
        state,
        (np.asarray(state.twists), np.asarray(response.twist_rates)),
        middle - start,
    )
    midpoint_response = _response(problem, midpoint, middle)
    final = _advance(
        state,
        (np.asarray(midpoint.twists), np.asarray(midpoint_response.twist_rates)),
        end - start,
    )
    return final, _response(problem, final, end), ((midpoint_response, 1.0),)


_StepResult = tuple[
    MovingChainState,
    MovingChainResponse,
    tuple[tuple[MovingChainResponse, float], ...],
]


@dataclass(frozen=True)
class _MethodPlan:
    evaluations_per_step: int
    step: Callable[
        [
            MovingTrajectoryProblem,
            MovingChainState,
            MovingChainResponse,
            tuple[float, float, float],
        ],
        _StepResult,
    ]


def _integrate(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    controls: MovingTrajectoryControls,
    method: _MethodPlan,
) -> MovingTrajectory:
    """Share owned endpoints, numerical refusal and explicit work quadrature."""
    if not isinstance(problem, MovingTrajectoryProblem):
        raise TypeError("problem must be MovingTrajectoryProblem")
    if not isinstance(initial, MovingChainState):
        raise TypeError("initial must be MovingChainState")
    if not isinstance(controls, MovingTrajectoryControls):
        raise TypeError("controls must be MovingTrajectoryControls")
    cells = controls.time_cells()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            response = _response(problem, initial, controls.bounds_s[0])
            ledger = _WorkLedger(response.total_energy_j)
            samples = [_sample(controls.bounds_s[0], initial, response, ledger)]
            state = initial
            for cell in cells:
                state, response, work_samples = method.step(
                    problem, state, response, cell
                )
                for work_response, weight in work_samples:
                    ledger = ledger.advance(work_response, weight * (cell[2] - cell[0]))
                samples.append(_sample(cell[2], state, response, ledger))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("moving trajectory numerical evaluation failed") from error
    return MovingTrajectory(
        controls, tuple(samples), method.evaluations_per_step * controls.steps + 1
    )


def integrate_moving_chain(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    controls: MovingTrajectoryControls,
) -> MovingTrajectory:
    """Integrate Hdot=H*hat(V), Vdot=a(t,H,V) using explicit Lie midpoint.

    Preconditions: fixed constitutive/load laws, consistent prescribed inertial
    anchor history, proper initial poses, material strain limits and sufficient
    finite work budget. Postcondition: every requested endpoint with midpoint
    work quadrature and unaltered energy defects, or an exception without a
    partial result. Second-order accuracy requires smoothness and refinement;
    no implicit stabilization, energy projection or domain clipping is applied.
    Require the concrete midpoint controls; another method's controls must not
    silently produce midpoint motion or a different evaluation-count contract.
    """
    if type(controls) is not MovingTrajectoryControls:
        raise TypeError("midpoint controls must be MovingTrajectoryControls")
    return _integrate(problem, initial, controls, _MethodPlan(2, _step))


__all__ = ()
