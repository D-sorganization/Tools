"""Classical RK4 in local SE(3) charts with explicit body-Jacobian correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._shaft_moving_chain import MovingChainResponse
from ._shaft_moving_contracts import MovingChainState
from ._shaft_moving_trajectory import (
    MovingTrajectory,
    _chart_state,
    _integrate,
    _MethodPlan,
    _response,
    _StepResult,
)
from ._shaft_se3 import right_jacobian
from ._shaft_trajectory_contracts import (
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
)

_RK4_WEIGHTS = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
_EVALUATIONS_PER_STEP = 4


@dataclass(frozen=True)
class RkmkTrajectoryControls(MovingTrajectoryControls):
    """Fixed RK4 grid with exactly 4*steps+1 acceleration evaluations.

    Three internal stage responses and each accepted endpoint supplement the
    cached start response. All count against the budget, preflighted before
    history access. The inherited SI/type/grid domains are unchanged.
    """

    def time_cells(self) -> tuple[tuple[float, float, float], ...]:
        if _EVALUATIONS_PER_STEP * self.steps + 1 > self.max_evaluations:
            raise ValueError(
                "RKMK trajectory acceleration evaluation budget is insufficient"
            )
        return tuple(super().time_cells())


@dataclass(frozen=True)
class _Stage:
    chart_rate: np.ndarray
    acceleration: np.ndarray
    response: MovingChainResponse


def _stage(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    increments: tuple[np.ndarray, np.ndarray],
    time_s: float,
) -> _Stage:
    coordinates, velocity_change = increments
    velocities = np.asarray(initial.twists) + velocity_change
    state = _chart_state(initial, coordinates, velocities)
    response = _response(problem, state, time_s)
    chart_rate = np.array(
        [
            np.linalg.solve(right_jacobian(coordinate), velocity)
            for coordinate, velocity in zip(coordinates, velocities, strict=True)
        ]
    )
    return _Stage(chart_rate, np.asarray(response.twist_rates), response)


def _step(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    response: MovingChainResponse,
    cell: tuple[float, float, float],
) -> _StepResult:
    start, middle, end = cell
    step_s = end - start
    stages = [
        _Stage(np.asarray(initial.twists), np.asarray(response.twist_rates), response)
    ]
    for fraction, time_s in zip((0.5, 0.5, 1.0), (middle, middle, end), strict=True):
        previous = stages[-1]
        increments = (
            step_s * fraction * previous.chart_rate,
            step_s * fraction * previous.acceleration,
        )
        stages.append(_stage(problem, initial, increments, time_s))
    coordinates = sum(
        weight * stage.chart_rate
        for weight, stage in zip(_RK4_WEIGHTS, stages, strict=True)
    )
    acceleration = sum(
        weight * stage.acceleration
        for weight, stage in zip(_RK4_WEIGHTS, stages, strict=True)
    )
    final = _chart_state(
        initial,
        step_s * coordinates,
        np.asarray(initial.twists) + step_s * acceleration,
    )
    work = tuple(
        (stage.response, weight)
        for stage, weight in zip(stages, _RK4_WEIGHTS, strict=True)
    )
    return final, _response(problem, final, end), work


def integrate_rkmk_chain(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    controls: RkmkTrajectoryControls,
) -> MovingTrajectory:
    """Integrate local qdot=Jr(q)^-1 V and Vdot=a(t,H0 Exp(q),V) by RK4.

    Reset q=0 each step; retain proper poses, complete requested endpoints,
    stage/endpoint constitutive guards, separate RK-weighted work integrals
    and unaltered energy defects. The existing midpoint entrypoint is distinct.
    Smoothness, temporal/mesh refinement and valid prescribed histories remain
    necessary. No adaptivity, continuous-domain enclosure, symplecticity,
    exact energy conservation or unconditional stability is claimed.
    """
    if not isinstance(controls, RkmkTrajectoryControls):
        raise TypeError("controls must be RkmkTrajectoryControls")
    return _integrate(
        problem, initial, controls, _MethodPlan(_EVALUATIONS_PER_STEP, _step)
    )


__all__ = ()
