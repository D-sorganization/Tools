"""Classical RK4 in local SE(3) charts with explicit body-Jacobian correction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from ._rkmk_step import RkmkStepModel, rkmk_step
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
from ._shaft_trajectory_contracts import (
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
)

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


def _step(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    response: MovingChainResponse,
    cell: tuple[float, float, float],
) -> _StepResult:
    model: RkmkStepModel[MovingChainState, MovingChainResponse] = RkmkStepModel(
        _chart_state,
        partial(_response, problem),
        lambda state: state.twists,
        lambda result: np.asarray(result.twist_rates),
    )
    return rkmk_step(model, initial, response, cell)


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
