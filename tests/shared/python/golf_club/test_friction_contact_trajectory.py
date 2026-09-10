"""Independent endpoint equations and history integrity for friction dynamics."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.impact._friction_contact_trajectory import (
    FrictionTrajectoryControls,
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
    integrate_friction_contact,
)
from shared.python.swing_sim.impact._friction_transport import ContactTransport
from shared.python.swing_sim.impact._tangential_contact_work import (
    TangentialContactLaw,
    TangentialContactState,
)

from .test_normal_contact_trajectory import _generator, _problem


def _friction_case(mu: float = 0.0) -> tuple:
    normal, mechanical = _problem()
    law = TangentialContactLaw(1000.0, mu)
    history = TangentialContactState(law, (0, 0, 0), (0, 0, 1), "observer")
    problem = FrictionTrajectoryProblem(normal, law, ContactTransport.FACE)
    return problem, FrictionTrajectoryState(mechanical, history)


def _controls(steps: int = 4) -> FrictionTrajectoryControls:
    return FrictionTrajectoryControls((0, 0.00004), steps, 10000, 100, 1e-10)


def _motion(state: FrictionTrajectoryState) -> np.ndarray:
    mechanical = state.mechanical
    return np.r_[
        np.asarray(mechanical.shaft.poses)[:, 2, 3] - [0, 1],
        np.asarray(mechanical.ball.pose)[2, 3] - 1.02,
        np.asarray(mechanical.shaft.twists)[:, 2],
        mechanical.ball.twist[2],
    ]


def test_zero_friction_matches_independent_backward_euler_three_mass_system() -> None:
    problem, initial = _friction_case()
    controls = _controls()
    matrix, start = _generator()
    expected = start[:6]
    step = controls.bounds_s[1] / controls.steps
    for index in range(controls.steps):
        time_s = (index + 1) * step
        forcing = matrix[:6, 6:] @ [1, time_s, time_s**2]
        expected = np.linalg.solve(
            np.eye(6) - step * matrix[:6, :6], expected + step * forcing
        )
    result = integrate_friction_contact(problem, initial, controls)
    np.testing.assert_allclose(_motion(result.samples[-1].state), expected, atol=3e-10)
    assert len(result.samples) == controls.steps + 1
    assert result.samples[-1].time_s == controls.bounds_s[1]
    assert result.evaluation_count <= controls.max_evaluations
    assert result.evidence_status == "friction-trajectory-unqualified"
    assert result.samples[-1].plastic_dissipation_j == 0
    assert result.samples[-1].tangential_algorithmic_loss_j == 0
    assert all(s.scaled_solver_residual <= 1e-10 for s in result.samples)


def test_friction_changes_ball_spin_and_preserves_previous_history() -> None:
    problem, initial = _friction_case(0.3)
    mechanical = initial.mechanical
    ball = replace(mechanical.ball, twist=(0.2, 0, -0.4, 0, 0, 0))
    initial = replace(initial, mechanical=replace(mechanical, ball=ball))
    controls = _controls(2)
    first = integrate_friction_contact(problem, initial, controls)
    second = integrate_friction_contact(problem, initial, controls)
    final = first.samples[-1]
    assert final.state.mechanical.ball.twist[0] < ball.twist[0]
    assert final.state.mechanical.ball.twist[4] > 0
    assert final.state.tangential.elastic_energy_j > 0
    assert final.tangential_algorithmic_loss_j > 0
    assert final.plastic_dissipation_j >= 0
    assert final == second.samples[-1]
    assert initial.tangential.elastic_deflection_m == (0, 0, 0)
    assert initial.mechanical.ball is ball
    assert np.isfinite(final.energy_balance_error_j)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 1.0, float("nan"), True])
def test_invalid_solver_tolerance_is_refused(tolerance: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        FrictionTrajectoryControls((0, 0.001), 2, 200, 50, tolerance)


def test_inadequate_global_budget_is_refused_before_history_access() -> None:
    problem, initial = _friction_case()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return ()

    problem = replace(problem, normal=replace(problem.normal, anchor_history=history))
    with pytest.raises(ValueError, match="budget"):
        integrate_friction_contact(
            problem, initial, FrictionTrajectoryControls((0, 0.001), 2, 2, 50, 1e-10)
        )
    assert calls == []


def test_failed_step_budget_returns_no_partial_result() -> None:
    problem, initial = _friction_case()
    with pytest.raises(ValueError, match="budget|converg"):
        integrate_friction_contact(
            problem, initial, replace(_controls(), max_step_evaluations=1)
        )
    assert initial.tangential.elastic_deflection_m == (0, 0, 0)


def test_initial_history_must_match_normal_observer_law_and_admissibility() -> None:
    problem, initial = _friction_case(0.2)
    for history in (
        replace(initial.tangential, normal=(0, 1, 0)),
        replace(initial.tangential, observer_id="other"),
        replace(initial.tangential, law=TangentialContactLaw(2000, 0.2)),
        replace(initial.tangential, elastic_deflection_m=(1, 0, 0)),
    ):
        with pytest.raises(ValueError):
            integrate_friction_contact(
                problem, replace(initial, tangential=history), _controls()
            )
