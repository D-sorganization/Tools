"""Independent three-mass temporal oracle for normal shaft/ball contact."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from shared.python.golf_club._shaft_moving_trajectory import MovingTrajectoryControls
from shared.python.golf_club._shaft_rkmk_trajectory import RkmkTrajectoryControls
from shared.python.swing_sim.impact._normal_contact_trajectory import (
    ContactWorkIntegrals,
    NormalContactTrajectoryProblem,
    NormalContactTrajectoryState,
    integrate_normal_contact,
)

from .test_normal_shaft_contact import _case
from .test_shaft_moving_trajectory import _polynomial_problem


def _problem() -> tuple:
    contact, shaft, ball = _case()
    prescribed, _ = _polynomial_problem()
    return (
        NormalContactTrajectoryProblem(contact, prescribed.anchor_history, "ball"),
        NormalContactTrajectoryState(shaft, ball),
    )


def _generator() -> tuple[np.ndarray, np.ndarray]:
    """Independent linear ODE valid strictly inside compressive contact."""
    mass = np.zeros((3, 3))
    mass[:2, :2] = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    mass[2, 2] = 0.046
    stiffness = np.zeros((3, 3))
    stiffness[:2, :2] = 1000 * np.array([[1, -1], [-1, 1]]) + np.diag([400, 0])
    normal_map = np.array([0, 1, -1])
    stiffness += 2e4 * np.outer(normal_map, normal_map)
    damping = np.diag([2.0, 0, 0]) + 3 * np.outer(normal_map, normal_map)
    generator = np.zeros((9, 9))
    generator[:3, 3:6] = np.eye(3)
    generator[3:6, :3] = -np.linalg.solve(mass, stiffness)
    generator[3:6, 3:6] = -np.linalg.solve(mass, damping)
    forcing = np.array([[-4 + 0.6 + 0.012, 120 + 0.8, 80], [0.7, 0, 0], [0, 0, 0]])
    generator[3:6, 6:] = np.linalg.solve(mass, forcing)
    generator[7, 6], generator[8, 7] = 1, 2
    return generator, np.array([0.01, 0.03, 0.029, 0.2, -0.1, -0.4, 1, 0, 0])


def _powers(time_s: float) -> np.ndarray:
    generator, initial = _generator()
    state = expm(time_s * generator) @ initial
    rate = generator @ state
    q, v = state[:3], state[3:6]
    anchor_q = -0.01 + 0.3 * time_s + 0.2 * time_s**2
    anchor_v = 0.3 + 0.4 * time_s
    effort = 0.03 * (rate[3] - 0.4) + 2 * (v[0] - anchor_v) + 400 * (q[0] - anchor_q)
    assert q[1] > q[2]
    assert 2e4 * (q[1] - q[2]) + 3 * (v[1] - v[2]) > 0
    return np.array(
        [
            0.7 * v[1],
            effort * anchor_v,
            2 * (v[0] - anchor_v) ** 2,
            3 * (v[1] - v[2]) ** 2,
            0,
        ]
    )


def test_compressive_contact_refines_against_independent_motion_and_work(
    record_property: Callable[[str, object], None],
) -> None:
    # Fixed before generated results: state error <2e-8, work/energy <1e-9,
    # two timestep-halving ratios 10..22; no contact switch in this control.
    problem, initial = _problem()
    generator, exact_initial = _generator()
    end = 0.0004
    expected = (expm(end * generator) @ exact_initial)[:6]
    expected_work = np.array(
        [
            quad(
                lambda t, index=i: _powers(t)[index], 0, end, epsabs=1e-13, epsrel=1e-11
            )[0]
            for i in range(5)
        ]
    )
    errors = []
    for steps in (4, 8, 16):
        result = integrate_normal_contact(
            problem, initial, RkmkTrajectoryControls((0, end), steps, 4 * steps + 1)
        )
        sample = result.samples[-1]
        shaft, ball = sample.state.shaft, sample.state.ball
        position = np.r_[
            np.asarray(shaft.poses)[:, 2, 3] - [0, 1],
            np.asarray(ball.pose)[2, 3] - 1.02,
        ]
        velocity = np.r_[np.asarray(shaft.twists)[:, 2], ball.twist[2]]
        errors.append(float(np.linalg.norm(np.r_[position, velocity] - expected)))
        assert result.evaluation_count == 4 * steps + 1
        assert len(result.samples) == steps + 1
        assert result.evidence_status == "time-discrete-contact-unqualified"
    assert errors[-1] < 2e-8
    assert 10 < errors[0] / errors[1] < 22
    assert 10 < errors[1] / errors[2] < 22
    np.testing.assert_allclose(sample.work.values, expected_work, atol=1e-9, rtol=0)
    assert abs(sample.energy_balance_error_j) < 1e-9
    assert sample.time_s == end
    record_property(
        "study",
        json.dumps(
            {
                "steps": [4, 8, 16],
                "state_errors": errors,
                "work_expected_j": expected_work.tolist(),
                "work_observed_j": sample.work.values,
                "energy_balance_error_j": sample.energy_balance_error_j,
            }
        ),
    )


def test_history_is_evaluated_at_every_stage_and_endpoint() -> None:
    problem, initial = _problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return problem.anchor_history(time_s)

    request = RkmkTrajectoryControls((0, 1e-5), 1, 5)
    integrate_normal_contact(replace(problem, anchor_history=history), initial, request)
    assert calls == [0, 5e-6, 5e-6, 1e-5, 1e-5]


@pytest.mark.parametrize(
    "bad_controls",
    [
        MovingTrajectoryControls((0, 1e-5), 1, 5),
        RkmkTrajectoryControls((0, 1e-5), 2, 5),
    ],
)
def test_invalid_method_or_budget_is_refused_before_history(
    bad_controls: object,
) -> None:
    problem, initial = _problem()

    def history(time_s: float) -> tuple:
        pytest.fail("invalid request reached prescribed history")

    with pytest.raises((TypeError, ValueError)):
        integrate_normal_contact(
            replace(problem, anchor_history=history), initial, bad_controls
        )


def test_contact_ceiling_failure_does_not_return_partial_trajectory() -> None:
    problem, initial = _problem()
    contact = problem.contact
    law = replace(contact.law, maximum_force_n=10)
    problem = replace(problem, contact=replace(contact, law=law))
    with pytest.raises(ValueError, match="ceiling"):
        integrate_normal_contact(
            problem, initial, RkmkTrajectoryControls((0, 1e-5), 1, 5)
        )


def test_clearance_preserves_independent_free_ball_translation_and_rotation() -> None:
    problem, _ = _problem()
    contact, shaft, ball = _case(gap=0.05)
    initial_twist = np.array([0.1, 0.2, 0.4, 0.3, -0.2, 0.1])
    initial = NormalContactTrajectoryState(shaft, replace(ball, twist=initial_twist))
    problem = replace(problem, contact=contact)
    end = 1e-4
    result = integrate_normal_contact(
        problem, initial, RkmkTrajectoryControls((0, end), 4, 17)
    )
    sample = result.samples[-1]
    final_ball = sample.state.ball
    pose = np.asarray(final_ball.pose)
    expected_position = np.asarray(ball.pose)[:3, 3] + end * initial_twist[:3]
    angular = initial_twist[3:]
    cross = np.array(
        [
            [0, -angular[2], angular[1]],
            [angular[2], 0, -angular[0]],
            [-angular[1], angular[0], 0],
        ]
    )
    np.testing.assert_allclose(pose[:3, 3], expected_position, atol=2e-12, rtol=0)
    np.testing.assert_allclose(pose[:3, :3], expm(end * cross), atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        pose[:3, :3] @ np.asarray(final_ball.twist)[:3],
        initial_twist[:3],
        atol=2e-12,
        rtol=0,
    )
    assert all(item.response.normal.force_n == 0 for item in result.samples)
    assert sample.work.values[3:] == (0, 0)


def test_history_failure_propagates_after_initial_evaluation() -> None:
    problem, initial = _problem()

    def history(time_s: float) -> tuple:
        if time_s > 0:
            raise ValueError("missing prescribed grip measurement")
        return problem.anchor_history(time_s)

    with pytest.raises(ValueError, match="missing prescribed grip"):
        integrate_normal_contact(
            replace(problem, anchor_history=history),
            initial,
            RkmkTrajectoryControls((0, 1e-5), 1, 5),
        )


@pytest.mark.parametrize(
    "values", [(0, 0, -1, 0, 0), (0, 0, 0, np.nan, 0), (0, 0, 0, 0), (True, 0, 0, 0, 0)]
)
def test_work_integrals_refuse_invalid_values(values: tuple) -> None:
    with pytest.raises((ValueError, TypeError)):
        ContactWorkIntegrals(values)


def test_frame_mismatch_is_refused_at_the_problem_or_state_boundary() -> None:
    problem, initial = _problem()
    with pytest.raises(ValueError, match="material frame"):
        replace(problem, ball_material_frame_id="unrelated")
    with pytest.raises(ValueError, match="observer"):
        replace(initial, ball=replace(initial.ball, observer_id="unrelated"))
