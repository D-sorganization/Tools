"""Independent finite-time moving-base controls for Tools #5072."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from shared.python.golf_club import _shaft_moving_trajectory as trajectory

from .test_shaft_moving_chain import _axial_case, _fixture


def _polynomial_problem() -> tuple:
    chain, initial, controls = _axial_case()
    anchor = chain.grips[0].anchor

    def anchors(time_s: float) -> tuple:
        pose = np.eye(4)
        pose[2, 3] = -0.01 + 0.3 * time_s + 0.2 * time_s**2
        return (
            replace(
                anchor,
                pose=pose,
                twist=[0, 0, 0.3 + 0.4 * time_s, 0, 0, 0],
                twist_rate=[0, 0, 0.4, 0, 0, 0],
            ),
        )

    return trajectory.MovingTrajectoryProblem(chain, controls, anchors), initial


def _exact_generator() -> tuple[np.ndarray, np.ndarray]:
    """Independent two-mass ODE, augmented with 1, t and t squared."""
    mass = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    stiffness = 1000 * np.array([[1, -1], [-1, 1]]) + np.diag([400, 0])
    damping = np.diag([2.0, 0])
    generator = np.zeros((7, 7))
    generator[:2, 2:4] = np.eye(2)
    generator[2:4, :2] = -np.linalg.solve(mass, stiffness)
    generator[2:4, 2:4] = -np.linalg.solve(mass, damping)
    forcing = np.array([[-4 + 0.6 + 0.012, 120 + 0.8, 80], [0.7, 0, 0]])
    generator[2:4, 4:] = np.linalg.solve(mass, forcing)
    generator[5, 4], generator[6, 5] = 1, 2
    return generator, np.array([0.01, 0.03, 0.2, -0.1, 1, 0, 0])


def _exact_powers(time_s: float) -> np.ndarray:
    generator, initial = _exact_generator()
    state = expm(generator * time_s) @ initial
    rate = generator @ state
    relative = state[0] - (-0.01 + 0.3 * time_s + 0.2 * time_s**2)
    anchor_velocity = 0.3 + 0.4 * time_s
    relative_velocity = state[2] - anchor_velocity
    effort = 400 * relative + 2 * relative_velocity + 0.03 * (rate[2] - 0.4)
    return np.array(
        [0.7 * state[3], effort * anchor_velocity, 2 * relative_velocity**2]
    )


def test_moving_base_trajectory_and_both_work_ports_converge_to_exact_ode() -> None:
    problem, initial = _polynomial_problem()
    duration = 0.006
    generator, exact_initial = _exact_generator()
    exact = (expm(generator * duration) @ exact_initial)[:4]
    work = np.array(
        [
            quad(lambda t, index=i: _exact_powers(t)[index], 0, duration, epsabs=1e-12)[
                0
            ]
            for i in range(3)
        ]
    )
    state_errors, work_errors, balance_errors = [], [], []
    # Coarse 8/16-step energy defects cross zero before the asymptotic regime.
    # Refine the grid; keep the interval, parameters and error limits unchanged.
    for count in (32, 64, 128):
        request = trajectory.MovingTrajectoryControls(
            (0, duration), count, 2 * count + 1
        )
        result = trajectory.integrate_moving_chain(problem, initial, request)
        final = result.samples[-1]
        actual = np.r_[
            np.asarray(final.state.poses)[:, 2, 3] - [0, 1],
            np.asarray(final.state.twists)[:, 2],
        ]
        state_errors.append(
            float(np.linalg.norm((actual - exact) / [0.01, 0.01, 1, 1]))
        )
        actual_work = [
            final.applied_work_j,
            final.anchor_work_j,
            final.dissipated_energy_j,
        ]
        work_errors.append(float(np.linalg.norm(np.asarray(actual_work) - work)))
        balance_errors.append(abs(final.energy_balance_error_j))
        assert result.evaluation_count == 2 * count + 1
        assert len(result.samples) == count + 1
        assert final.time_s == duration
        assert result.evidence_status == "time-discrete-unqualified"
        assert result.stability_status == "unqualified"
    for errors in (state_errors, work_errors, balance_errors):
        assert 3 < errors[0] / errors[1] < 5
        assert 3 < errors[1] / errors[2] < 5
    assert state_errors[-1] < 1e-3
    assert work_errors[-1] < 1e-4
    assert balance_errors[-1] < 1e-4
    assert np.all(np.abs(work) > 1e-6)


def test_budget_is_checked_before_any_history_or_force_evaluation() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return tuple(problem.anchor_history(time_s))

    checked = replace(problem, anchor_history=history)
    request = trajectory.MovingTrajectoryControls((0, 0.001), 4, 8)
    with pytest.raises(ValueError, match="budget"):
        trajectory.integrate_moving_chain(checked, initial, request)
    assert calls == []


def test_history_failure_after_progress_refuses_incomplete_trajectory() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        if time_s > 0.0005:
            raise ValueError("synthetic missing anchor history")
        return tuple(problem.anchor_history(time_s))

    request = trajectory.MovingTrajectoryControls((0, 0.001), 4, 9)
    with pytest.raises(ValueError, match="missing anchor history"):
        trajectory.integrate_moving_chain(
            replace(problem, anchor_history=history), initial, request
        )
    assert any(0 < time <= 0.0005 for time in calls)
    assert len(calls) <= 9


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", 0, -1])
@pytest.mark.parametrize("field", ["steps", "max_evaluations"])
def test_work_counts_are_strict_positive_integers(field: str, value: object) -> None:
    request = trajectory.MovingTrajectoryControls((0, 0.01), 4, 9)
    with pytest.raises((TypeError, ValueError)):
        replace(request, **{field: value})


@pytest.mark.parametrize("bounds", [(0, 0), (1, 0), (-1, 1), (0, np.inf), (False, 1)])
def test_time_bounds_are_strict_finite_forward_intervals(bounds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        trajectory.MovingTrajectoryControls(bounds, 4, 9)


def test_time_grid_refuses_unrepresentable_midpoint_before_history() -> None:
    problem, initial = _polynomial_problem()
    request = trajectory.MovingTrajectoryControls(
        (1.0, np.nextafter(1.0, np.inf)), 1, 3
    )
    with pytest.raises(ValueError, match="time.*unresolved"):
        trajectory.integrate_moving_chain(problem, initial, request)


def test_history_cannot_silently_drop_a_grip_or_change_observer() -> None:
    problem, initial = _polynomial_problem()
    request = trajectory.MovingTrajectoryControls((0, 0.001), 1, 3)
    with pytest.raises(ValueError, match="anchor count"):
        trajectory.integrate_moving_chain(
            replace(problem, anchor_history=lambda _: ()), initial, request
        )
    alien = replace(problem.chain.grips[0].anchor, observer_id="other")
    with pytest.raises(ValueError, match="observers"):
        trajectory.integrate_moving_chain(
            replace(problem, anchor_history=lambda _: (alien,)), initial, request
        )


def test_free_uniform_translation_preserves_poses_velocity_and_energy() -> None:
    chain, initial, controls = _fixture()
    chain = replace(chain, grips=())
    twists = np.tile([0.2, -0.1, 0.3, 0, 0, 0], (2, 1))
    initial = replace(initial, twists=twists)
    problem = trajectory.MovingTrajectoryProblem(chain, controls, lambda _: ())
    request = trajectory.MovingTrajectoryControls((0, 0.01), 4, 9)
    result = trajectory.integrate_moving_chain(problem, initial, request)
    for sample in result.samples:
        expected = np.asarray(initial.poses).copy()
        expected[:, :3, 3] += sample.time_s * twists[:, :3]
        np.testing.assert_allclose(sample.state.poses, expected, atol=1e-13)
        np.testing.assert_allclose(sample.state.twists, twists, atol=1e-12)
        assert abs(sample.energy_balance_error_j) < 1e-13
