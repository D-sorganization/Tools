"""Independent affine motion/work and strict fourth-order Lie method controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from shared.python.golf_club._shaft_rkmk_trajectory import (
    RkmkTrajectoryControls,
    integrate_rkmk_chain,
)

from .test_shaft_moving_trajectory import (
    _exact_generator,
    _exact_powers,
    _polynomial_problem,
)


def test_affine_moving_grip_state_and_three_work_integrals_have_fourth_order() -> None:
    problem, initial = _polynomial_problem()
    duration = 0.006
    generator, exact_initial = _exact_generator()
    exact = (expm(generator * duration) @ exact_initial)[:4]
    work = np.array(
        [
            quad(lambda t, i=axis: _exact_powers(t)[i], 0, duration)[0]
            for axis in range(3)
        ]
    )
    state_errors, work_errors = [], []
    for steps in (4, 8, 16):
        request = RkmkTrajectoryControls((0, duration), steps, 4 * steps + 1)
        result = integrate_rkmk_chain(problem, initial, request)
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
        assert result.evaluation_count == 4 * steps + 1
        assert len(result.samples) == steps + 1
        assert final.time_s == duration
        assert result.stability_status == "unqualified"
        assert result.evidence_status == "time-discrete-unqualified"
    for errors in (state_errors, work_errors):
        assert 12 < errors[0] / errors[1] < 20
        assert 12 < errors[1] / errors[2] < 20
    assert state_errors[-1] < 1e-5
    assert work_errors[-1] < 1e-7
    assert abs(final.energy_balance_error_j) < 1e-7
    assert np.all(abs(work) > 1e-6)


def test_fourth_order_budget_is_checked_before_history_access() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return tuple(problem.anchor_history(time_s))

    request = RkmkTrajectoryControls((0, 0.001), 4, 16)
    with pytest.raises(ValueError, match="budget"):
        integrate_rkmk_chain(replace(problem, anchor_history=history), initial, request)
    assert calls == []


def test_all_stage_and_endpoint_acceleration_calls_are_counted() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return tuple(problem.anchor_history(time_s))

    request = RkmkTrajectoryControls((0, 0.001), 2, 9)
    result = integrate_rkmk_chain(
        replace(problem, anchor_history=history), initial, request
    )
    np.testing.assert_allclose(
        calls,
        [0, 0.00025, 0.00025, 0.0005, 0.0005, 0.00075, 0.00075, 0.001, 0.001],
        rtol=1e-15,
        atol=0,
    )
    assert len(calls) == result.evaluation_count


def test_midpoint_controls_cannot_silently_select_the_new_method() -> None:
    from shared.python.golf_club._shaft_moving_trajectory import (
        MovingTrajectoryControls,
    )

    problem, initial = _polynomial_problem()
    with pytest.raises(TypeError, match="RkmkTrajectoryControls"):
        integrate_rkmk_chain(
            problem, initial, MovingTrajectoryControls((0, 0.001), 2, 9)
        )


def test_fourth_order_controls_cannot_silently_run_the_midpoint_method() -> None:
    from shared.python.golf_club._shaft_moving_trajectory import integrate_moving_chain

    problem, initial = _polynomial_problem()
    with pytest.raises(TypeError, match="midpoint"):
        integrate_moving_chain(
            problem, initial, RkmkTrajectoryControls((0, 0.001), 2, 9)
        )


def test_fourth_order_grid_refuses_unrepresentable_time_before_history() -> None:
    problem, initial = _polynomial_problem()
    request = RkmkTrajectoryControls((1, np.nextafter(1.0, np.inf)), 1, 5)
    with pytest.raises(ValueError, match="time.*unresolved"):
        integrate_rkmk_chain(problem, initial, request)


def test_history_failure_never_returns_a_partial_fourth_order_trajectory() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        if time_s >= 0.00075:
            raise ValueError("synthetic missing anchor history")
        return tuple(problem.anchor_history(time_s))

    request = RkmkTrajectoryControls((0, 0.001), 2, 9)
    with pytest.raises(ValueError, match="missing anchor history"):
        integrate_rkmk_chain(replace(problem, anchor_history=history), initial, request)
    assert 0.0005 in calls
    assert len(calls) <= 9
