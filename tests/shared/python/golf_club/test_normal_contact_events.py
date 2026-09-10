"""Spatial event controls against independent piecewise matrix exponentials."""

import json
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from shared.python.swing_sim.impact import _normal_contact_events as implementation
from shared.python.swing_sim.impact._normal_contact_events import (
    AdaptiveContactControls,
    ContactAbsoluteTolerances,
    integrate_adaptive_normal_contact,
)
from shared.python.swing_sim.impact._normal_contact_trajectory import (
    NormalContactTrajectoryState,
)

from . import _normal_event_oracle as oracle
from ._normal_event_oracle import reference
from .test_normal_contact_trajectory import _problem
from .test_normal_shaft_contact import _case


def test_independent_oracle_refuses_an_inexact_event_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve = oracle.brentq

    def shifted(*args: object, **kwargs: object) -> float:
        return float(solve(*args, **kwargs)) + 1e-7

    monkeypatch.setattr(oracle, "brentq", shifted)
    with pytest.raises(AssertionError, match="reference event root"):
        oracle.reference()


def _request(
    tolerance: float = 1e-9, max_step: float = 1e-4
) -> AdaptiveContactControls:
    return AdaptiveContactControls(
        (0, 0.003),
        6,
        20000,
        max_step,
        tolerance,
        ContactAbsoluteTolerances(*(tolerance * 0.01 for _ in range(5))),
    )


def _initial() -> tuple:
    problem, _ = _problem()
    contact, shaft, ball = _case(gap=0.0001)
    return replace(problem, contact=contact), NormalContactTrajectoryState(shaft, ball)


def _motion(state: NormalContactTrajectoryState) -> np.ndarray:
    return np.r_[
        np.asarray(state.shaft.poses)[:, 2, 3] - [0, 1],
        np.asarray(state.ball.pose)[2, 3] - 1.02,
        np.asarray(state.shaft.twists)[:, 2],
        state.ball.twist[2],
    ]


def test_transversal_events_and_work_match_independent_piecewise_solution(
    record_property: Callable[[str, object], None],
) -> None:
    # Fixed before implementation: state 2e-8, event time 2e-9 s, work/energy 2e-9 J.
    # Tighten both local tolerance and accepted step bound, independently of RK4.
    problem, initial = _initial()
    expected, expected_work, times = reference()
    errors = []
    for tolerance, step in ((1e-6, 2e-4), (1e-8, 1e-4), (1e-10, 5e-5)):
        result = integrate_adaptive_normal_contact(
            problem, initial, _request(tolerance, step)
        )
        final = result.samples[-1]
        errors.append(float(np.linalg.norm(_motion(final.state) - expected)))
        assert result.evidence_status == "event-sampled-contact-unqualified"
        assert len(result.samples) == 7 and final.time_s == 0.003
        assert tuple(item.kind for item in result.events) == (
            "first_touch",
            "force_release",
            "geometric_separation",
        )
        assert result.evaluation_count <= 20000
    assert errors[-1] < 2e-8
    assert errors[-1] < errors[0] / 10
    np.testing.assert_allclose(
        [item.sample.time_s for item in result.events], times, atol=2e-9, rtol=0
    )
    np.testing.assert_allclose(final.work.values, expected_work, atol=2e-9, rtol=0)
    assert abs(final.energy_balance_error_j) < 2e-9
    touch, release, separation = result.events
    assert touch.incoming_force_limit_n == pytest.approx(0.7337854620518724, abs=2e-7)
    assert release.sample.response.contact.gap_m < -2e-5
    assert abs(separation.sample.response.contact.gap_m) < 1e-11
    record_property(
        "study",
        json.dumps(
            {
                "state_errors": errors,
                "reference_event_times_s": times,
                "observed_event_times_s": [
                    item.sample.time_s for item in result.events
                ],
                "reference_work_j": expected_work.tolist(),
                "observed_work_j": final.work.values,
                "energy_balance_error_j": final.energy_balance_error_j,
                "evaluation_count": result.evaluation_count,
            }
        ),
    )


def test_budget_exhaustion_returns_no_partial_trajectory() -> None:
    problem, initial = _initial()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return problem.anchor_history(time_s)

    with pytest.raises(ValueError, match="evaluation budget"):
        integrate_adaptive_normal_contact(
            replace(problem, anchor_history=history),
            initial,
            replace(_request(), max_evaluations=13),
        )
    assert len(calls) == 13


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_step_s", 0),
        ("relative_tolerance", 1e-18),
        ("relative_tolerance", True),
        ("relative_tolerance", np.nan),
    ],
)
def test_invalid_adaptive_controls_are_refused(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(_request(), **{field: value})


def test_force_ceiling_and_history_failures_are_not_solver_success() -> None:
    problem, initial = _initial()
    contact = problem.contact
    capped = replace(contact, law=replace(contact.law, maximum_force_n=0.1))
    with pytest.raises(ValueError, match="ceiling"):
        integrate_adaptive_normal_contact(
            replace(problem, contact=capped), initial, _request()
        )

    def history(time_s: float) -> tuple:
        raise ValueError("unavailable measured anchor")

    with pytest.raises(ValueError, match="unavailable measured"):
        integrate_adaptive_normal_contact(
            replace(problem, anchor_history=history), initial, _request()
        )


def test_separated_spinning_ball_retains_inertial_motion() -> None:
    problem, _ = _problem()
    contact, shaft, ball = _case(gap=0.05)
    velocity = np.array([0.1, 0.2, 0.4, 20, -15, 30])
    initial = NormalContactTrajectoryState(shaft, replace(ball, twist=velocity))
    end = 1e-4
    controls = replace(_request(), bounds_s=(0, end), steps=2)
    result = integrate_adaptive_normal_contact(
        replace(problem, contact=contact), initial, controls
    )
    final = result.samples[-1].state.ball
    wx, wy, wz = velocity[3:]
    rotation = expm(end * np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]))
    pose = np.asarray(final.pose)
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=2e-12, rtol=0)
    np.testing.assert_allclose(
        pose[:3, 3],
        np.asarray(ball.pose)[:3, 3] + end * velocity[:3],
        atol=2e-12,
        rtol=0,
    )
    np.testing.assert_allclose(
        rotation @ np.asarray(final.twist)[:3], velocity[:3], atol=2e-12, rtol=0
    )
    assert result.events == ()
    assert result.samples[-1].work.values[3:] == (0, 0)


@pytest.mark.parametrize(
    "field",
    [
        "position_m",
        "rotation_rad",
        "linear_velocity_mps",
        "angular_velocity_radps",
        "work_j",
    ],
)
def test_each_absolute_tolerance_retains_a_positive_si_domain(field: str) -> None:
    with pytest.raises(ValueError):
        replace(_request().absolute_tolerances, **{field: 0})


def test_incomplete_solver_result_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    problem, initial = _initial()
    monkeypatch.setattr(
        implementation,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(success=False),
    )
    with pytest.raises(ValueError, match="did not complete"):
        integrate_adaptive_normal_contact(problem, initial, _request())


def test_grid_budget_preflight_precedes_prescribed_history() -> None:
    problem, initial = _initial()

    def history(time_s: float) -> tuple:
        pytest.fail("invalid grid reached the history")

    with pytest.raises(ValueError, match="budget"):
        integrate_adaptive_normal_contact(
            replace(problem, anchor_history=history),
            initial,
            replace(_request(), max_evaluations=2),
        )
