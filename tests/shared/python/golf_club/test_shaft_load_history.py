"""Independent time-dependent force/work and explicit observer-load contracts."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from shared.python.golf_club._grip_moving_kinematics import MaterialPointMotion
from shared.python.golf_club._shaft_chain import IndexedPointLoad
from shared.python.golf_club._shaft_load_history import PrescribedPointLoads
from shared.python.golf_club._shaft_moving_chain import moving_chain_response
from shared.python.golf_club._shaft_point_load import SpatialPointLoad
from shared.python.golf_club._shaft_rkmk_trajectory import RkmkTrajectoryControls
from shared.python.golf_club._shaft_trajectory_contracts import MovingTrajectoryProblem
from shared.python.swing_sim.impact._normal_contact_events import (
    integrate_adaptive_normal_contact,
)
from shared.python.swing_sim.impact._normal_contact_trajectory import (
    integrate_normal_contact,
)

from .test_normal_contact_events import _motion, _request
from .test_normal_contact_trajectory import _generator, _problem
from .test_shaft_moving_chain import _nonplanar_case


def _polynomial_load(t: float) -> tuple[IndexedPointLoad, ...]:
    return (
        IndexedPointLoad(
            1,
            SpatialPointLoad((0, 0, 0.4 + 500 * t + 1e5 * t * t), (0, 0, 0), (0, 0, 0)),
        ),
    )


def _history(
    sample: object = None, observer: str = "observer", bounds: tuple = (0, 0.001)
) -> PrescribedPointLoads:
    return PrescribedPointLoads(
        observer, bounds, _polynomial_load if sample is None else sample
    )


def _reference(time_s: float) -> tuple[np.ndarray, np.ndarray]:
    # Independent three-mass equations; augment the existing polynomial forcing.
    generator, initial = _generator()
    mass = np.zeros((3, 3))
    mass[:2, :2] = 0.2 / 6 * np.array([[2, 1], [1, 2]]) + np.diag([0.03, 0.1])
    mass[2, 2] = 0.046
    generator[3:6, 6:] += np.linalg.solve(
        mass, np.array([[0, 0, 0], [0.4, 500, 1e5], [0, 0, 0]])
    )
    state = expm(time_s * generator) @ initial
    rate = generator @ state
    q, v = state[:3], state[3:6]
    anchor = -0.01 + 0.3 * time_s + 0.2 * time_s**2
    anchor_v = 0.3 + 0.4 * time_s
    effort = 0.03 * (rate[3] - 0.4) + 2 * (v[0] - anchor_v) + 400 * (q[0] - anchor)
    assert q[1] > q[2] and 2e4 * (q[1] - q[2]) + 3 * (v[1] - v[2]) > 0
    work = np.array(
        [
            (1.1 + 500 * time_s + 1e5 * time_s**2) * v[1],
            effort * anchor_v,
            2 * (v[0] - anchor_v) ** 2,
            3 * (v[1] - v[2]) ** 2,
            0,
        ]
    )
    return state[:6], work


def test_prescribed_force_contact_motion_and_work_have_independent_refinement(
    record_property: Callable[[str, object], None],
) -> None:
    # Preset: fine state <2e-8, work/defect <1e-9J, two RK4 ratios10..22.
    problem, initial = _problem()
    problem = replace(problem, additional_load_history=_history())
    end = 0.0004
    expected, _ = _reference(end)
    exact_work = [
        quad(lambda t, i=i: _reference(t)[1][i], 0, end, epsabs=1e-13)[0]
        for i in range(5)
    ]
    errors = []
    for steps in (4, 8, 16):
        result = integrate_normal_contact(
            problem, initial, RkmkTrajectoryControls((0, end), steps, 4 * steps + 1)
        )
        final = result.samples[-1]
        errors.append(np.linalg.norm(_motion(final.state) - expected))
    assert errors[-1] < 2e-8
    assert all(10 < a / b < 22 for a, b in zip(errors[:-1], errors[1:], strict=True))
    np.testing.assert_allclose(final.work.values, exact_work, atol=1e-9, rtol=0)
    assert abs(final.energy_balance_error_j) < 1e-9
    adaptive = integrate_adaptive_normal_contact(
        problem, initial, replace(_request(1e-10, 5e-5), bounds_s=(0, end), steps=2)
    )
    np.testing.assert_allclose(
        _motion(adaptive.samples[-1].state), expected, atol=2e-9, rtol=0
    )
    np.testing.assert_allclose(
        adaptive.samples[-1].work.values, exact_work, atol=1e-9, rtol=0
    )
    generator, baseline_initial = _generator()
    baseline = (expm(end * generator) @ baseline_initial)[:6]
    record_property(
        "load_history_study",
        json.dumps(
            {
                "end_s": end,
                "rk4_steps": [4, 8, 16],
                "state_errors": errors,
                "expected_state": expected.tolist(),
                "baseline_state": baseline.tolist(),
                "expected_work_j": exact_work,
                "observed_work_j": final.work.values,
                "energy_defect_j": final.energy_balance_error_j,
                "qualification": "synthetic-prescribed-load-only",
            }
        ),
    )


def test_force_offset_and_free_couple_use_material_point_power_without_mutation() -> (
    None
):
    chain, state, controls = _nonplanar_case()
    original = chain.shaft.elastic.loads
    force, couple, offset = (
        np.array([0.3, -0.1, 0.2]),
        np.array([0.02, 0.03, -0.01]),
        np.array([0.01, -0.02, 0.03]),
    )
    load = IndexedPointLoad(1, SpatialPointLoad(force, couple, offset))
    calls = []

    def sample(t: float) -> tuple[IndexedPointLoad, ...]:
        calls.append(t)
        return (load,)

    history = _history(sample)

    def anchor(t: float) -> tuple[MaterialPointMotion, ...]:
        return tuple(port.anchor for port in chain.grips)

    problem = MovingTrajectoryProblem(chain, controls, anchor, history)
    updated = problem.chain_at(0.0002)
    assert calls == [0.0002] and updated.shaft.elastic.loads == (*original, load)
    before = moving_chain_response(chain, state, controls)
    after = moving_chain_response(updated, state, controls)
    rotation = np.asarray(state.poses)[1, :3, :3]
    velocity = np.asarray(state.twists)[1]
    expected = force @ (
        rotation @ (velocity[:3] + np.cross(velocity[3:], offset))
    ) + couple @ (rotation @ velocity[3:])
    assert after.applied_power_w - before.applied_power_w == pytest.approx(
        expected, abs=1e-12
    )
    assert abs(after.power_residual_w) < 1e-9
    assert np.linalg.norm(np.asarray(after.twist_rates) - before.twist_rates) > 1e-4
    assert chain.shaft.elastic.loads == original


@pytest.mark.parametrize(
    "bounds", [(0, 0), (-1, 1), (1, 0), (0, float("nan")), (False, 1)]
)
def test_history_rejects_invalid_time_domain(bounds: tuple) -> None:
    with pytest.raises((TypeError, ValueError)):
        _history(bounds=bounds)


@pytest.mark.parametrize("time_s", [-0.01, 0.002, float("nan"), True])
def test_history_refuses_uncovered_time_before_callback(time_s: object) -> None:
    problem, _ = _problem()
    calls = []
    history = _history(lambda t: calls.append(t) or ())
    with pytest.raises((TypeError, ValueError)):
        history.apply(problem.contact.chain, time_s)
    assert not calls


def test_history_refuses_wrong_observer_and_invalid_callback() -> None:
    problem, _ = _problem()
    calls = []
    with pytest.raises(ValueError, match="observer"):
        _history(lambda t: calls.append(t) or (), observer="another").apply(
            problem.contact.chain, 0
        )
    assert not calls
    with pytest.raises(TypeError):
        _history(42)


@pytest.mark.parametrize(
    "sampled",
    [
        None,
        ["load"],
        ("load",),
        (IndexedPointLoad(3, SpatialPointLoad((0, 0, 1), (0, 0, 0), (0, 0, 0))),),
    ],
)
def test_history_refuses_untyped_or_outside_loads(sampled: object) -> None:
    problem, _ = _problem()
    with pytest.raises((TypeError, ValueError)):
        _history(lambda t: sampled).apply(problem.contact.chain, 0)


def test_callback_failure_and_missing_coverage_return_no_trajectory() -> None:
    problem, initial = _problem()

    def missing(t: float) -> tuple[IndexedPointLoad, ...]:
        raise LookupError("force measurement missing")

    with pytest.raises(LookupError, match="measurement missing"):
        integrate_normal_contact(
            replace(problem, additional_load_history=_history(missing)),
            initial,
            RkmkTrajectoryControls((0, 0.0004), 4, 17),
        )
    with pytest.raises(ValueError, match="coverage"):
        integrate_normal_contact(
            replace(problem, additional_load_history=_history(bounds=(0, 0.0001))),
            initial,
            RkmkTrajectoryControls((0, 0.0004), 4, 17),
        )


def test_each_history_sample_appends_once_without_accumulating_stage_loads() -> None:
    problem, initial = _problem()
    calls = []
    history = _history(lambda t: calls.append(t) or ())
    driven = replace(problem, additional_load_history=history)
    baseline = problem.evaluate(initial, 0)
    for t in (0, 0.0001, 0):
        actual = driven.evaluate(initial, t)
        if t == 0:
            np.testing.assert_array_equal(
                actual.shaft.twist_rates, baseline.shaft.twist_rates
            )
    assert calls == [0, 0.0001, 0]
    assert len(problem.contact.chain.shaft.elastic.loads) == 1


@pytest.mark.parametrize("contact_problem", [False, True])
def test_problem_refuses_untyped_load_history(contact_problem: bool) -> None:
    problem, _ = _problem()
    target = (
        problem
        if contact_problem
        else MovingTrajectoryProblem(
            problem.contact.chain, problem.contact.controls, problem.anchor_history
        )
    )
    with pytest.raises(TypeError, match="PrescribedPointLoads"):
        replace(target, additional_load_history=lambda t: ())
