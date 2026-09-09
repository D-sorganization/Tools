"""Noncommuting geometric accuracy and stage-domain controls for local RK4."""

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import _shaft_rkmk_trajectory as rkmk
from shared.python.golf_club._shaft_moving_trajectory import MovingTrajectoryProblem

from .test_shaft_moving_chain import _fixture
from .test_shaft_trajectory_geometry import _reference, _screw_problem, _state_error


def test_noncommuting_motion_has_fourth_order_against_quaternion_reference(
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _screw_problem()
    reference = _reference(problem, initial, 1e-10)
    tighter = _reference(problem, initial, 1e-12)
    reference_error = _state_error(reference, tighter)
    assert reference_error < 1e-8
    errors = []
    for steps in (16, 32, 64):
        request = rkmk.RkmkTrajectoryControls((0, 0.004), steps, 4 * steps + 1)
        result = rkmk.integrate_rkmk_chain(problem, initial, request)
        errors.append(_state_error(result.samples[-1].state, tighter))
        for sample in result.samples:
            for pose in np.asarray(sample.state.poses):
                rotation = pose[:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-12)
                np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])
    assert 10 < errors[0] / errors[1] < 22
    assert 10 < errors[1] / errors[2] < 22
    assert errors[-1] < 1e-4
    assert reference_error < 0.01 * errors[-1]
    record_property("study", json.dumps({"steps": [16, 32, 64], "errors": errors}))


@pytest.mark.parametrize("mistake", ["omit", "opposite_sign"])
def test_missing_or_wrong_sign_chart_correction_fails_the_same_reference(
    mistake: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem, initial = _screw_problem()
    reference = _reference(problem, initial, 1e-12)
    request = rkmk.RkmkTrajectoryControls((0, 0.004), 64, 257)
    correct = rkmk.integrate_rkmk_chain(problem, initial, request)
    correct_error = _state_error(correct.samples[-1].state, reference)
    jacobian = rkmk.right_jacobian
    wrong = (lambda q: np.eye(6)) if mistake == "omit" else (lambda q: jacobian(-q))
    monkeypatch.setattr(rkmk, "right_jacobian", wrong)
    corrupted = rkmk.integrate_rkmk_chain(problem, initial, request)
    wrong_error = _state_error(corrupted.samples[-1].state, reference)
    assert correct_error < 1e-4
    assert wrong_error > 5 * correct_error


def test_fourth_order_stage_rotation_is_refused_without_clipping() -> None:
    chain, initial, controls = _fixture()
    initial = replace(initial, twists=np.tile([0, 0, 0, 0, 0, 10], (2, 1)))
    problem = MovingTrajectoryProblem(replace(chain, grips=()), controls, lambda _: ())
    with pytest.raises(ValueError, match="branch boundary"):
        rkmk.integrate_rkmk_chain(
            problem, initial, rkmk.RkmkTrajectoryControls((0, 1), 1, 5)
        )


def test_fourth_order_stage_strain_is_refused() -> None:
    chain, initial, controls = _fixture()
    twists = np.zeros((2, 6))
    twists[1, 2] = 1
    initial = replace(initial, twists=twists)
    problem = MovingTrajectoryProblem(
        chain,
        replace(controls, strain_limits=(1e-7,) * 6),
        lambda _: (chain.grips[0].anchor,),
    )
    with pytest.raises(ValueError, match="strain"):
        rkmk.integrate_rkmk_chain(
            problem, initial, rkmk.RkmkTrajectoryControls((0, 0.001), 1, 5)
        )


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", 0, -1])
@pytest.mark.parametrize("field", ["steps", "max_evaluations"])
def test_fourth_order_counts_retain_strict_integer_contract(
    field: str, value: object
) -> None:
    request = rkmk.RkmkTrajectoryControls((0, 0.001), 1, 5)
    with pytest.raises((TypeError, ValueError)):
        replace(request, **{field: value})
