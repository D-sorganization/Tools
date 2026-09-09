"""Resolved rotating bending disturbances against independent time integration.

The quaternion reference shares the nonlinear force law; it is independent
integration, not an independent mechanical or experimentally fitted model.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_moving_contracts import MovingChainState
from shared.python.golf_club._shaft_moving_trajectory import (
    MovingTrajectory,
    MovingTrajectoryControls,
    MovingTrajectoryProblem,
    integrate_moving_chain,
)
from shared.python.golf_club._shaft_rkmk_trajectory import (
    RkmkTrajectoryControls,
    integrate_rkmk_chain,
)
from shared.python.golf_club._shaft_se3 import exp_twist

from .test_shaft_trajectory_geometry import _reference
from .test_shaft_trajectory_rotation import _body_twist, _problem

_AMPLITUDE_M = 0.001
_LENGTH_M = 1.0
_RATE_REFERENCE_RAD_S = 100.0
_DURATION_S = 0.004


def _disturbed_problem(
    count: int, amplitude_m: float = _AMPLITUDE_M
) -> tuple[MovingTrajectoryProblem, MovingChainState]:
    problem, equilibrium = _problem(count)
    poses = np.array(equilibrium.poses)
    for index, position in enumerate(np.linspace(0, 1, count + 1)):
        generator = amplitude_m * np.array(
            [
                position**2,
                -0.5 * position**2,
                0,
                position / _LENGTH_M,
                2 * position / _LENGTH_M,
                0,
            ]
        )
        poses[index] = poses[index] @ exp_twist(generator)
    # The initial disturbance has zero relative velocity in the rotating frame.
    initial = replace(
        equilibrium, poses=poses, twists=np.array([_body_twist(pose) for pose in poses])
    )
    return problem, initial


def _disturbance_error(left: MovingChainState, right: MovingChainState) -> float:
    pose_difference = np.asarray(left.poses) - np.asarray(right.poses)
    pose_difference[:, :3, 3] /= _AMPLITUDE_M
    pose_difference[:, :3, :3] /= _AMPLITUDE_M / _LENGTH_M
    velocity_scales = (
        _AMPLITUDE_M
        * _RATE_REFERENCE_RAD_S
        * np.array([1, 1, 1, 1 / _LENGTH_M, 1 / _LENGTH_M, 1 / _LENGTH_M])
    )
    velocity_difference = (np.asarray(left.twists) - np.asarray(right.twists)) / (
        velocity_scales
    )
    return float(
        np.linalg.norm(np.r_[pose_difference.ravel(), velocity_difference.ravel()])
    )


@dataclass(frozen=True)
class _TimeMethod:
    steps: tuple[int, ...]
    controls: type[MovingTrajectoryControls]
    integrate: Callable[..., MovingTrajectory]
    evaluations_per_step: int


def _qualified_reference(
    problem: MovingTrajectoryProblem, initial: MovingChainState
) -> tuple[MovingChainState, float]:
    reference = _reference(problem, initial, 1e-10)
    tighter = _reference(problem, initial, 1e-12)
    reference_error = _disturbance_error(reference, tighter)
    assert reference_error < 1e-6
    return tighter, reference_error


def _time_errors(
    problem: MovingTrajectoryProblem,
    initial: MovingChainState,
    reference: MovingChainState,
    method: _TimeMethod,
) -> list[dict[str, float]]:
    rows = []
    for steps in method.steps:
        evaluations = method.evaluations_per_step * steps + 1
        request = method.controls((0, _DURATION_S), steps, evaluations)
        result = method.integrate(problem, initial, request)
        final = result.samples[-1]
        rows.append(
            {
                "steps": steps,
                "scaled_state_error": _disturbance_error(final.state, reference),
                "energy_balance_error_j": final.energy_balance_error_j,
                "anchor_work_j": final.anchor_work_j,
            }
        )
        assert final.applied_work_j == 0
        assert final.dissipated_energy_j == 0
        assert result.evaluation_count == evaluations
        assert result.stability_status == "unqualified"
    return rows


def _record_study(
    record_property: Callable[[str, object], None],
    elements: int,
    rows: list[dict[str, float]],
    reference_error: float,
) -> None:
    record_property(
        "study",
        json.dumps(
            {
                "elements": elements,
                "time_s": _DURATION_S,
                "scaled_reference_difference": reference_error,
                "refinements": rows,
            }
        ),
    )


@pytest.mark.parametrize("elements", [2, 4])
def test_midpoint_converges_but_256_steps_miss_disturbance_accuracy_target(
    elements: int,
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _disturbed_problem(elements)
    reference, reference_error = _qualified_reference(problem, initial)
    method = _TimeMethod(
        (64, 128, 256), MovingTrajectoryControls, integrate_moving_chain, 2
    )
    rows = _time_errors(problem, initial, reference, method)
    _record_study(record_property, elements, rows, reference_error)
    errors = [row["scaled_state_error"] for row in rows]
    assert 3 < errors[0] / errors[1] < 5
    assert 3 < errors[1] / errors[2] < 5
    assert reference_error < 0.01 * errors[-1]
    # Retain the original accuracy failures as explicit counterexamples.
    assert errors[-1] > 1e-3


@pytest.mark.parametrize("elements", [2, 4])
def test_rkmk_rotating_disturbance_meets_the_same_accuracy_target(
    elements: int,
    record_property: Callable[[str, object], None],
) -> None:
    problem, initial = _disturbed_problem(elements)
    reference, reference_error = _qualified_reference(problem, initial)
    method = _TimeMethod((16, 32, 64), RkmkTrajectoryControls, integrate_rkmk_chain, 4)
    rows = _time_errors(problem, initial, reference, method)
    _record_study(record_property, elements, rows, reference_error)
    errors = [row["scaled_state_error"] for row in rows]
    assert 10 < errors[0] / errors[1] < 22
    assert 10 < errors[1] / errors[2] < 22
    assert reference_error < 0.01 * errors[-1]
    assert errors[-1] < 1e-3
    assert abs(rows[-1]["energy_balance_error_j"]) < 1e-6
