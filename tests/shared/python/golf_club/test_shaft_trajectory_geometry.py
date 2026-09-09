"""Independent quaternion integration and geometric trajectory controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.golf_club import _shaft_moving_trajectory as trajectory
from shared.python.golf_club._shaft_moving_chain import moving_chain_response
from shared.python.golf_club._shaft_moving_contracts import MovingChainState

from .test_shaft_inertia import _hat
from .test_shaft_moving_chain import _fixture, _nonplanar_case
from .test_shaft_moving_trajectory import _polynomial_problem


def _screw_problem() -> tuple:
    chain, initial, controls = _nonplanar_case()
    anchor = chain.grips[0].anchor
    twist = np.asarray(anchor.twist)

    def anchors(time_s: float) -> tuple:
        return (
            replace(
                anchor,
                pose=np.asarray(anchor.pose) @ expm(_hat(time_s * twist)),
                twist_rate=np.zeros(6),
            ),
        )

    return trajectory.MovingTrajectoryProblem(chain, controls, anchors), initial


def _quaternion_state(values: np.ndarray) -> MovingChainState:
    rows = values.reshape(-1, 13)
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    poses[:, :3, 3] = rows[:, :3]
    return MovingChainState(poses, rows[:, 7:], "observer")


def _quaternion_rhs(
    problem: trajectory.MovingTrajectoryProblem, time_s: float, values: np.ndarray
) -> np.ndarray:
    state = _quaternion_state(values)
    response = moving_chain_response(problem.chain_at(time_s), state, problem.controls)
    rows = values.reshape(-1, 13)
    output = np.zeros_like(rows)
    for index, row in enumerate(rows):
        rotation = np.asarray(state.poses)[index, :3, :3]
        vector, scalar, omega = row[3:6], row[6], row[10:13]
        output[index, :3] = rotation @ row[7:10]
        output[index, 3:6] = (scalar * omega + np.cross(vector, omega)) / 2
        output[index, 6] = -vector @ omega / 2
    output[:, 7:] = response.twist_rates
    return output.ravel()


def _reference(
    problem: trajectory.MovingTrajectoryProblem,
    initial: MovingChainState,
    tolerance: float,
) -> MovingChainState:
    poses = np.asarray(initial.poses)
    values = np.c_[
        poses[:, :3, 3],
        Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        initial.twists,
    ]
    calls = 0

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert calls <= 2000, "independent reference work budget exhausted"
        return _quaternion_rhs(problem, time_s, state)

    result = solve_ivp(
        rhs,
        (0, 0.004),
        values.ravel(),
        method="DOP853",
        rtol=tolerance,
        atol=tolerance / 10,
    )
    assert result.success and result.t[-1] == 0.004
    return _quaternion_state(result.y[:, -1])


def _state_error(left: MovingChainState, right: MovingChainState) -> float:
    poses = np.asarray(left.poses) - np.asarray(right.poses)
    poses[:, :3, 3] /= 0.1  # Declared synthetic position scale, in metres.
    twists = np.asarray(left.twists) - np.asarray(right.twists)
    return float(np.linalg.norm(np.r_[poses.ravel(), twists.ravel()]))


def test_nonplanar_pose_and_velocity_converge_to_independent_quaternion_solver() -> (
    None
):
    problem, initial = _screw_problem()
    reference = _reference(problem, initial, 1e-10)
    tighter = _reference(problem, initial, 1e-12)
    assert _state_error(reference, tighter) < 1e-8
    errors = []
    # The 32-step error was 0.0178: refine without changing the 1e-4 target.
    for steps in (128, 256, 512):
        request = trajectory.MovingTrajectoryControls((0, 0.004), steps, 2 * steps + 1)
        result = trajectory.integrate_moving_chain(problem, initial, request)
        errors.append(_state_error(result.samples[-1].state, tighter))
        for sample in result.samples:
            for pose in np.asarray(sample.state.poses):
                rotation = pose[:3, :3]
                np.testing.assert_allclose(
                    rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0
                )
                assert np.linalg.det(rotation) == pytest.approx(1, abs=1e-12)
                np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])
    assert 3 < errors[0] / errors[1] < 5
    assert 3 < errors[1] / errors[2] < 5
    assert errors[-1] < 1e-4


def test_history_is_evaluated_once_per_requested_stage_and_endpoint() -> None:
    problem, initial = _polynomial_problem()
    calls = []

    def history(time_s: float) -> tuple:
        calls.append(time_s)
        return tuple(problem.anchor_history(time_s))

    request = trajectory.MovingTrajectoryControls((0, 0.001), 4, 9)
    result = trajectory.integrate_moving_chain(
        replace(problem, anchor_history=history), initial, request
    )
    np.testing.assert_allclose(calls, np.linspace(0, 0.001, 9), atol=0, rtol=1e-15)
    assert len(calls) == result.evaluation_count


def test_time_shifted_history_preserves_the_same_physical_trajectory() -> None:
    problem, initial = _polynomial_problem()
    plain = trajectory.integrate_moving_chain(
        problem, initial, trajectory.MovingTrajectoryControls((0, 0.001), 4, 9)
    )
    shifted = replace(problem, anchor_history=lambda t: problem.anchor_history(t - 2))
    result = trajectory.integrate_moving_chain(
        shifted, initial, trajectory.MovingTrajectoryControls((2, 2.001), 4, 9)
    )
    assert _state_error(plain.samples[-1].state, result.samples[-1].state) < 1e-11
    assert plain.samples[-1].anchor_work_j == pytest.approx(
        result.samples[-1].anchor_work_j, abs=1e-12
    )


def test_excessive_rotation_step_is_refused_without_chart_clipping() -> None:
    chain, initial, controls = _fixture()
    twists = np.tile([0, 0, 0, 0, 0, 10], (2, 1))
    initial = replace(initial, twists=twists)
    problem = trajectory.MovingTrajectoryProblem(
        replace(chain, grips=()), controls, lambda _: ()
    )
    with pytest.raises(ValueError, match="branch boundary"):
        trajectory.integrate_moving_chain(
            problem, initial, trajectory.MovingTrajectoryControls((0, 1), 1, 3)
        )


def test_stage_strain_limits_are_enforced_during_the_trajectory() -> None:
    chain, initial, controls = _fixture()
    twists = np.zeros((2, 6))
    twists[1, 2] = 1
    initial = replace(initial, twists=twists)
    limited = replace(controls, strain_limits=(1e-7,) * 6)
    problem = trajectory.MovingTrajectoryProblem(
        chain, limited, lambda _: (chain.grips[0].anchor,)
    )
    with pytest.raises(ValueError, match="strain"):
        trajectory.integrate_moving_chain(
            problem, initial, trajectory.MovingTrajectoryControls((0, 0.001), 1, 3)
        )
