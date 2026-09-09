"""Independent moving-anchor controls; no human impedance is inferred."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.golf_club._grip_moving_kinematics import (
    MaterialPointMotion,
    moving_grip_kinematics,
)


def _pose(position: tuple[float, ...], rotation: tuple[float, ...]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = position
    return result


def _motions() -> tuple[MaterialPointMotion, MaterialPointMotion]:
    root = MaterialPointMotion(
        _pose((0.1, 0.3, -0.2), (0.2, 0.4, -0.3)),
        (0.8, -0.3, 0.6, 0.2, -0.5, 0.1),
        (0.1, 0.4, -0.2, -0.3, 0.2, 0.4),
        "world",
    )
    anchor = MaterialPointMotion(
        _pose((-0.2, 0.1, 0.4), (-0.4, 0.3, 0.2)),
        (0.3, 0.1, -0.2, -0.4, 0.3, 0.2),
        (-0.2, 0.3, 0.1, 0.1, -0.2, 0.3),
        "world",
    )
    return root, anchor


def _curve(motion: MaterialPointMotion, time: float) -> np.ndarray:
    # H(t)=H0 Exp(t V0 + t² A0/2) has the declared body twist/rate at zero.
    vector = time * np.asarray(motion.twist) + time**2 / 2 * np.asarray(
        motion.twist_rate
    )
    generator = np.zeros((4, 4))
    x, y, z = vector[3:]
    generator[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    generator[:3, 3] = vector[:3]
    return np.asarray(motion.pose) @ expm(generator)


def _relative(root: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    rotation = anchor[:3, :3].T
    return np.r_[
        rotation @ (root[:3, 3] - anchor[:3, 3]),
        Rotation.from_matrix(rotation @ root[:3, :3]).as_rotvec(),
    ]


def test_rates_match_independent_finite_pose_curves() -> None:
    root, anchor = _motions()
    result = moving_grip_kinematics(root, anchor)
    step = 2e-4
    values = {
        k: _relative(_curve(root, k * step), _curve(anchor, k * step))
        for k in (-2, -1, 0, 1, 2)
    }
    velocity = (-values[2] + 8 * values[1] - 8 * values[-1] + values[-2]) / (12 * step)
    acceleration = (
        -values[2] + 16 * values[1] - 30 * values[0] + 16 * values[-1] - values[-2]
    ) / (12 * step**2)
    np.testing.assert_allclose(result.displacement, values[0], atol=1e-12)
    np.testing.assert_allclose(result.velocity, velocity, atol=2e-10)
    np.testing.assert_allclose(result.acceleration, acceleration, atol=3e-7)


def test_dual_wrenches_close_world_force_moment_and_power() -> None:
    root, anchor = _motions()
    result = moving_grip_kinematics(root, anchor)
    generalized = np.array([4.0, -2, 3, 0.2, -0.4, 0.5])
    root_wrench = -result.root_motion_map.T @ generalized
    anchor_wrench = -result.anchor_motion_map.T @ generalized
    root_pose, anchor_pose = np.asarray(root.pose), np.asarray(anchor.pose)
    force = root_pose[:3, :3] @ root_wrench[:3]
    np.testing.assert_allclose(
        force + anchor_pose[:3, :3] @ anchor_wrench[:3], 0, atol=1e-12
    )
    moment = (
        root_pose[:3, :3] @ root_wrench[3:]
        + np.cross(root_pose[:3, 3] - anchor_pose[:3, 3], force)
        + anchor_pose[:3, :3] @ anchor_wrench[3:]
    )
    np.testing.assert_allclose(moment, 0, atol=1e-12)
    power = root_wrench @ root.twist + anchor_wrench @ anchor.twist
    assert power == pytest.approx(-generalized @ result.velocity, abs=1e-12)


def test_fixed_observer_change_preserves_all_relative_outputs() -> None:
    root, anchor = _motions()
    transform = _pose((2, -1, 3), (0.7, -0.4, 0.2))
    original = moving_grip_kinematics(root, anchor)
    changed = moving_grip_kinematics(
        replace(root, pose=transform @ root.pose, observer_id="other"),
        replace(anchor, pose=transform @ anchor.pose, observer_id="other"),
    )
    for name in (
        "displacement",
        "velocity",
        "acceleration",
        "root_motion_map",
        "anchor_motion_map",
    ):
        np.testing.assert_allclose(
            getattr(changed, name), getattr(original, name), atol=2e-12
        )


def test_rigid_common_motion_has_no_relative_rate_or_acceleration() -> None:
    _, anchor = _motions()
    relative = _pose((0.3, -0.2, 0.4), (0.2, 0.1, -0.3))
    rotation, offset = relative[:3, :3], relative[:3, 3]

    def transported(value: tuple[float, ...]) -> np.ndarray:
        vector = np.asarray(value)
        return np.r_[
            rotation.T @ (vector[:3] + np.cross(vector[3:], offset)),
            rotation.T @ vector[3:],
        ]

    root = MaterialPointMotion(
        np.asarray(anchor.pose) @ relative,
        transported(anchor.twist),
        transported(anchor.twist_rate),
        "world",
    )
    result = moving_grip_kinematics(root, anchor)
    np.testing.assert_allclose(result.velocity, 0, atol=2e-12)
    np.testing.assert_allclose(result.acceleration, 0, atol=2e-12)


def test_zero_angle_is_regular_but_principal_branch_boundary_is_refused() -> None:
    zero = MaterialPointMotion(np.eye(4), np.zeros(6), np.zeros(6), "world")
    result = moving_grip_kinematics(zero, zero)
    np.testing.assert_array_equal(result.root_motion_map, np.eye(6))
    np.testing.assert_array_equal(result.anchor_motion_map, -np.eye(6))
    rotated = replace(zero, pose=_pose((0, 0, 0), (np.pi, 0, 0)))
    with pytest.raises(ValueError, match="branch"):
        moving_grip_kinematics(rotated, zero)


def test_mismatched_observers_and_malformed_state_are_refused() -> None:
    root, anchor = _motions()
    with pytest.raises(ValueError, match="observer"):
        moving_grip_kinematics(root, replace(anchor, observer_id="other"))
    with pytest.raises((TypeError, ValueError)):
        replace(root, twist=[True, 0, 0, 0, 0, 0])
    with pytest.raises((TypeError, ValueError)):
        replace(root, pose=np.zeros((4, 4)))


def test_motion_records_own_pose_and_rate_values() -> None:
    pose, twist, rate = np.eye(4), np.arange(6.0), np.arange(6.0) / 10
    motion = MaterialPointMotion(pose, twist, rate, "world")
    pose[0, 3], twist[0], rate[0] = 99, 99, 99
    np.testing.assert_array_equal(motion.pose, np.eye(4))
    np.testing.assert_array_equal(motion.twist, np.arange(6.0))
    np.testing.assert_array_equal(motion.twist_rate, np.arange(6.0) / 10)


def test_unknown_state_type_and_arithmetic_overflow_are_refused() -> None:
    root, anchor = _motions()
    with pytest.raises(TypeError, match="MaterialPointMotion"):
        moving_grip_kinematics(object(), anchor)  # type: ignore[arg-type]
    root = replace(root, pose=_pose((1e308, 0, 0), (0, 0, 0)))
    anchor = replace(anchor, pose=_pose((-1e308, 0, 0), (0, 0, 0)))
    with pytest.raises(ValueError, match="numerical"):
        moving_grip_kinematics(root, anchor)
