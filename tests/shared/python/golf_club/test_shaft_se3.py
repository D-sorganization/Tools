"""Independent kinematic oracles for the loaded-shaft section chart."""

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from shared.python.golf_club._shaft_se3 import exp_twist, interpolate_pose, log_pose


def _pose(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    return pose


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-7, 0.5, 2.8])
def test_linear_first_twist_round_trip_preserves_small_rotations(angle: float) -> None:
    twist = np.array([0.2, -0.3, 0.7, angle, 0.0, 0.0])
    actual = log_pose(exp_twist(twist))
    np.testing.assert_allclose(actual[:3], twist[:3], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(actual[3:], twist[3:], rtol=1e-13, atol=1e-27)


def test_screw_motion_matches_independent_four_matrix_exponential() -> None:
    twist = np.array([0.3, -0.7, 0.8, -0.4, 0.7, 0.2])
    generator = np.zeros((4, 4))
    x, y, z = twist[3:]
    generator[:3, :3] = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    generator[:3, 3] = twist[:3]
    np.testing.assert_allclose(exp_twist(twist), expm(generator), atol=2e-15)


@pytest.mark.parametrize("fraction", [0.0, 0.17, 0.5, 1.0])
def test_pure_bending_is_exact_circular_arc_under_rigid_motion(fraction: float) -> None:
    length, curvature = 0.9, 0.8
    angle = curvature * length
    right = _pose(
        Rotation.from_rotvec([0, angle, 0]).as_matrix(),
        np.array([(1 - np.cos(angle)) / curvature, 0, np.sin(angle) / curvature]),
    )
    world = _pose(
        Rotation.from_rotvec([0.3, -0.4, 0.2]).as_matrix(),
        np.array([1.7, -2.2, 0.8]),
    )
    partial_angle = fraction * angle
    expected = _pose(
        Rotation.from_rotvec([0, partial_angle, 0]).as_matrix(),
        np.array(
            [
                (1 - np.cos(partial_angle)) / curvature,
                0,
                np.sin(partial_angle) / curvature,
            ]
        ),
    )
    actual = interpolate_pose(world, world @ right, fraction)
    np.testing.assert_allclose(actual, world @ expected, atol=2e-15)
    # A straight unstretched +z reference has no shear or axial strain here.
    np.testing.assert_allclose(
        log_pose(right) / length, [0, 0, 1, 0, curvature, 0], atol=1e-14
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.inf, np.nan, True, "0.5"])
def test_interpolation_refuses_invalid_material_fraction(fraction: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        interpolate_pose(np.eye(4), np.eye(4), fraction)


@pytest.mark.parametrize("angle", [np.pi, np.pi - 1e-8])
def test_relative_rotation_branch_boundary_is_explicit(angle: float) -> None:
    pose = _pose(Rotation.from_rotvec([angle, 0, 0]).as_matrix(), np.zeros(3))
    with pytest.raises(ValueError, match="branch"):
        log_pose(pose)


@pytest.mark.parametrize("kind", ["reflection", "scale", "bottom-row", "nonfinite"])
def test_pose_contract_rejects_non_rigid_transforms(kind: str) -> None:
    pose = np.eye(4)
    if kind == "reflection":
        pose[0, 0] = -1
    elif kind == "scale":
        pose[0, 0] = 1.01
    elif kind == "bottom-row":
        pose[3, 0] = 0.01
    else:
        pose[0, 3] = np.nan
    with pytest.raises(ValueError):
        log_pose(pose)


@pytest.mark.parametrize("value", [[True] * 6, ["0"] * 6, [1j] * 6, [0] * 5])
def test_twist_contract_refuses_coercion_and_wrong_shape(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        exp_twist(value)
