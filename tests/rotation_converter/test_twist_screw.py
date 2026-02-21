"""TDD tests for twist and screw axis conversion module.

Representations covered:
- Twist (6-vector: [omega; v] or se(3) 4x4 matrix)
- Screw axis (axis direction, point on axis, pitch, theta)
- Homogeneous transformation matrix SE(3)

Written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter.twist_screw import (
    adjoint_representation,
    homogeneous_to_twist_angle,
    screw_to_twist,
    se3_matrix_to_twist_vector,
    twist_angle_to_homogeneous,
    twist_to_screw,
    twist_vector_to_se3_matrix,
)

ATOL = 1e-10


# ===========================================================================
# Twist vector <-> se(3) matrix
# ===========================================================================


class TestTwistMatrixConversion:
    """Conversion between 6-vector twist and 4x4 se(3) matrix."""

    def test_zero_twist_to_zero_matrix(self) -> None:
        xi = np.zeros(6)
        M = twist_vector_to_se3_matrix(xi)
        np.testing.assert_allclose(M, np.zeros((4, 4)), atol=ATOL)

    def test_pure_rotation_twist(self) -> None:
        """Twist with omega=[0,0,1], v=[0,0,0] -> rotation about z."""
        xi = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        M = twist_vector_to_se3_matrix(xi)
        expected = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(M, expected, atol=ATOL)

    def test_pure_translation_twist(self) -> None:
        """Twist with omega=[0,0,0], v=[1,0,0] -> translation along x."""
        xi = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        M = twist_vector_to_se3_matrix(xi)
        expected = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(M, expected, atol=ATOL)

    def test_se3_matrix_roundtrip(self) -> None:
        xi = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        M = twist_vector_to_se3_matrix(xi)
        xi2 = se3_matrix_to_twist_vector(M)
        np.testing.assert_allclose(xi2, xi, atol=ATOL)

    def test_invalid_matrix_shape_raises(self) -> None:
        with pytest.raises(Exception):
            se3_matrix_to_twist_vector(np.zeros((3, 3)))

    def test_invalid_twist_length_raises(self) -> None:
        with pytest.raises(Exception):
            twist_vector_to_se3_matrix(np.zeros(5))


# ===========================================================================
# Twist + angle -> SE(3) (matrix exponential)
# ===========================================================================


class TestTwistToHomogeneous:
    """Twist + angle -> homogeneous transformation via matrix exponential."""

    def test_zero_angle_gives_identity(self) -> None:
        xi = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        T = twist_angle_to_homogeneous(xi, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=ATOL)

    def test_pure_rotation_90deg_z(self) -> None:
        xi = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        T = twist_angle_to_homogeneous(xi, math.pi / 2)
        expected_R = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(T[:3, :3], expected_R, atol=ATOL)
        np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_pure_translation(self) -> None:
        xi = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        T = twist_angle_to_homogeneous(xi, 3.0)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=ATOL)
        np.testing.assert_allclose(T[:3, 3], [3.0, 0, 0], atol=ATOL)

    def test_result_is_SE3(self) -> None:
        """Rotation part should be SO(3), bottom row [0,0,0,1]."""
        xi = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        T = twist_angle_to_homogeneous(xi, 1.5)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=ATOL)

    def test_homogeneous_roundtrip(self) -> None:
        xi = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 2.0])
        theta = 0.8
        T = twist_angle_to_homogeneous(xi, theta)
        xi2, theta2 = homogeneous_to_twist_angle(T)
        T2 = twist_angle_to_homogeneous(xi2, theta2)
        np.testing.assert_allclose(T2, T, atol=ATOL)


# ===========================================================================
# Twist <-> Screw axis
# ===========================================================================


class TestTwistScrewConversion:
    """Conversions between twist vectors and screw axis parameters."""

    def test_pure_rotation_screw_has_zero_pitch(self) -> None:
        """Pure rotation twist -> screw with pitch = 0."""
        xi = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        screw = twist_to_screw(xi)
        assert abs(screw["pitch"]) < ATOL
        np.testing.assert_allclose(screw["axis"], [0, 0, 1], atol=ATOL)

    def test_pure_translation_screw_has_infinite_pitch(self) -> None:
        """Pure translation twist -> screw with pitch = inf."""
        xi = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        screw = twist_to_screw(xi)
        assert screw["pitch"] == float("inf")
        np.testing.assert_allclose(screw["axis"], [1, 0, 0], atol=ATOL)

    def test_general_screw_roundtrip(self) -> None:
        """General twist (rotation + translation along axis) roundtrip."""
        # Screw: rotate about z at origin with pitch 2.0
        omega = np.array([0.0, 0.0, 1.0])
        v = np.array([0.0, 0.0, 2.0])  # pitch * omega = [0,0,2]
        xi = np.concatenate([omega, v])
        screw = twist_to_screw(xi)
        xi2 = screw_to_twist(screw)
        np.testing.assert_allclose(xi2, xi, atol=ATOL)

    def test_screw_to_twist_pure_rotation(self) -> None:
        screw = {
            "axis": np.array([1.0, 0.0, 0.0]),
            "point": np.array([0.0, 0.0, 0.0]),
            "pitch": 0.0,
        }
        xi = screw_to_twist(screw)
        np.testing.assert_allclose(xi[:3], [1, 0, 0], atol=ATOL)
        np.testing.assert_allclose(xi[3:], [0, 0, 0], atol=ATOL)

    def test_screw_to_twist_with_offset_point(self) -> None:
        """Rotation about z-axis through point (1,0,0) with zero pitch."""
        screw = {
            "axis": np.array([0.0, 0.0, 1.0]),
            "point": np.array([1.0, 0.0, 0.0]),
            "pitch": 0.0,
        }
        xi = screw_to_twist(screw)
        np.testing.assert_allclose(xi[:3], [0, 0, 1], atol=ATOL)
        # v = -omega x point = -[0,0,1] x [1,0,0] = -[0,1,0] = [0,-1,0]
        # Actually v = point x omega for screw convention
        # v = -omega x q + h * omega where q is point on axis, h is pitch
        expected_v = np.cross(-np.array([0, 0, 1]), np.array([1, 0, 0]))
        np.testing.assert_allclose(xi[3:], expected_v, atol=ATOL)


# ===========================================================================
# SE(3) -> twist+angle decomposition
# ===========================================================================


class TestHomogeneousToTwist:
    """Decompose SE(3) matrix into twist + angle."""

    def test_identity_gives_zero_twist(self) -> None:
        T = np.eye(4)
        xi, theta = homogeneous_to_twist_angle(T)
        assert abs(theta) < ATOL

    def test_pure_rotation_decomposition(self) -> None:
        T = np.eye(4)
        T[:3, :3] = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        xi, theta = homogeneous_to_twist_angle(T)
        assert abs(theta - math.pi / 2) < ATOL
        omega = xi[:3]
        np.testing.assert_allclose(omega, [0, 0, 1], atol=ATOL)

    def test_pure_translation_decomposition(self) -> None:
        T = np.eye(4)
        T[:3, 3] = [5.0, 0.0, 0.0]
        xi, theta = homogeneous_to_twist_angle(T)
        # For pure translation: xi = [0,0,0, v_hat], theta = ||p||
        assert abs(theta - 5.0) < ATOL
        np.testing.assert_allclose(xi[:3], [0, 0, 0], atol=ATOL)


# ===========================================================================
# Adjoint representation
# ===========================================================================


class TestAdjointRepresentation:
    """Tests for the 6x6 adjoint matrix of SE(3)."""

    def test_identity_adjoint_is_identity(self) -> None:
        T = np.eye(4)
        Ad = adjoint_representation(T)
        np.testing.assert_allclose(Ad, np.eye(6), atol=ATOL)

    def test_adjoint_shape(self) -> None:
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        Ad = adjoint_representation(T)
        assert Ad.shape == (6, 6)

    def test_adjoint_pure_rotation(self) -> None:
        T = np.eye(4)
        R = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        T[:3, :3] = R
        Ad = adjoint_representation(T)
        # Top-left 3x3 should be R
        np.testing.assert_allclose(Ad[:3, :3], R, atol=ATOL)
        # Bottom-right 3x3 should be R
        np.testing.assert_allclose(Ad[3:, 3:], R, atol=ATOL)
        # Top-right should be zero (no translation)
        np.testing.assert_allclose(Ad[:3, 3:], np.zeros((3, 3)), atol=ATOL)


# ===========================================================================
# Random round-trip stress tests
# ===========================================================================


class TestRandomTwistScrewRoundTrips:
    """Randomised round-trip property tests for twist/screw conversions."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=123)

    @pytest.mark.parametrize("trial", range(30))
    def test_twist_homogeneous_roundtrip_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        omega = rng.normal(size=3)
        omega /= np.linalg.norm(omega)
        v = rng.normal(size=3)
        xi = np.concatenate([omega, v])
        theta = rng.uniform(0.1, 2.0)
        T = twist_angle_to_homogeneous(xi, theta)
        xi2, theta2 = homogeneous_to_twist_angle(T)
        T2 = twist_angle_to_homogeneous(xi2, theta2)
        np.testing.assert_allclose(T2, T, atol=1e-9)

    @pytest.mark.parametrize("trial", range(30))
    def test_twist_screw_roundtrip_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        omega = rng.normal(size=3)
        omega /= np.linalg.norm(omega)
        v = rng.normal(size=3)
        xi = np.concatenate([omega, v])
        screw = twist_to_screw(xi)
        xi2 = screw_to_twist(screw)
        np.testing.assert_allclose(xi2, xi, atol=1e-9)
