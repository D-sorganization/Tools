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

from rotation_converter._contracts import PreconditionError
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
        with pytest.raises(PreconditionError):
            se3_matrix_to_twist_vector(np.zeros((3, 3)))

    def test_invalid_twist_length_raises(self) -> None:
        with pytest.raises(PreconditionError):
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


# ===========================================================================
# Input validation on screw_to_twist
# ===========================================================================


class TestScrewToTwistValidation:
    """screw_to_twist should reject non-unit axis for finite pitch."""

    def test_non_unit_axis_finite_pitch_raises(self) -> None:
        screw = {
            "axis": np.array([2.0, 0.0, 0.0]),  # NOT unit
            "point": np.zeros(3),
            "pitch": 0.0,
        }
        with pytest.raises(PreconditionError):
            screw_to_twist(screw)

    def test_unit_axis_finite_pitch_ok(self) -> None:
        screw = {
            "axis": np.array([1.0, 0.0, 0.0]),
            "point": np.zeros(3),
            "pitch": 0.0,
        }
        xi = screw_to_twist(screw)
        assert xi.shape == (6,)

    def test_non_unit_axis_infinite_pitch_normalized(self) -> None:
        """Infinite pitch normalizes axis automatically."""
        screw = {
            "axis": np.array([3.0, 0.0, 0.0]),
            "point": np.zeros(3),
            "pitch": float("inf"),
        }
        xi = screw_to_twist(screw)
        np.testing.assert_allclose(xi[3:], [1, 0, 0], atol=ATOL)

    def test_zero_axis_infinite_pitch_raises(self) -> None:
        """Zero axis with infinite pitch should raise."""
        screw = {
            "axis": np.zeros(3),
            "point": np.zeros(3),
            "pitch": float("inf"),
        }
        with pytest.raises(PreconditionError):
            screw_to_twist(screw)


# ===========================================================================
# Near-pi rotation and twist contract edge cases
# ===========================================================================


class TestTwistScrewEdgeCases:
    """Edge cases for twist/screw conversions."""

    def test_near_pi_rotation_homogeneous_roundtrip(self) -> None:
        """180-degree rotation should roundtrip through twist decomposition."""
        omega = np.array([0.0, 0.0, 1.0])
        theta = math.pi - 1e-10  # just under pi
        xi = np.concatenate([omega, np.array([1.0, 0.0, 0.0])])
        T = twist_angle_to_homogeneous(xi, theta)
        xi2, theta2 = homogeneous_to_twist_angle(T)
        T2 = twist_angle_to_homogeneous(xi2, theta2)
        np.testing.assert_allclose(T2, T, atol=1e-7)

    def test_zero_twist_to_screw_raises(self) -> None:
        """All-zero twist should raise PreconditionError."""
        with pytest.raises(PreconditionError):
            twist_to_screw(np.zeros(6))

    def test_non_unit_omega_twist_to_homogeneous_raises(self) -> None:
        """Non-unit omega should raise PreconditionError."""
        xi = np.array([2.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        with pytest.raises(PreconditionError):
            twist_angle_to_homogeneous(xi, 1.0)

    def test_bad_bottom_row_homogeneous_raises(self) -> None:
        """Non-SE(3) bottom row should raise PreconditionError."""
        T = np.eye(4)
        T[3, 0] = 1.0
        with pytest.raises(PreconditionError):
            homogeneous_to_twist_angle(T)

    def test_bad_bottom_row_se3_raises(self) -> None:
        """se(3) matrix with non-zero bottom row should raise."""
        M = np.zeros((4, 4))
        M[3, 0] = 1.0
        with pytest.raises(PreconditionError):
            se3_matrix_to_twist_vector(M)

    def test_pure_translation_homogeneous_roundtrip(self) -> None:
        """Pure translation SE(3) matrix roundtrip through twist."""
        T = np.eye(4)
        T[:3, 3] = [3.0, 4.0, 0.0]
        xi, theta = homogeneous_to_twist_angle(T)
        T2 = twist_angle_to_homogeneous(xi, theta)
        np.testing.assert_allclose(T2, T, atol=1e-9)

    def test_adjoint_property(self) -> None:
        """Adjoint should satisfy Ad_T * Vb = T * [Vb] * T^-1 (twist mapping)."""
        rng = np.random.default_rng(42)
        omega = rng.normal(size=3)
        omega /= np.linalg.norm(omega)
        xi = np.concatenate([omega, rng.normal(size=3)])
        T = twist_angle_to_homogeneous(xi, 0.8)
        Ad = adjoint_representation(T)

        # Body twist
        Vb = np.concatenate([omega, rng.normal(size=3)])
        Vs = Ad @ Vb

        # Verify via matrix form: [Vs] = T @ [Vb] @ T^-1
        from rotation_converter.twist_screw import (
            se3_matrix_to_twist_vector,
            twist_vector_to_se3_matrix,
        )

        Vb_mat = twist_vector_to_se3_matrix(Vb)
        T_inv = np.linalg.inv(T)
        Vs_mat = T @ Vb_mat @ T_inv
        Vs_check = se3_matrix_to_twist_vector(Vs_mat)
        np.testing.assert_allclose(Vs, Vs_check, atol=1e-9)
