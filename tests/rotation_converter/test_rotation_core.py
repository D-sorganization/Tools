"""TDD tests for the rotation converter core module.

Tests cover all rotation representations and round-trip conversions:
- Quaternion (w, x, y, z)
- Rotation matrix (3x3 SO(3))
- Euler angles (multiple conventions)
- Axis-angle (unit axis + angle)
- Rodrigues vector (axis * angle)
- Rotation vector (same as Rodrigues, explicit alias)

Each test is written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter._contracts import PreconditionError
from rotation_converter.core import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_euler,
    quaternion_to_rodrigues,
    quaternion_to_rotation_matrix,
    rodrigues_to_quaternion,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
ATOL = 1e-10


# ===========================================================================
# Quaternion basics
# ===========================================================================


class TestQuaternionBasics:
    """Quaternion normalisation, conjugate, multiply."""

    def test_normalize_unit_quaternion_unchanged(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        result = normalize_quaternion(q)
        np.testing.assert_allclose(result, q, atol=ATOL)

    def test_normalize_scales_to_unit(self) -> None:
        q = np.array([2.0, 0.0, 0.0, 0.0])
        result = normalize_quaternion(q)
        assert abs(np.linalg.norm(result) - 1.0) < ATOL

    def test_normalize_zero_quaternion_raises(self) -> None:
        with pytest.raises(PreconditionError):
            normalize_quaternion(np.array([0.0, 0.0, 0.0, 0.0]))

    def test_conjugate(self) -> None:
        q = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        np.testing.assert_allclose(quaternion_conjugate(q), expected, atol=ATOL)

    def test_multiply_identity(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = np.array([0.7071, 0.7071, 0.0, 0.0])
        result = quaternion_multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-4)

    def test_multiply_inverse_gives_identity(self) -> None:
        q = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        np.testing.assert_allclose(result, [1.0, 0.0, 0.0, 0.0], atol=ATOL)


# ===========================================================================
# Quaternion <-> Rotation Matrix
# ===========================================================================


class TestQuaternionRotationMatrix:
    """Round-trip and known-value tests for quaternion <-> rotation matrix."""

    def test_identity_quaternion_to_matrix(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_90deg_x_rotation(self) -> None:
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_180deg_z_rotation(self) -> None:
        q = np.array([0.0, 0.0, 0.0, 1.0])
        R = quaternion_to_rotation_matrix(q)
        expected = np.array(
            [
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_rotation_matrix_to_quaternion_identity(self) -> None:
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)
        # Either [1,0,0,0] or [-1,0,0,0] are valid
        assert abs(abs(q[0]) - 1.0) < ATOL
        np.testing.assert_allclose(q[1:], [0, 0, 0], atol=ATOL)

    def test_quaternion_rotation_matrix_roundtrip(self) -> None:
        q_orig = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        R = quaternion_to_rotation_matrix(q_orig)
        q_back = rotation_matrix_to_quaternion(R)
        # Quaternions q and -q represent same rotation
        if np.dot(q_orig, q_back) < 0:
            q_back = -q_back
        np.testing.assert_allclose(q_back, q_orig, atol=ATOL)

    def test_rotation_matrix_is_SO3(self) -> None:
        q = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        R = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL

    def test_non_unit_quaternion_raises(self) -> None:
        with pytest.raises(PreconditionError):
            quaternion_to_rotation_matrix(np.array([2.0, 0.0, 0.0, 0.0]))


# ===========================================================================
# Quaternion <-> Euler angles
# ===========================================================================


class TestQuaternionEuler:
    """Round-trip and known-value tests for quaternion <-> Euler angles."""

    @pytest.mark.parametrize("convention", ["xyz", "zyx", "zyz", "xyx"])
    def test_zero_euler_gives_identity_quaternion(self, convention: str) -> None:
        q = euler_to_quaternion(0.0, 0.0, 0.0, convention)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    @pytest.mark.parametrize("convention", ["xyz", "zyx", "zyz", "xyx"])
    def test_euler_quaternion_roundtrip(self, convention: str) -> None:
        a, b, c = 0.3, 0.5, -0.7
        q = euler_to_quaternion(a, b, c, convention)
        a2, b2, c2 = quaternion_to_euler(q, convention)
        q2 = euler_to_quaternion(a2, b2, c2, convention)
        # Compare via quaternion since Euler has multiple representations
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=ATOL)

    def test_euler_xyz_known_value(self) -> None:
        """90-degree rotation about X via XYZ Euler."""
        q = euler_to_quaternion(math.pi / 2, 0.0, 0.0, "xyz")
        R = quaternion_to_rotation_matrix(q)
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_euler_zyx_known_value(self) -> None:
        """90-degree rotation about Z via ZYX Euler (yaw-pitch-roll)."""
        q = euler_to_quaternion(math.pi / 2, 0.0, 0.0, "zyx")
        R = quaternion_to_rotation_matrix(q)
        expected = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_invalid_convention_raises(self) -> None:
        with pytest.raises(PreconditionError):
            euler_to_quaternion(0.0, 0.0, 0.0, "abc")


# ===========================================================================
# Quaternion <-> Axis-Angle
# ===========================================================================


class TestQuaternionAxisAngle:
    """Round-trip and known-value tests for quaternion <-> axis-angle."""

    def test_zero_angle_gives_identity(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        q = axis_angle_to_quaternion(axis, 0.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=ATOL)

    def test_identity_quaternion_to_axis_angle(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        axis, angle = quaternion_to_axis_angle(q)
        assert abs(angle) < ATOL

    def test_90deg_x_axis_angle(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        angle = math.pi / 2
        q = axis_angle_to_quaternion(axis, angle)
        expected_q = np.array([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
        np.testing.assert_allclose(q, expected_q, atol=ATOL)

    def test_axis_angle_quaternion_roundtrip(self) -> None:
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
        angle = 1.23
        q = axis_angle_to_quaternion(axis, angle)
        axis2, angle2 = quaternion_to_axis_angle(q)
        # Reconstruct and compare
        q2 = axis_angle_to_quaternion(axis2, angle2)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=ATOL)

    def test_non_unit_axis_raises(self) -> None:
        with pytest.raises(PreconditionError):
            axis_angle_to_quaternion(np.array([2.0, 0.0, 0.0]), math.pi / 4)

    def test_180deg_rotation(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        angle = math.pi
        q = axis_angle_to_quaternion(axis, angle)
        axis2, angle2 = quaternion_to_axis_angle(q)
        assert abs(angle2 - math.pi) < ATOL
        np.testing.assert_allclose(axis2, axis, atol=ATOL)


# ===========================================================================
# Axis-Angle <-> Rotation Matrix
# ===========================================================================


class TestAxisAngleRotationMatrix:
    """Direct axis-angle <-> rotation matrix (Rodrigues formula)."""

    def test_zero_angle_gives_identity(self) -> None:
        axis = np.array([1.0, 0.0, 0.0])
        R = axis_angle_to_rotation_matrix(axis, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_90deg_z_rotation(self) -> None:
        axis = np.array([0.0, 0.0, 1.0])
        R = axis_angle_to_rotation_matrix(axis, math.pi / 2)
        expected = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(R, expected, atol=ATOL)

    def test_rotation_matrix_to_axis_angle_roundtrip(self) -> None:
        axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        angle = 0.87
        R = axis_angle_to_rotation_matrix(axis, angle)
        axis2, angle2 = rotation_matrix_to_axis_angle(R)
        R2 = axis_angle_to_rotation_matrix(axis2, angle2)
        np.testing.assert_allclose(R2, R, atol=ATOL)


# ===========================================================================
# Quaternion <-> Rodrigues vector
# ===========================================================================


class TestQuaternionRodrigues:
    """Round-trip and known-value tests for quaternion <-> Rodrigues vector."""

    def test_identity_quaternion_to_rodrigues(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        r = quaternion_to_rodrigues(q)
        np.testing.assert_allclose(r, [0, 0, 0], atol=ATOL)

    def test_rodrigues_to_quaternion_zero(self) -> None:
        r = np.array([0.0, 0.0, 0.0])
        q = rodrigues_to_quaternion(r)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=ATOL)

    def test_rodrigues_roundtrip(self) -> None:
        q_orig = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        r = quaternion_to_rodrigues(q_orig)
        q_back = rodrigues_to_quaternion(r)
        if np.dot(q_orig, q_back) < 0:
            q_back = -q_back
        np.testing.assert_allclose(q_back, q_orig, atol=ATOL)

    def test_rodrigues_magnitude_is_angle(self) -> None:
        axis = np.array([0.0, 1.0, 0.0])
        angle = 1.5
        q = axis_angle_to_quaternion(axis, angle)
        r = quaternion_to_rodrigues(q)
        assert abs(np.linalg.norm(r) - angle) < ATOL


# ===========================================================================
# Euler <-> Rotation Matrix (via quaternion hub)
# ===========================================================================


class TestEulerRotationMatrix:
    """Euler angles <-> rotation matrix conversions."""

    @pytest.mark.parametrize("convention", ["xyz", "zyx", "zyz"])
    def test_euler_to_matrix_roundtrip(self, convention: str) -> None:
        a, b, c = 0.4, -0.2, 0.6
        R = euler_to_rotation_matrix(a, b, c, convention)
        a2, b2, c2 = rotation_matrix_to_euler(R, convention)
        R2 = euler_to_rotation_matrix(a2, b2, c2, convention)
        np.testing.assert_allclose(R2, R, atol=ATOL)

    def test_euler_matrix_is_SO3(self) -> None:
        R = euler_to_rotation_matrix(0.1, 0.2, 0.3, "xyz")
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert abs(np.linalg.det(R) - 1.0) < ATOL


# ===========================================================================
# Random round-trip stress tests
# ===========================================================================


class TestRandomRoundTrips:
    """Randomised round-trip property tests."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(seed=42)

    def _random_quaternion(self, rng: np.random.Generator) -> np.ndarray:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        if q[0] < 0:
            q = -q
        return q

    @pytest.mark.parametrize("trial", range(50))
    def test_quat_matrix_roundtrip_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        q = self._random_quaternion(rng)
        R = quaternion_to_rotation_matrix(q)
        q2 = rotation_matrix_to_quaternion(R)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=1e-9)

    @pytest.mark.parametrize("trial", range(50))
    def test_quat_axis_angle_roundtrip_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        q = self._random_quaternion(rng)
        axis, angle = quaternion_to_axis_angle(q)
        q2 = axis_angle_to_quaternion(axis, angle)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=1e-9)

    @pytest.mark.parametrize("trial", range(50))
    def test_quat_rodrigues_roundtrip_random(
        self, rng: np.random.Generator, trial: int
    ) -> None:
        q = self._random_quaternion(rng)
        r = quaternion_to_rodrigues(q)
        q2 = rodrigues_to_quaternion(r)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=1e-9)

    @pytest.mark.parametrize(
        "convention", ["xyz", "zyx", "zyz", "xyx", "yxy", "yzy", "xzx", "zxz"]
    )
    def test_euler_roundtrip_random(
        self, rng: np.random.Generator, convention: str
    ) -> None:
        q = self._random_quaternion(rng)
        e = quaternion_to_euler(q, convention)
        q2 = euler_to_quaternion(*e, convention)
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q2, q, atol=1e-9)


# ===========================================================================
# Gimbal lock edge cases
# ===========================================================================


class TestGimbalLock:
    """Exercise gimbal lock code paths in Euler extraction."""

    @pytest.mark.parametrize("convention", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    def test_tait_bryan_gimbal_lock_positive(self, convention: str) -> None:
        """Tait-Bryan at b = +pi/2 (cos(b) ~ 0)."""
        a, c = 0.3, 0.0  # c is degenerate at gimbal lock
        b = math.pi / 2.0
        R = euler_to_rotation_matrix(a, b, c, convention)
        a2, b2, c2 = rotation_matrix_to_euler(R, convention)
        R2 = euler_to_rotation_matrix(a2, b2, c2, convention)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    @pytest.mark.parametrize("convention", ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"])
    def test_tait_bryan_gimbal_lock_negative(self, convention: str) -> None:
        """Tait-Bryan at b = -pi/2."""
        a, c = -0.5, 0.0
        b = -math.pi / 2.0
        R = euler_to_rotation_matrix(a, b, c, convention)
        a2, b2, c2 = rotation_matrix_to_euler(R, convention)
        R2 = euler_to_rotation_matrix(a2, b2, c2, convention)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    @pytest.mark.parametrize("convention", ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"])
    def test_proper_euler_gimbal_lock_zero(self, convention: str) -> None:
        """Proper Euler at b = 0 (gimbal lock)."""
        a, c = 0.7, 0.0
        b = 0.0
        R = euler_to_rotation_matrix(a, b, c, convention)
        a2, b2, c2 = rotation_matrix_to_euler(R, convention)
        R2 = euler_to_rotation_matrix(a2, b2, c2, convention)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    @pytest.mark.parametrize("convention", ["xyx", "xzx", "yxy", "yzy", "zxz", "zyz"])
    def test_proper_euler_gimbal_lock_pi(self, convention: str) -> None:
        """Proper Euler at b = pi (gimbal lock)."""
        a, c = -0.2, 0.0
        b = math.pi
        R = euler_to_rotation_matrix(a, b, c, convention)
        a2, b2, c2 = rotation_matrix_to_euler(R, convention)
        R2 = euler_to_rotation_matrix(a2, b2, c2, convention)
        np.testing.assert_allclose(R2, R, atol=1e-9)


# ===========================================================================
# Contract type specificity and NaN/Inf edge cases
# ===========================================================================


class TestContractTypes:
    """Tests that DbC contracts raise PreconditionError, not generic Exception."""

    def test_normalize_zero_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            normalize_quaternion([0, 0, 0, 0])

    def test_non_unit_quaternion_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            quaternion_to_rotation_matrix([2, 0, 0, 0])

    def test_non_unit_axis_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            axis_angle_to_quaternion([2, 0, 0], 1.0)

    def test_invalid_convention_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            euler_to_quaternion(0, 0, 0, "abc")

    def test_nan_quaternion_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            quaternion_to_rotation_matrix([float("nan"), 0, 0, 0])

    def test_inf_rotation_matrix_raises_precondition(self) -> None:
        R = np.eye(3)
        R[0, 0] = float("inf")
        with pytest.raises(PreconditionError):
            rotation_matrix_to_quaternion(R)

    def test_nan_axis_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            axis_angle_to_quaternion([float("nan"), 0, 0], 1.0)

    def test_nan_rodrigues_raises_precondition(self) -> None:
        with pytest.raises(PreconditionError):
            rodrigues_to_quaternion([float("inf"), 0, 0])


# ===========================================================================
# Shepperd branch coverage for 180-degree rotations
# ===========================================================================


class TestShepperdBranches:
    """Ensure rotation_matrix_to_quaternion covers all Shepperd branches."""

    def test_180_deg_about_x(self) -> None:
        R = axis_angle_to_rotation_matrix([1, 0, 0], math.pi)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    def test_180_deg_about_y(self) -> None:
        R = axis_angle_to_rotation_matrix([0, 1, 0], math.pi)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    def test_180_deg_about_z(self) -> None:
        R = axis_angle_to_rotation_matrix([0, 0, 1], math.pi)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R2, R, atol=1e-9)

    def test_180_deg_about_diagonal(self) -> None:
        axis = np.array([1, 1, 1]) / math.sqrt(3)
        R = axis_angle_to_rotation_matrix(axis, math.pi)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R2, R, atol=1e-9)
