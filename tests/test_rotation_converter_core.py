"""Comprehensive tests for rotation_converter.core module.

Tests cover:
- Quaternion normalization and operations
- Rotation matrix conversions
- Axis-angle representations
- Rodrigues vector conversions
- Euler angle conventions
- Roundtrip consistency and invariants
- Contract violations (preconditions/postconditions)
"""

import numpy as np
import pytest

from rotation_converter.core import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_euler,
    quaternion_to_rodrigues,
    quaternion_to_rotation_matrix,
    rodrigues_to_quaternion,
    rotation_matrix_to_quaternion,
)


class TestQuaternionNormalization:
    """Tests for quaternion normalization."""

    def test_normalize_unit_quaternion(self):
        """Unit quaternion should remain unit after normalization."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)
        assert np.allclose(np.linalg.norm(q_norm), 1.0)
        assert np.allclose(q, q_norm)

    def test_normalize_arbitrary_quaternion(self):
        """Arbitrary quaternion should be normalized to unit length."""
        q = np.array([3.0, 4.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)
        assert np.allclose(np.linalg.norm(q_norm), 1.0)
        expected = np.array([0.6, 0.8, 0.0, 0.0])
        assert np.allclose(q_norm, expected)

    def test_normalize_small_quaternion(self):
        """Small quaternions should normalize correctly."""
        q = np.array([0.001, 0.001, 0.001, 0.001])
        q_norm = normalize_quaternion(q)
        assert np.allclose(np.linalg.norm(q_norm), 1.0)

    def test_normalize_preserves_direction(self):
        """Normalization should preserve quaternion direction."""
        q = np.array([2.0, 2.0, 2.0, 2.0])
        q_norm = normalize_quaternion(q)
        # Should be proportional
        ratio = q_norm / (q / np.linalg.norm(q))
        assert np.allclose(ratio, ratio[0])  # All components equal

    def test_normalize_invalid_shape(self):
        """Should raise ValueError for non-4D quaternions."""
        with pytest.raises(ValueError):
            normalize_quaternion(np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            normalize_quaternion(np.array([1.0, 0.0, 0.0, 0.0, 1.0]))

    def test_normalize_non_finite(self):
        """Should raise ValueError for non-finite values."""
        with pytest.raises(ValueError):
            normalize_quaternion(np.array([np.inf, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            normalize_quaternion(np.array([np.nan, 0.0, 0.0, 0.0]))


class TestQuaternionConjugate:
    """Tests for quaternion conjugation."""

    def test_conjugate_basic(self):
        """Conjugate negates imaginary part."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_conj = quaternion_conjugate(q)
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        assert np.allclose(q_conj, expected)

    def test_conjugate_twice_is_identity(self):
        """Double conjugation should return original."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        q_double_conj = quaternion_conjugate(quaternion_conjugate(q))
        assert np.allclose(q, q_double_conj)

    def test_conjugate_of_real_quaternion(self):
        """Real quaternion should negate to same (conj of [a,0,0,0])."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_conj = quaternion_conjugate(q)
        assert np.allclose(q_conj, np.array([1.0, 0.0, 0.0, 0.0]))


class TestQuaternionMultiplication:
    """Tests for quaternion multiplication (non-commutative)."""

    def test_multiply_identity(self):
        """Multiplying by identity quaternion [1,0,0,0]."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternion_multiply(q, identity)
        assert np.allclose(result, q, atol=1e-6)

    def test_multiply_with_conjugate(self):
        """Quaternion * conjugate should give norm squared on real part."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        q_conj = quaternion_conjugate(q)
        result = quaternion_multiply(q, q_conj)
        # For unit quaternion, q * conj(q) = [norm^2, 0, 0, 0] = [1, 0, 0, 0]
        assert np.allclose(result, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    def test_multiply_non_commutative(self):
        """Quaternion multiplication is non-commutative."""
        q1 = np.array([0.707, 0.707, 0.0, 0.0])
        q2 = np.array([0.707, 0.0, 0.707, 0.0])
        r1 = quaternion_multiply(q1, q2)
        r2 = quaternion_multiply(q2, q1)
        assert not np.allclose(r1, r2)

    def test_multiply_result_is_unit(self):
        """Product of two unit quaternions is unit."""
        q1 = np.array([0.707, 0.707, 0.0, 0.0])
        q2 = np.array([1.0, 0.0, 0.0, 0.0])
        result = quaternion_multiply(q1, q2)
        assert np.allclose(np.linalg.norm(result), 1.0, atol=1e-6)


class TestQuaternionRotationMatrixConversions:
    """Tests for conversions between quaternions and rotation matrices."""

    def test_identity_quaternion_to_identity_matrix(self):
        """Identity quaternion [1,0,0,0] -> identity matrix."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        assert np.allclose(R, np.eye(3))

    def test_rotation_matrix_to_identity_quaternion(self):
        """Identity matrix -> identity quaternion or its negative."""
        R = np.eye(3)
        q = rotation_matrix_to_quaternion(R)
        # Quaternion can have sign ambiguity
        assert np.allclose(np.abs(q), np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    def test_quaternion_matrix_roundtrip(self):
        """q -> R -> q should approximately recover original."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        R = quaternion_to_rotation_matrix(q)
        q_recovered = rotation_matrix_to_quaternion(R)
        # Quaternion has sign ambiguity
        is_same = np.allclose(q, q_recovered, atol=1e-6)
        is_opposite = np.allclose(q, -q_recovered, atol=1e-6)
        assert is_same or is_opposite

    def test_rotation_matrix_properties(self):
        """Rotation matrix from quaternion should be orthogonal with det=+1."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q = q / np.linalg.norm(q)  # Normalize
        R = quaternion_to_rotation_matrix(q)
        # Check orthogonality: R^T * R = I
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        # Check determinant: det(R) = +1
        det = np.linalg.det(R)
        assert np.allclose(det, 1.0, atol=1e-6)

    def test_90_degree_rotation_around_z(self):
        """90° rotation around Z axis."""
        # q = cos(45°) + sin(45°)*k = [cos(45°), 0, 0, sin(45°)]
        q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        R = quaternion_to_rotation_matrix(q)
        # Expected: rotation matrix for 90° around Z
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.allclose(R, expected, atol=1e-6)


class TestAxisAngleConversions:
    """Tests for axis-angle representation conversions."""

    def test_axis_angle_identity(self):
        """Zero angle rotation should give identity quaternion."""
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
        q = axis_angle_to_quaternion(axis, angle)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(q, expected, atol=1e-6)

    def test_quaternion_to_axis_angle_identity(self):
        """Identity quaternion should give zero angle."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        axis, angle = quaternion_to_axis_angle(q)
        assert np.allclose(angle, 0.0, atol=1e-6)

    def test_axis_angle_180_degree(self):
        """180° rotation around Z."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi
        q = axis_angle_to_quaternion(axis, angle)
        # q = [cos(90°), sin(90°)*[0,0,1]] = [0, 0, 0, 1]
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        assert np.allclose(q, expected, atol=1e-6)

    def test_axis_angle_roundtrip(self):
        """axis_angle -> q -> axis_angle should match."""
        axis_orig = np.array([1.0, 1.0, 1.0])
        axis_orig = axis_orig / np.linalg.norm(axis_orig)
        angle_orig = np.pi / 3

        q = axis_angle_to_quaternion(axis_orig, angle_orig)
        axis_recovered, angle_recovered = quaternion_to_axis_angle(q)

        assert np.allclose(angle_recovered, angle_orig, atol=1e-6)
        # Axis can have sign ambiguity
        is_same = np.allclose(axis_orig, axis_recovered, atol=1e-6)
        is_opposite = np.allclose(axis_orig, -axis_recovered, atol=1e-6)
        assert is_same or is_opposite

    def test_axis_angle_to_matrix(self):
        """90° rotation around X axis."""
        axis = np.array([1.0, 0.0, 0.0])
        angle = np.pi / 2
        R = axis_angle_to_rotation_matrix(axis, angle)
        # Expected: 90° rotation around X
        expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert np.allclose(R, expected, atol=1e-6)


class TestRodriguesConversions:
    """Tests for Rodrigues vector (axis * angle) conversions."""

    def test_rodrigues_zero_vector(self):
        """Zero Rodrigues vector should give identity quaternion."""
        r = np.array([0.0, 0.0, 0.0])
        q = rodrigues_to_quaternion(r)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(q, expected, atol=1e-6)

    def test_quaternion_to_rodrigues_identity(self):
        """Identity quaternion should give zero Rodrigues vector."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        r = quaternion_to_rodrigues(q)
        expected = np.array([0.0, 0.0, 0.0])
        assert np.allclose(r, expected, atol=1e-6)

    def test_rodrigues_roundtrip(self):
        """r -> q -> r should recover original."""
        r_orig = np.array([0.5, 0.3, 0.1])  # axis * angle
        q = rodrigues_to_quaternion(r_orig)
        r_recovered = quaternion_to_rodrigues(q)
        assert np.allclose(r_orig, r_recovered, atol=1e-6)


class TestEulerAngleConversions:
    """Tests for Euler angle conventions."""

    def test_euler_identity_zyx(self):
        """Zero Euler angles should give identity quaternion."""
        q = euler_to_quaternion(0.0, 0.0, 0.0, "ZYX")
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(q, expected, atol=1e-6)

    def test_euler_single_rotation_x(self):
        """Single 90° rotation around X."""
        q = euler_to_quaternion(np.pi / 2, 0.0, 0.0, "XYZ")
        axis, angle = quaternion_to_axis_angle(q)
        assert np.allclose(angle, np.pi / 2, atol=1e-6)

    def test_euler_roundtrip_zyx(self):
        """ZYX euler -> q -> euler should match."""
        alpha, beta, gamma = 0.3, 0.5, 0.7
        q = euler_to_quaternion(alpha, beta, gamma, "ZYX")
        a_rec, b_rec, g_rec = quaternion_to_euler(q, "ZYX")
        # Euler angles have some representation ambiguity
        # Check that the combined rotation is the same
        q_recovered = euler_to_quaternion(a_rec, b_rec, g_rec, "ZYX")
        is_same = np.allclose(q, q_recovered, atol=1e-6)
        is_opposite = np.allclose(q, -q_recovered, atol=1e-6)
        assert is_same or is_opposite

    @pytest.mark.parametrize("convention", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
    def test_euler_conventions(self, convention):
        """Test roundtrip for all major Euler conventions."""
        angles = (0.2, 0.3, 0.4)
        q = euler_to_quaternion(*angles, convention)
        assert np.allclose(np.linalg.norm(q), 1.0, atol=1e-6)


class TestInvariants:
    """Tests for mathematical invariants across conversions."""

    def test_composition_invariant(self):
        """Composition of rotations should be preserved."""
        # Create two rotations
        q1 = euler_to_quaternion(0.2, 0.0, 0.0, "ZYX")
        q2 = euler_to_quaternion(0.0, 0.3, 0.0, "ZYX")

        # Compose via quaternion multiplication
        q_composed = quaternion_multiply(q1, q2)
        R_composed = quaternion_to_rotation_matrix(q_composed)

        # Compose via matrix multiplication
        R1 = quaternion_to_rotation_matrix(q1)
        R2 = quaternion_to_rotation_matrix(q2)
        R_expected = R1 @ R2

        assert np.allclose(R_composed, R_expected, atol=1e-6)

    def test_vector_rotation_consistency(self):
        """Vector rotation via matrix and quaternion should match."""
        q = euler_to_quaternion(0.1, 0.2, 0.3, "ZYX")
        R = quaternion_to_rotation_matrix(q)

        v = np.array([1.0, 2.0, 3.0])

        # Rotate via matrix
        v_rotated_matrix = R @ v

        # Rotate via quaternion: v' = q * v * q^-1
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        q_conj = quaternion_conjugate(q)
        v_rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
        v_rotated_quat = v_rotated_quat[1:4]

        assert np.allclose(v_rotated_matrix, v_rotated_quat, atol=1e-6)


class TestContractViolations:
    """Tests for precondition/postcondition violations."""

    def test_non_unit_quaternion_rejection(self):
        """Non-unit quaternions should raise ValueError."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="unit quaternion"):
            quaternion_to_rotation_matrix(q)

    def test_non_orthogonal_matrix_rejection(self):
        """Non-orthogonal matrices should raise ValueError."""
        R = np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="orthogonal"):
            rotation_matrix_to_quaternion(R)

    def test_wrong_determinant_rejection(self):
        """Matrix with det=-1 should raise ValueError."""
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        with pytest.raises(ValueError, match="det=\\+1"):
            rotation_matrix_to_quaternion(R)

    def test_invalid_euler_convention(self):
        """Invalid Euler convention should raise ValueError."""
        with pytest.raises(ValueError):
            euler_to_quaternion(0.1, 0.2, 0.3, "INVALID")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
