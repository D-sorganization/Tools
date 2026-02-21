"""TDD tests for the unified rotation converter API.

Tests the high-level ``RotationConverter`` that orchestrates all conversions
through a hub-and-spoke architecture (quaternion hub) with DbC contracts.

Written BEFORE implementation (TDD).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rotation_converter.converter import Rotation, RotationConverter

ATOL = 1e-10


# ===========================================================================
# Rotation data class
# ===========================================================================


class TestRotationDataClass:
    """The Rotation object holds a rotation in any representation."""

    def test_from_quaternion(self) -> None:
        r = Rotation.from_quaternion([1, 0, 0, 0])
        np.testing.assert_allclose(r.as_quaternion(), [1, 0, 0, 0], atol=ATOL)

    def test_from_rotation_matrix(self) -> None:
        r = Rotation.from_rotation_matrix(np.eye(3))
        np.testing.assert_allclose(r.as_rotation_matrix(), np.eye(3), atol=ATOL)

    def test_from_euler(self) -> None:
        r = Rotation.from_euler(0.1, 0.2, 0.3, "xyz")
        e = r.as_euler("xyz")
        assert len(e) == 3

    def test_from_axis_angle(self) -> None:
        r = Rotation.from_axis_angle([0, 0, 1], math.pi / 4)
        axis, angle = r.as_axis_angle()
        assert abs(angle - math.pi / 4) < ATOL

    def test_from_rodrigues(self) -> None:
        r = Rotation.from_rodrigues([0, 0, 0.5])
        rv = r.as_rodrigues()
        np.testing.assert_allclose(rv, [0, 0, 0.5], atol=ATOL)

    def test_identity_factory(self) -> None:
        r = Rotation.identity()
        np.testing.assert_allclose(r.as_quaternion(), [1, 0, 0, 0], atol=ATOL)
        np.testing.assert_allclose(r.as_rotation_matrix(), np.eye(3), atol=ATOL)


# ===========================================================================
# Cross-representation conversions via Rotation
# ===========================================================================


class TestRotationCrossConversions:
    """Test that any input representation can output any other."""

    def test_quaternion_to_matrix(self) -> None:
        r = Rotation.from_quaternion([1, 0, 0, 0])
        np.testing.assert_allclose(r.as_rotation_matrix(), np.eye(3), atol=ATOL)

    def test_matrix_to_quaternion(self) -> None:
        r = Rotation.from_rotation_matrix(np.eye(3))
        q = r.as_quaternion()
        assert abs(abs(q[0]) - 1.0) < ATOL

    def test_euler_to_axis_angle(self) -> None:
        r = Rotation.from_euler(math.pi / 2, 0, 0, "xyz")
        axis, angle = r.as_axis_angle()
        assert abs(angle - math.pi / 2) < ATOL
        np.testing.assert_allclose(axis, [1, 0, 0], atol=ATOL)

    def test_rodrigues_to_euler(self) -> None:
        r = Rotation.from_rodrigues([0.5, 0.0, 0.0])
        euler = r.as_euler("xyz")
        assert len(euler) == 3

    def test_axis_angle_to_rodrigues(self) -> None:
        axis = np.array([0, 0, 1])
        angle = 1.2
        r = Rotation.from_axis_angle(axis, angle)
        rv = r.as_rodrigues()
        np.testing.assert_allclose(rv, [0, 0, 1.2], atol=ATOL)


# ===========================================================================
# Rotation composition
# ===========================================================================


class TestRotationComposition:
    """Test that rotations can be composed (multiplied)."""

    def test_compose_with_identity(self) -> None:
        r = Rotation.from_axis_angle([0, 0, 1], 0.5)
        identity = Rotation.identity()
        composed = r.compose(identity)
        np.testing.assert_allclose(
            composed.as_quaternion(), r.as_quaternion(), atol=ATOL
        )

    def test_compose_inverse_gives_identity(self) -> None:
        r = Rotation.from_axis_angle([0, 0, 1], 0.5)
        r_inv = r.inverse()
        composed = r.compose(r_inv)
        np.testing.assert_allclose(composed.as_rotation_matrix(), np.eye(3), atol=ATOL)

    def test_compose_order_matters(self) -> None:
        r1 = Rotation.from_axis_angle([1, 0, 0], math.pi / 2)
        r2 = Rotation.from_axis_angle([0, 1, 0], math.pi / 2)
        c1 = r1.compose(r2)
        c2 = r2.compose(r1)
        # Non-commutative: results should differ
        assert not np.allclose(c1.as_quaternion(), c2.as_quaternion(), atol=ATOL)


# ===========================================================================
# RotationConverter class (static utility)
# ===========================================================================


class TestRotationConverterStaticAPI:
    """Test the RotationConverter static/class-method API."""

    def test_convert_quaternion_to_matrix(self) -> None:
        q = np.array([1.0, 0, 0, 0])
        R = RotationConverter.quaternion_to_rotation_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=ATOL)

    def test_convert_matrix_to_quaternion(self) -> None:
        R = np.eye(3)
        q = RotationConverter.rotation_matrix_to_quaternion(R)
        assert abs(abs(q[0]) - 1.0) < ATOL

    def test_convert_euler_to_quaternion(self) -> None:
        q = RotationConverter.euler_to_quaternion(0, 0, 0, "xyz")
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=ATOL)

    def test_convert_quaternion_to_euler(self) -> None:
        q = np.array([1.0, 0, 0, 0])
        euler = RotationConverter.quaternion_to_euler(q, "xyz")
        np.testing.assert_allclose(euler, [0, 0, 0], atol=ATOL)

    def test_convert_axis_angle_to_quaternion(self) -> None:
        q = RotationConverter.axis_angle_to_quaternion([0, 0, 1], math.pi)
        expected = np.array([0, 0, 0, 1], dtype=float)
        np.testing.assert_allclose(q, expected, atol=ATOL)

    def test_convert_quaternion_to_rodrigues(self) -> None:
        q = np.array([1.0, 0, 0, 0])
        r = RotationConverter.quaternion_to_rodrigues(q)
        np.testing.assert_allclose(r, [0, 0, 0], atol=ATOL)


# ===========================================================================
# Contract violations (DbC)
# ===========================================================================


class TestContracts:
    """Verify DbC precondition enforcement."""

    def test_quaternion_wrong_length_raises(self) -> None:
        with pytest.raises(Exception):
            Rotation.from_quaternion([1, 0, 0])

    def test_rotation_matrix_wrong_shape_raises(self) -> None:
        with pytest.raises(Exception):
            Rotation.from_rotation_matrix(np.eye(4))

    def test_rotation_matrix_not_SO3_raises(self) -> None:
        bad = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        with pytest.raises(Exception):
            Rotation.from_rotation_matrix(bad)

    def test_axis_angle_non_unit_axis_raises(self) -> None:
        with pytest.raises(Exception):
            Rotation.from_axis_angle([2, 0, 0], 0.5)

    def test_rodrigues_wrong_length_raises(self) -> None:
        with pytest.raises(Exception):
            Rotation.from_rodrigues([1, 2])
