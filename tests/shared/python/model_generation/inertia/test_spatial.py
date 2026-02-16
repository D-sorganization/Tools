"""Tests for model_generation.inertia.spatial module.

Covers:
- mcI: constructing 6x6 spatial inertia
- spatial_inertia_to_urdf / urdf_to_spatial_inertia round-trip
- spatial_transform construction
- transform_spatial_inertia
- composite_rigid_body_inertia
"""

from __future__ import annotations

import numpy as np
import pytest
from model_generation.inertia.spatial import (
    composite_rigid_body_inertia,
    mcI,
    spatial_inertia_to_urdf,
    spatial_transform,
    transform_spatial_inertia,
    urdf_to_spatial_inertia,
)


class TestMcI:
    """Test spatial inertia construction."""

    def test_shape(self) -> None:
        I_com = np.eye(3) * 0.1
        result = mcI(mass=2.0, com=np.zeros(3), I_com=I_com)
        assert result.shape == (6, 6)

    def test_symmetric(self) -> None:
        I_com = np.diag([0.1, 0.2, 0.3])
        result = mcI(mass=1.0, com=np.array([0.1, 0.0, 0.0]), I_com=I_com)
        np.testing.assert_allclose(result, result.T, atol=1e-12)

    def test_zero_com(self) -> None:
        """With zero COM, lower-right block = m*I₃, upper-left = I_com."""
        mass = 3.0
        I_com = np.diag([0.1, 0.2, 0.3])
        result = mcI(mass=mass, com=np.zeros(3), I_com=I_com)
        np.testing.assert_allclose(result[:3, :3], I_com, atol=1e-12)
        np.testing.assert_allclose(result[3:, 3:], mass * np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result[:3, 3:], np.zeros((3, 3)), atol=1e-12)

    def test_bad_com_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="3-vector"):
            mcI(1.0, np.zeros(4), np.eye(3))

    def test_bad_inertia_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            mcI(1.0, np.zeros(3), np.eye(4))


class TestUrdfRoundTrip:
    """Test URDF <-> spatial inertia conversion."""

    def test_round_trip_com_at_origin(self) -> None:
        mass, com = 5.0, (0.0, 0.0, 0.0)
        ixx, iyy, izz = 0.1, 0.2, 0.3
        spatial = urdf_to_spatial_inertia(mass, com, ixx, iyy, izz)
        params = spatial_inertia_to_urdf(spatial)
        assert params["mass"] == pytest.approx(mass)
        assert params["ixx"] == pytest.approx(ixx)
        assert params["iyy"] == pytest.approx(iyy)
        assert params["izz"] == pytest.approx(izz)
        np.testing.assert_allclose(params["com"], list(com), atol=1e-12)

    def test_round_trip_with_com_offset(self) -> None:
        mass, com = 2.0, (0.05, -0.1, 0.15)
        ixx, iyy, izz = 0.01, 0.02, 0.03
        spatial = urdf_to_spatial_inertia(mass, com, ixx, iyy, izz)
        params = spatial_inertia_to_urdf(spatial)
        assert params["mass"] == pytest.approx(mass, rel=1e-10)
        np.testing.assert_allclose(params["com"], list(com), atol=1e-10)
        assert params["ixx"] == pytest.approx(ixx, rel=1e-10)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="6x6"):
            spatial_inertia_to_urdf(np.eye(5))


class TestSpatialTransform:
    """Test 6x6 spatial transform construction."""

    def test_identity_transform(self) -> None:
        X = spatial_transform(np.eye(3), np.zeros(3))
        np.testing.assert_allclose(X, np.eye(6), atol=1e-12)

    def test_shape(self) -> None:
        X = spatial_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
        assert X.shape == (6, 6)

    def test_bad_rotation_raises(self) -> None:
        with pytest.raises(ValueError, match="3x3"):
            spatial_transform(np.eye(2), np.zeros(3))

    def test_bad_translation_raises(self) -> None:
        with pytest.raises(ValueError, match="3-vector"):
            spatial_transform(np.eye(3), np.zeros(4))


class TestTransformSpatialInertia:
    """Test transforming spatial inertia between frames."""

    def test_identity_transform_unchanged(self) -> None:
        I_com = np.diag([0.1, 0.2, 0.3])
        I_s = mcI(mass=1.0, com=np.zeros(3), I_com=I_com)
        X = np.eye(6)
        result = transform_spatial_inertia(I_s, X)
        np.testing.assert_allclose(result, I_s, atol=1e-12)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="6x6"):
            transform_spatial_inertia(np.eye(5), np.eye(6))


class TestCompositeRigidBodyInertia:
    """Test combining multiple spatial inertias."""

    def test_single_body_identity(self) -> None:
        I_com = np.diag([0.1, 0.2, 0.3])
        I_s = mcI(mass=1.0, com=np.zeros(3), I_com=I_com)
        X = np.eye(6)
        result = composite_rigid_body_inertia([(I_s, X)])
        np.testing.assert_allclose(result, I_s, atol=1e-12)

    def test_two_bodies_sum(self) -> None:
        """Two identical bodies at same frame should double the inertia."""
        I_com = np.diag([0.1, 0.2, 0.3])
        I_s = mcI(mass=1.0, com=np.zeros(3), I_com=I_com)
        X = np.eye(6)
        result = composite_rigid_body_inertia([(I_s, X), (I_s, X)])
        np.testing.assert_allclose(result, 2.0 * I_s, atol=1e-12)
