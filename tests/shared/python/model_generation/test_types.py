"""Tests for model_generation.core.types module.

Covers:
- GeometryType enum
- JointType enum
- Origin dataclass (from_position, from_dict, to_dict, to_urdf_string)
- Inertia dataclass (from_box, from_sphere, from_cylinder, is_positive_definite,
  satisfies_triangle_inequality, to_matrix, from_matrix round-trip, scale_to_mass)
- Material dataclass (factory classmethods, to_urdf_string)
"""

from __future__ import annotations

import pytest
from model_generation.core.types import (
    GeometryType,
    Inertia,
    JointType,
    Material,
    Origin,
)

# ── Enums ───────────────────────────────────────────────────────────────


class TestEnums:
    """Validate enum values."""

    def test_geometry_types(self) -> None:
        assert GeometryType.BOX.value == "box"
        assert GeometryType.CYLINDER.value == "cylinder"
        assert GeometryType.SPHERE.value == "sphere"

    def test_joint_types(self) -> None:
        assert JointType.FIXED.value == "fixed"
        assert JointType.REVOLUTE.value == "revolute"
        assert JointType.PRISMATIC.value == "prismatic"


# ── Origin ──────────────────────────────────────────────────────────────


class TestOrigin:
    """Test Origin dataclass."""

    def test_default_origin(self) -> None:
        o = Origin()
        assert o.xyz == (0.0, 0.0, 0.0)
        assert o.rpy == (0.0, 0.0, 0.0)

    def test_from_position(self) -> None:
        o = Origin.from_position(1.0, 2.0, 3.0)
        assert o.xyz == (1.0, 2.0, 3.0)
        assert o.rpy == (0.0, 0.0, 0.0)

    def test_to_dict_roundtrip(self) -> None:
        o = Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3))
        d = o.to_dict()
        o2 = Origin.from_dict(d)
        assert o2.xyz == pytest.approx(o.xyz)
        assert o2.rpy == pytest.approx(o.rpy)

    def test_to_urdf_string_contains_xyz(self) -> None:
        o = Origin(xyz=(1.0, 2.0, 3.0))
        s = o.to_urdf_string()
        assert "1.0" in s or "1" in s
        assert "origin" in s.lower() or "xyz" in s


# ── Inertia ────────────────────────────────────────────────────────────


class TestInertia:
    """Test Inertia dataclass."""

    def test_from_box(self) -> None:
        inertia = Inertia.from_box(mass=12.0, size_x=1.0, size_y=1.0, size_z=1.0)
        # m/12 * (a^2+b^2) = 12/12 * 2 = 2.0
        assert inertia.ixx == pytest.approx(2.0)
        assert inertia.iyy == pytest.approx(2.0)
        assert inertia.izz == pytest.approx(2.0)

    def test_from_sphere(self) -> None:
        inertia = Inertia.from_sphere(mass=10.0, radius=0.5)
        expected = 2.0 / 5.0 * 10.0 * 0.25
        assert inertia.ixx == pytest.approx(expected)
        assert inertia.ixx == pytest.approx(inertia.iyy)
        assert inertia.iyy == pytest.approx(inertia.izz)

    def test_from_cylinder(self) -> None:
        inertia = Inertia.from_cylinder(mass=4.0, radius=0.2, length=1.0)
        assert inertia.ixx == pytest.approx(inertia.iyy)
        assert inertia.ixx != pytest.approx(inertia.izz)

    def test_positive_definite(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=0.1)
        assert inertia.is_positive_definite()

    def test_diagonal_sphere(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=0.1)
        assert inertia.is_diagonal()

    def test_triangle_inequality(self) -> None:
        inertia = Inertia.from_box(mass=5.0, size_x=0.3, size_y=0.5, size_z=0.7)
        assert inertia.satisfies_triangle_inequality()

    def test_to_matrix_shape(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=0.1)
        mat = inertia.to_matrix()
        assert mat.shape == (3, 3)

    def test_from_matrix_roundtrip(self) -> None:
        original = Inertia.from_box(mass=2.0, size_x=0.3, size_y=0.4, size_z=0.5)
        mat = original.to_matrix()
        restored = Inertia.from_matrix(mat, mass=2.0)
        assert restored.ixx == pytest.approx(original.ixx)
        assert restored.iyy == pytest.approx(original.iyy)
        assert restored.izz == pytest.approx(original.izz)

    def test_scale_to_mass(self) -> None:
        inertia = Inertia.from_sphere(mass=2.0, radius=0.1)
        scaled = inertia.scale_to_mass(4.0)
        assert scaled.ixx == pytest.approx(inertia.ixx * 2.0)
        assert scaled.mass == pytest.approx(4.0)

    def test_to_urdf_string(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=0.1)
        s = inertia.to_urdf_string()
        assert "inertia" in s.lower() or "ixx" in s

    def test_to_dict_roundtrip(self) -> None:
        original = Inertia.from_box(mass=3.0, size_x=0.2, size_y=0.3, size_z=0.4)
        d = original.to_dict()
        restored = Inertia.from_dict(d)
        assert restored.ixx == pytest.approx(original.ixx)
        assert restored.mass == pytest.approx(original.mass)


# ── Material ───────────────────────────────────────────────────────────


class TestMaterial:
    """Test Material dataclass."""

    def test_default_color(self) -> None:
        m = Material(name="test")
        assert len(m.color) == 4
        assert all(0.0 <= c <= 1.0 for c in m.color)

    def test_factory_skin(self) -> None:
        m = Material.skin()
        assert m.name == "skin"

    def test_factory_bone(self) -> None:
        m = Material.bone()
        assert m.name == "bone"

    def test_factory_metal(self) -> None:
        m = Material.metal()
        assert m.name == "metal"

    def test_to_urdf_string(self) -> None:
        m = Material(name="test", color=(1.0, 0.0, 0.0, 1.0))
        s = m.to_urdf_string()
        assert "material" in s.lower() or "test" in s

    def test_to_dict_roundtrip(self) -> None:
        original = Material(name="custom", color=(0.5, 0.6, 0.7, 0.8))
        d = original.to_dict()
        restored = Material.from_dict(d)
        assert restored.name == original.name
        assert restored.color == pytest.approx(original.color)
