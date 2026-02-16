"""Tests for model_generation.inertia.primitives module.

Covers:
- box, cylinder, sphere, capsule, ellipsoid, hollow_cylinder, cone inertia
- parallel axis theorem
- combine_inertias
- Symmetry and dimensional invariants
"""

from __future__ import annotations

import pytest
from model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    combine_inertias,
    cone_inertia,
    cylinder_inertia,
    ellipsoid_inertia,
    hollow_cylinder_inertia,
    parallel_axis,
    sphere_inertia,
)

# ── Box ─────────────────────────────────────────────────────────────────


class TestBoxInertia:
    """Test inertia of a solid box."""

    def test_unit_cube(self) -> None:
        r = box_inertia(mass=12.0, size_x=1.0, size_y=1.0, size_z=1.0)
        # I = m/12 * (a^2 + b^2) for each axis
        expected = 12.0 / 12.0 * (1.0 + 1.0)
        assert r["ixx"] == pytest.approx(expected)
        assert r["iyy"] == pytest.approx(expected)
        assert r["izz"] == pytest.approx(expected)

    def test_off_diagonal_zero(self) -> None:
        r = box_inertia(1.0, 2.0, 3.0, 4.0)
        assert r["ixy"] == pytest.approx(0.0)
        assert r["ixz"] == pytest.approx(0.0)
        assert r["iyz"] == pytest.approx(0.0)

    def test_asymmetric(self) -> None:
        r = box_inertia(mass=6.0, size_x=1.0, size_y=2.0, size_z=3.0)
        assert r["ixx"] > r["izz"]  # ixx = m/12*(y^2+z^2) > izz = m/12*(x^2+y^2)


# ── Sphere ──────────────────────────────────────────────────────────────


class TestSphereInertia:
    """Test inertia of a solid sphere."""

    def test_isotropic(self) -> None:
        r = sphere_inertia(mass=5.0, radius=0.3)
        assert r["ixx"] == pytest.approx(r["iyy"])
        assert r["iyy"] == pytest.approx(r["izz"])

    def test_formula(self) -> None:
        m, rad = 10.0, 0.5
        r = sphere_inertia(m, rad)
        expected = 2.0 / 5.0 * m * rad**2
        assert r["ixx"] == pytest.approx(expected)

    def test_off_diagonal_zero(self) -> None:
        r = sphere_inertia(1.0, 1.0)
        assert r["ixy"] == pytest.approx(0.0)


# ── Cylinder ────────────────────────────────────────────────────────────


class TestCylinderInertia:
    """Test inertia of a solid cylinder."""

    def test_z_axis_symmetry(self) -> None:
        r = cylinder_inertia(mass=4.0, radius=0.2, length=1.0, axis="z")
        assert r["ixx"] == pytest.approx(r["iyy"])
        assert r["ixx"] != pytest.approx(r["izz"])

    def test_x_axis_symmetry(self) -> None:
        r = cylinder_inertia(mass=4.0, radius=0.2, length=1.0, axis="x")
        assert r["iyy"] == pytest.approx(r["izz"])

    def test_y_axis_symmetry(self) -> None:
        r = cylinder_inertia(mass=4.0, radius=0.2, length=1.0, axis="y")
        assert r["ixx"] == pytest.approx(r["izz"])


# ── Capsule ─────────────────────────────────────────────────────────────


class TestCapsuleInertia:
    """Test inertia of a solid capsule."""

    def test_positive_values(self) -> None:
        r = capsule_inertia(mass=3.0, radius=0.1, length=0.5, axis="z")
        assert r["ixx"] > 0
        assert r["iyy"] > 0
        assert r["izz"] > 0

    def test_larger_mass_larger_inertia(self) -> None:
        light = capsule_inertia(1.0, 0.1, 0.5, "z")
        heavy = capsule_inertia(5.0, 0.1, 0.5, "z")
        assert heavy["ixx"] > light["ixx"]


# ── Ellipsoid ───────────────────────────────────────────────────────────


class TestEllipsoidInertia:
    """Test inertia of a solid ellipsoid."""

    def test_sphere_limit(self) -> None:
        """When a=b=c, ellipsoid should equal sphere."""
        r = 0.3
        ell = ellipsoid_inertia(mass=5.0, a=r, b=r, c=r)
        sph = sphere_inertia(mass=5.0, radius=r)
        assert ell["ixx"] == pytest.approx(sph["ixx"], rel=1e-6)
        assert ell["iyy"] == pytest.approx(sph["iyy"], rel=1e-6)

    def test_off_diagonal_zero(self) -> None:
        r = ellipsoid_inertia(1.0, 0.2, 0.3, 0.4)
        assert r["ixy"] == pytest.approx(0.0)


# ── Hollow Cylinder ────────────────────────────────────────────────────


class TestHollowCylinderInertia:
    """Test inertia of a hollow cylinder (tube)."""

    def test_larger_than_solid(self) -> None:
        """Hollow cylinder should have larger radial inertia per unit mass."""
        solid = cylinder_inertia(mass=1.0, radius=0.1, length=0.5, axis="z")
        hollow = hollow_cylinder_inertia(
            mass=1.0, inner_radius=0.08, outer_radius=0.1, length=0.5, axis="z"
        )
        # izz (about cylinder axis) should be larger for hollow
        assert hollow["izz"] > solid["izz"]


# ── Cone ────────────────────────────────────────────────────────────────


class TestConeInertia:
    """Test inertia of a solid cone."""

    def test_positive_values(self) -> None:
        r = cone_inertia(mass=2.0, radius=0.15, height=0.4, axis="z")
        assert r["ixx"] > 0
        assert r["iyy"] > 0
        assert r["izz"] > 0


# ── Parallel Axis Theorem ──────────────────────────────────────────────


class TestParallelAxis:
    """Test parallel axis theorem."""

    def test_offset_increases_inertia(self) -> None:
        base = sphere_inertia(mass=2.0, radius=0.1)
        shifted = parallel_axis(base, mass=2.0, offset=(0.5, 0.0, 0.0))
        assert shifted["ixx"] == pytest.approx(base["ixx"])  # No change about x
        assert shifted["iyy"] > base["iyy"]
        assert shifted["izz"] > base["izz"]

    def test_zero_offset_unchanged(self) -> None:
        base = box_inertia(3.0, 0.2, 0.3, 0.4)
        shifted = parallel_axis(base, mass=3.0, offset=(0.0, 0.0, 0.0))
        for key in ("ixx", "iyy", "izz"):
            assert shifted[key] == pytest.approx(base[key])


# ── Combine Inertias ───────────────────────────────────────────────────


class TestCombineInertias:
    """Test combining multiple inertias."""

    def test_single_at_origin(self) -> None:
        base = sphere_inertia(mass=2.0, radius=0.1)
        combined = combine_inertias([(base, 2.0, (0.0, 0.0, 0.0))])
        assert combined["ixx"] == pytest.approx(base["ixx"])

    def test_two_bodies_symmetric(self) -> None:
        """Two identical bodies at equal distance should produce symmetric inertia."""
        mass = 1.0
        base = sphere_inertia(mass=mass, radius=0.05)
        combined = combine_inertias(
            [
                (base, mass, (0.5, 0.0, 0.0)),
                (base, mass, (-0.5, 0.0, 0.0)),
            ]
        )
        # Should be symmetric about x
        assert combined["iyy"] == pytest.approx(combined["izz"])
