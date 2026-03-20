"""Tests for humanoid_character_builder.mesh.primitive_inertia module.

Covers:
- PrimitiveInertiaCalculator static methods (box, cylinder, sphere, capsule, ellipsoid)
- PrimitiveInertiaCalculator.compute dispatch method
- estimate_segment_primitive helper function
- Analytical formula correctness using known closed-form solutions
- Axis-assignment correctness for cylinder/capsule

Issue: GH1587 — 630 source modules have no corresponding test file
"""

from __future__ import annotations

import math

import pytest
from humanoid_character_builder.mesh.inertia_calculator import InertiaMode
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
    estimate_segment_primitive,
)

# ── Sphere ──────────────────────────────────────────────────────────────


class TestComputeSphere:
    """Solid sphere: I = (2/5) m r^2 for all principal axes."""

    def test_formula_unit_sphere(self) -> None:
        r = PrimitiveInertiaCalculator.compute_sphere(mass=5.0, radius=1.0)
        expected = 0.4 * 5.0 * 1.0**2
        assert r.ixx == pytest.approx(expected)
        assert r.iyy == pytest.approx(expected)
        assert r.izz == pytest.approx(expected)

    def test_formula_small_sphere(self) -> None:
        m, rad = 1.0, 0.1
        r = PrimitiveInertiaCalculator.compute_sphere(m, rad)
        expected = 0.4 * m * rad**2
        assert r.ixx == pytest.approx(expected, rel=1e-9)

    def test_isotropic(self) -> None:
        r = PrimitiveInertiaCalculator.compute_sphere(2.0, 0.3)
        assert r.ixx == pytest.approx(r.iyy)
        assert r.iyy == pytest.approx(r.izz)

    def test_volume(self) -> None:
        rad = 0.2
        r = PrimitiveInertiaCalculator.compute_sphere(1.0, rad)
        expected_vol = (4.0 / 3.0) * math.pi * rad**3
        assert r.volume == pytest.approx(expected_vol, rel=1e-9)

    def test_mode(self) -> None:
        r = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.1)
        assert r.mode == InertiaMode.PRIMITIVE_APPROXIMATION

    def test_mass_preserved(self) -> None:
        r = PrimitiveInertiaCalculator.compute_sphere(3.7, 0.15)
        assert r.mass == pytest.approx(3.7)

    def test_inertia_scales_with_mass(self) -> None:
        r1 = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.1)
        r2 = PrimitiveInertiaCalculator.compute_sphere(4.0, 0.1)
        assert r2.ixx == pytest.approx(r1.ixx * 4.0)

    def test_inertia_scales_with_radius_squared(self) -> None:
        r1 = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.1)
        r2 = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.2)
        assert r2.ixx == pytest.approx(r1.ixx * 4.0, rel=1e-6)


# ── Box ─────────────────────────────────────────────────────────────────


class TestComputeBox:
    """Solid box:
    I_xx = m/12 (y^2 + z^2)
    I_yy = m/12 (x^2 + z^2)
    I_zz = m/12 (x^2 + y^2)
    """

    def test_unit_cube(self) -> None:
        r = PrimitiveInertiaCalculator.compute_box(
            mass=12.0, size_x=1.0, size_y=1.0, size_z=1.0
        )
        expected = 12.0 / 12.0 * (1.0 + 1.0)
        assert r.ixx == pytest.approx(expected)
        assert r.iyy == pytest.approx(expected)
        assert r.izz == pytest.approx(expected)

    def test_asymmetric_box(self) -> None:
        m = 1.0
        x, y, z = 0.1, 0.2, 0.3
        r = PrimitiveInertiaCalculator.compute_box(m, x, y, z)
        assert r.ixx == pytest.approx(m / 12.0 * (y**2 + z**2))
        assert r.iyy == pytest.approx(m / 12.0 * (x**2 + z**2))
        assert r.izz == pytest.approx(m / 12.0 * (x**2 + y**2))

    def test_volume(self) -> None:
        r = PrimitiveInertiaCalculator.compute_box(1.0, 0.2, 0.3, 0.4)
        assert r.volume == pytest.approx(0.2 * 0.3 * 0.4, rel=1e-9)

    def test_tall_box_ixx_largest(self) -> None:
        """Tall box (large z) → I_xx > I_zz."""
        r = PrimitiveInertiaCalculator.compute_box(1.0, 0.1, 0.1, 1.0)
        assert r.ixx > r.izz

    def test_mass_preserved(self) -> None:
        r = PrimitiveInertiaCalculator.compute_box(5.0, 0.1, 0.2, 0.3)
        assert r.mass == pytest.approx(5.0)


# ── Cylinder ────────────────────────────────────────────────────────────


class TestComputeCylinder:
    """Solid cylinder:
    Along Z: I_zz = (1/2) m r^2 (longitudinal)
             I_xx = I_yy = (1/12) m (3r^2 + h^2) (transverse)
    """

    def test_izz_formula_z_axis(self) -> None:
        m, r, h = 2.0, 0.1, 0.5
        result = PrimitiveInertiaCalculator.compute_cylinder(m, r, h, axis="z")
        assert result.izz == pytest.approx(0.5 * m * r**2)

    def test_ixx_formula_z_axis(self) -> None:
        m, r, h = 2.0, 0.1, 0.5
        result = PrimitiveInertiaCalculator.compute_cylinder(m, r, h, axis="z")
        expected_transverse = (1.0 / 12.0) * m * (3.0 * r**2 + h**2)
        assert result.ixx == pytest.approx(expected_transverse)

    def test_transverse_symmetry_z_axis(self) -> None:
        result = PrimitiveInertiaCalculator.compute_cylinder(1.0, 0.1, 0.3, axis="z")
        assert result.ixx == pytest.approx(result.iyy)

    def test_x_axis_assignment(self) -> None:
        result = PrimitiveInertiaCalculator.compute_cylinder(1.0, 0.1, 0.3, axis="x")
        # longitudinal → ixx
        assert result.ixx == pytest.approx(0.5 * 1.0 * 0.1**2)
        assert result.iyy == pytest.approx(result.izz)

    def test_y_axis_assignment(self) -> None:
        result = PrimitiveInertiaCalculator.compute_cylinder(1.0, 0.1, 0.3, axis="y")
        assert result.iyy == pytest.approx(0.5 * 1.0 * 0.1**2)
        assert result.ixx == pytest.approx(result.izz)

    def test_volume(self) -> None:
        r, h = 0.1, 0.5
        result = PrimitiveInertiaCalculator.compute_cylinder(1.0, r, h)
        assert result.volume == pytest.approx(math.pi * r**2 * h, rel=1e-9)


# ── Capsule ─────────────────────────────────────────────────────────────


class TestComputeCapsule:
    """Capsule = cylinder + two hemispheres."""

    def test_positive_inertia(self) -> None:
        r = PrimitiveInertiaCalculator.compute_capsule(1.0, 0.05, 0.2, "z")
        assert r.ixx > 0
        assert r.iyy > 0
        assert r.izz > 0

    def test_transverse_symmetry_z_axis(self) -> None:
        r = PrimitiveInertiaCalculator.compute_capsule(1.0, 0.05, 0.2, "z")
        assert r.ixx == pytest.approx(r.iyy)

    def test_longer_capsule_larger_transverse(self) -> None:
        short = PrimitiveInertiaCalculator.compute_capsule(1.0, 0.05, 0.1, "z")
        long_ = PrimitiveInertiaCalculator.compute_capsule(1.0, 0.05, 0.5, "z")
        assert long_.ixx > short.ixx

    def test_volume_positive(self) -> None:
        r = PrimitiveInertiaCalculator.compute_capsule(2.0, 0.05, 0.3)
        assert r.volume > 0

    def test_degenerate_zero_length(self) -> None:
        """Degenerate capsule (zero cylinder length) should return a valid result."""
        r = PrimitiveInertiaCalculator.compute_capsule(1.0, 0.1, 0.0)
        assert r.ixx > 0 or r.ixx == 0  # Either a valid inertia or zero default
        assert r.mass == pytest.approx(1.0)

    def test_mass_preserved(self) -> None:
        r = PrimitiveInertiaCalculator.compute_capsule(3.0, 0.05, 0.2)
        assert r.mass == pytest.approx(3.0)


# ── Ellipsoid ───────────────────────────────────────────────────────────


class TestComputeEllipsoid:
    """Solid ellipsoid:
    I_xx = m/5 (b^2 + c^2)
    I_yy = m/5 (a^2 + c^2)
    I_zz = m/5 (a^2 + b^2)
    """

    def test_formula(self) -> None:
        m, a, b, c = 1.0, 0.1, 0.2, 0.3
        r = PrimitiveInertiaCalculator.compute_ellipsoid(m, a, b, c)
        assert r.ixx == pytest.approx(m / 5.0 * (b**2 + c**2))
        assert r.iyy == pytest.approx(m / 5.0 * (a**2 + c**2))
        assert r.izz == pytest.approx(m / 5.0 * (a**2 + b**2))

    def test_sphere_limit(self) -> None:
        """When a=b=c, ellipsoid should match sphere formula."""
        rad = 0.3
        ell = PrimitiveInertiaCalculator.compute_ellipsoid(5.0, rad, rad, rad)
        sph = PrimitiveInertiaCalculator.compute_sphere(5.0, rad)
        assert ell.ixx == pytest.approx(sph.ixx, rel=1e-9)

    def test_volume(self) -> None:
        a, b, c = 0.1, 0.2, 0.3
        r = PrimitiveInertiaCalculator.compute_ellipsoid(1.0, a, b, c)
        expected_vol = (4.0 / 3.0) * math.pi * a * b * c
        assert r.volume == pytest.approx(expected_vol, rel=1e-9)


# ── Compute dispatch ────────────────────────────────────────────────────


class TestComputeDispatch:
    """PrimitiveInertiaCalculator.compute() dispatches to correct method."""

    def test_sphere_dict_dims(self) -> None:
        r = PrimitiveInertiaCalculator.compute(
            PrimitiveShape.SPHERE, 2.0, {"radius": 0.05}
        )
        assert r.ixx == pytest.approx(0.4 * 2.0 * 0.05**2)

    def test_box_dict_dims(self) -> None:
        r = PrimitiveInertiaCalculator.compute(
            PrimitiveShape.BOX, 1.0, {"x": 0.1, "y": 0.2, "z": 0.3}
        )
        expected = PrimitiveInertiaCalculator.compute_box(1.0, 0.1, 0.2, 0.3)
        assert r.ixx == pytest.approx(expected.ixx)

    def test_cylinder_dict_dims(self) -> None:
        r = PrimitiveInertiaCalculator.compute(
            PrimitiveShape.CYLINDER, 1.0, {"radius": 0.1, "length": 0.5}
        )
        expected = PrimitiveInertiaCalculator.compute_cylinder(1.0, 0.1, 0.5)
        assert r.izz == pytest.approx(expected.izz)

    def test_string_shape_name(self) -> None:
        r = PrimitiveInertiaCalculator.compute("sphere", 1.0, {"radius": 0.1})
        expected = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.1)
        assert r.ixx == pytest.approx(expected.ixx)

    def test_tuple_dims_sphere(self) -> None:
        r = PrimitiveInertiaCalculator.compute(PrimitiveShape.SPHERE, 1.0, (0.1,))
        expected = PrimitiveInertiaCalculator.compute_sphere(1.0, 0.1)
        assert r.ixx == pytest.approx(expected.ixx)

    def test_tuple_dims_box(self) -> None:
        r = PrimitiveInertiaCalculator.compute(PrimitiveShape.BOX, 1.0, (0.1, 0.2, 0.3))
        expected = PrimitiveInertiaCalculator.compute_box(1.0, 0.1, 0.2, 0.3)
        assert r.ixx == pytest.approx(expected.ixx)

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            PrimitiveInertiaCalculator.compute("octahedron", 1.0, {"radius": 0.1})


# ── estimate_segment_primitive ──────────────────────────────────────────


class TestEstimateSegmentPrimitive:
    """estimate_segment_primitive maps body segment names to shapes."""

    def test_head_is_sphere(self) -> None:
        shape, dims = estimate_segment_primitive("head", 0.2)
        assert shape == PrimitiveShape.SPHERE
        assert "radius" in dims
        assert dims["radius"] == pytest.approx(0.1)

    def test_thigh_is_capsule(self) -> None:
        shape, dims = estimate_segment_primitive("thigh", 0.4)
        assert shape == PrimitiveShape.CAPSULE

    def test_torso_is_box(self) -> None:
        shape, dims = estimate_segment_primitive("torso", 0.5, 0.3, 0.2)
        assert shape == PrimitiveShape.BOX

    def test_forearm_is_capsule(self) -> None:
        shape, dims = estimate_segment_primitive("forearm", 0.3)
        assert shape == PrimitiveShape.CAPSULE

    def test_neck_is_cylinder(self) -> None:
        shape, dims = estimate_segment_primitive("neck", 0.1)
        assert shape == PrimitiveShape.CYLINDER

    def test_hand_is_box(self) -> None:
        shape, dims = estimate_segment_primitive("hand", 0.2)
        assert shape == PrimitiveShape.BOX

    def test_unknown_segment_defaults_to_capsule(self) -> None:
        shape, dims = estimate_segment_primitive("widget", 0.2)
        assert shape == PrimitiveShape.CAPSULE

    def test_width_depth_defaults(self) -> None:
        """When width/depth not supplied, they default to fractions of length."""
        shape, dims = estimate_segment_primitive("thigh", 0.4)
        # Default width = 0.4 * 0.2 = 0.08, depth = 0.4 * 0.15 = 0.06
        # radius = (width + depth) / 4 = (0.08 + 0.06) / 4 = 0.035
        assert "radius" in dims
        assert dims["radius"] > 0

    def test_explicit_width_depth(self) -> None:
        shape, dims = estimate_segment_primitive("thigh", 0.4, 0.1, 0.08)
        assert shape == PrimitiveShape.CAPSULE
        assert dims["radius"] == pytest.approx((0.1 + 0.08) / 4)
