"""Tests for analytical primitive inertia formulas.

These validate the closed-form inertia tensors against known physics
identities (symmetry, parallel-axis theorem, additivity) rather than
just re-deriving the implementation.
"""

from __future__ import annotations

import math

import pytest
from model_generation.inertia import primitives as P

_ZERO_PRODUCTS = ("ixy", "ixz", "iyz")


def _assert_diagonal(inertia: dict[str, float]) -> None:
    """A centered, axis-aligned primitive has zero products of inertia."""
    for key in _ZERO_PRODUCTS:
        assert inertia[key] == 0.0


class TestBoxInertia:
    def test_cube_is_isotropic(self) -> None:
        # A cube has equal diagonal moments and zero products.
        inertia = P.box_inertia(12.0, 1.0, 1.0, 1.0)
        # I = m/12 * (a^2 + a^2) = 12/12 * 2 = 2
        assert inertia["ixx"] == pytest.approx(2.0)
        assert inertia["iyy"] == pytest.approx(2.0)
        assert inertia["izz"] == pytest.approx(2.0)
        _assert_diagonal(inertia)

    def test_anisotropic_box_axes_differ(self) -> None:
        inertia = P.box_inertia(1.0, 2.0, 4.0, 6.0)
        # ixx depends on y,z; iyy on x,z; izz on x,y -> all distinct.
        assert inertia["ixx"] != inertia["iyy"] != inertia["izz"]
        assert inertia["ixx"] == pytest.approx((1.0 / 12.0) * (16 + 36))

    def test_mass_scales_linearly(self) -> None:
        small = P.box_inertia(1.0, 2.0, 3.0, 4.0)
        big = P.box_inertia(5.0, 2.0, 3.0, 4.0)
        assert big["ixx"] == pytest.approx(5.0 * small["ixx"])

    def test_none_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            P.box_inertia(None, 1.0, 1.0, 1.0)  # type: ignore[arg-type]


class TestCylinderInertia:
    def test_axial_vs_perpendicular(self) -> None:
        inertia = P.cylinder_inertia(2.0, 0.5, 1.0)
        # axial about z: 0.5*m*r^2
        assert inertia["izz"] == pytest.approx(0.5 * 2.0 * 0.25)
        # perpendicular equal to each other
        assert inertia["ixx"] == pytest.approx(inertia["iyy"])
        _assert_diagonal(inertia)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_axis_places_axial_moment_correctly(self, axis: str) -> None:
        inertia = P.cylinder_inertia(3.0, 0.4, 1.2, axis=axis)
        axial = 0.5 * 3.0 * 0.4**2
        key = {"x": "ixx", "y": "iyy", "z": "izz"}[axis]
        assert inertia[key] == pytest.approx(axial)
        # The other two diagonal entries are the (equal) perpendicular moment.
        others = [
            v for k, v in inertia.items() if k in {"ixx", "iyy", "izz"} and k != key
        ]
        assert others[0] == pytest.approx(others[1])

    def test_none_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            P.cylinder_inertia(None, 1.0, 1.0)  # type: ignore[arg-type]


class TestSphereInertia:
    def test_isotropic(self) -> None:
        inertia = P.sphere_inertia(5.0, 2.0)
        expected = (2.0 / 5.0) * 5.0 * 4.0
        assert inertia["ixx"] == pytest.approx(expected)
        assert inertia["ixx"] == inertia["iyy"] == inertia["izz"]
        _assert_diagonal(inertia)

    def test_none_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            P.sphere_inertia(None, 1.0)  # type: ignore[arg-type]


class TestCapsuleInertia:
    def test_zero_dimensions_falls_back_to_sphere(self) -> None:
        # With radius 0 and length 0 the total volume is 0 -> sphere fallback.
        inertia = P.capsule_inertia(1.0, 0.0, 0.0)
        assert inertia == P.sphere_inertia(1.0, 0.0)

    def test_perpendicular_exceeds_axial(self) -> None:
        # A long capsule is harder to spin end-over-end than about its axis.
        inertia = P.capsule_inertia(2.0, 0.1, 1.0, axis="z")
        assert inertia["ixx"] > inertia["izz"]
        assert inertia["ixx"] == pytest.approx(inertia["iyy"])

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_axial_is_smallest_for_long_capsule(self, axis: str) -> None:
        inertia = P.capsule_inertia(2.0, 0.1, 1.0, axis=axis)
        key = {"x": "ixx", "y": "iyy", "z": "izz"}[axis]
        assert inertia[key] == min(inertia["ixx"], inertia["iyy"], inertia["izz"])


class TestEllipsoidInertia:
    def test_reduces_to_sphere_when_equal_axes(self) -> None:
        ell = P.ellipsoid_inertia(3.0, 1.5, 1.5, 1.5)
        sph = P.sphere_inertia(3.0, 1.5)
        assert ell["ixx"] == pytest.approx(sph["ixx"])
        assert ell["iyy"] == pytest.approx(sph["iyy"])
        assert ell["izz"] == pytest.approx(sph["izz"])

    def test_axis_ordering(self) -> None:
        # Largest semi-axis contributes most to the perpendicular moments.
        ell = P.ellipsoid_inertia(1.0, 1.0, 2.0, 3.0)
        # ixx ~ b^2 + c^2 (largest), izz ~ a^2 + b^2 (smallest)
        assert ell["ixx"] > ell["iyy"] > ell["izz"]


class TestHollowCylinderInertia:
    def test_reduces_to_solid_when_inner_zero(self) -> None:
        hollow = P.hollow_cylinder_inertia(2.0, 0.0, 0.5, 1.0)
        solid = P.cylinder_inertia(2.0, 0.5, 1.0)
        assert hollow["izz"] == pytest.approx(solid["izz"])
        assert hollow["ixx"] == pytest.approx(solid["ixx"])

    def test_thin_shell_axial_moment(self) -> None:
        # For a thin shell (r1 ~ r2 = R) axial moment approaches m*R^2.
        inertia = P.hollow_cylinder_inertia(1.0, 0.999, 1.0, 0.1)
        assert inertia["izz"] == pytest.approx(1.0, rel=1e-2)


class TestConeInertia:
    def test_axial_formula(self) -> None:
        inertia = P.cone_inertia(10.0, 1.0, 2.0)
        assert inertia["izz"] == pytest.approx((3.0 / 10.0) * 10.0 * 1.0)
        _assert_diagonal(inertia)

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_axis_selection(self, axis: str) -> None:
        inertia = P.cone_inertia(4.0, 1.0, 1.0, axis=axis)
        key = {"x": "ixx", "y": "iyy"}[axis]
        assert inertia[key] == pytest.approx((3.0 / 10.0) * 4.0)


class TestParallelAxis:
    def test_zero_offset_is_identity(self) -> None:
        base = {"ixx": 1.0, "iyy": 2.0, "izz": 3.0}
        shifted = P.parallel_axis(base, 5.0, (0.0, 0.0, 0.0))
        assert shifted["ixx"] == pytest.approx(1.0)
        assert shifted["iyy"] == pytest.approx(2.0)
        assert shifted["izz"] == pytest.approx(3.0)

    def test_offset_adds_md2_terms(self) -> None:
        base = {"ixx": 0.0, "iyy": 0.0, "izz": 0.0}
        shifted = P.parallel_axis(base, 2.0, (1.0, 0.0, 0.0))
        # offset along x: ixx unchanged, iyy/izz gain m*dx^2.
        assert shifted["ixx"] == pytest.approx(0.0)
        assert shifted["iyy"] == pytest.approx(2.0)
        assert shifted["izz"] == pytest.approx(2.0)

    def test_products_of_inertia_subtracted(self) -> None:
        base = {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}
        shifted = P.parallel_axis(base, 3.0, (1.0, 2.0, 0.0))
        assert shifted["ixy"] == pytest.approx(-3.0 * 1.0 * 2.0)

    def test_none_inertia_raises(self) -> None:
        with pytest.raises(ValueError):
            P.parallel_axis(None, 1.0, (0.0, 0.0, 0.0))  # type: ignore[arg-type]


class TestCombineInertias:
    def test_empty_returns_zero(self) -> None:
        assert P.combine_inertias([]) == {
            "ixx": 0.0,
            "iyy": 0.0,
            "izz": 0.0,
            "ixy": 0.0,
            "ixz": 0.0,
            "iyz": 0.0,
        }

    def test_zero_total_mass_returns_zero(self) -> None:
        unit = {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}
        result = P.combine_inertias([(unit, 0.0, (1.0, 0.0, 0.0))])
        assert result["ixx"] == 0.0

    def test_two_point_masses_match_analytic(self) -> None:
        # Two equal point masses on the x-axis at +/-d about combined COM.
        # Combined izz = sum m*d^2 = 2 * (1 * 1^2) = 2.
        zero = {"ixx": 0.0, "iyy": 0.0, "izz": 0.0}
        result = P.combine_inertias(
            [
                (zero, 1.0, (-1.0, 0.0, 0.0)),
                (zero, 1.0, (1.0, 0.0, 0.0)),
            ]
        )
        assert result["izz"] == pytest.approx(2.0)
        assert result["iyy"] == pytest.approx(2.0)
        assert result["ixx"] == pytest.approx(0.0)

    def test_single_body_at_com_is_unchanged(self) -> None:
        base = {"ixx": 0.5, "iyy": 0.7, "izz": 0.9}
        result = P.combine_inertias([(base, 2.0, (3.0, -1.0, 2.0))])
        # COM coincides with the single body, so no parallel-axis shift.
        assert result["ixx"] == pytest.approx(0.5)
        assert result["izz"] == pytest.approx(0.9)


def test_capsule_total_length_consistency() -> None:
    # Sanity: a capsule is heavier-distributed than a bare cylinder of the
    # same cylindrical length because of the hemispherical caps.
    cap = P.capsule_inertia(2.0, 0.2, 0.6, axis="z")
    cyl = P.cylinder_inertia(2.0, 0.2, 0.6, axis="z")
    assert cap["ixx"] > cyl["ixx"]
    assert math.isfinite(cap["izz"])
