"""Tests for the club package: specs, library, inertia, parametric head.

The numeric pins in :class:`TestParity` are mirrored verbatim by the
vitest suite (``web/src/model/club.test.ts``), keeping the Python and
TypeScript implementations in lock-step.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure._contracts import PreconditionError
from rate_of_closure.club import (
    CLUB_LIBRARY,
    REFERENCE_HEAD_MASS_KG,
    ClubSpec,
    ClubType,
    build_parametric_head,
    club_inertia,
    club_names,
    face_normal_at_offset,
    face_sagitta,
    get_club,
    parametric_head_mesh,
)
from rate_of_closure.club.geometry import RING_POINTS, superellipse_ring
from rate_of_closure.club.inertia import (
    GRIP_LENGTH_M,
    GRIP_TUBE_RADIUS_M,
    SHAFT_TUBE_RADIUS_M,
)

pytestmark = pytest.mark.unit

_DRIVER = "Driver 10.5°"

#: A hand-checkable composite: 1 m club, 200 g head, 100 g uniform
#: shaft, 50 g grip over 0.25 m (worked in the test docstrings below).
_HAND_SPEC = ClubSpec(
    name="Hand case",
    club_type=ClubType.IRON,
    length_m=1.0,
    head_mass_kg=0.2,
    loft_deg=30.0,
    lie_deg=60.0,
    moi_about_shaft_kg_m2=5.0e-4,
    cg_depth_m=0.02,
    cg_height_m=0.02,
)


class TestClubSpec:
    def test_bounds_reject_out_of_range_values(self) -> None:
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), head_mass_kg=0.05)
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), loft_deg=75.0)
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), face_bulge_radius_m=0.01)

    def test_rejects_non_numeric_and_non_finite(self) -> None:
        with pytest.raises(TypeError):
            replace(get_club(_DRIVER), length_m="long")  # type: ignore[arg-type]
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), length_m=math.nan)

    def test_rejects_empty_name_and_bad_type(self) -> None:
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), name="")
        with pytest.raises(PreconditionError):
            replace(get_club(_DRIVER), club_type="Driver")  # type: ignore[arg-type]

    def test_curved_face_flag(self) -> None:
        assert get_club(_DRIVER).has_curved_face
        assert not get_club("7-Iron").has_curved_face


class TestLibrary:
    def test_holds_exactly_sixteen_clubs_in_ladder_order(self) -> None:
        names = club_names()
        assert len(names) == 16
        assert names[0] == "Driver 9.5°"
        assert names[-2:] == ["Blade Putter", "Mallet Putter"]

    def test_driver_normalizes_source_row_to_si(self) -> None:
        """Source row: 45.5 in, 200 g, 5200 g·cm², CG 25 mm."""
        driver = get_club(_DRIVER)
        assert driver.length_m == pytest.approx(45.5 * 0.0254)
        assert driver.head_mass_kg == pytest.approx(0.200)
        assert driver.moi_about_shaft_kg_m2 == pytest.approx(5.2e-4)
        assert driver.cg_depth_m == pytest.approx(0.025)
        assert driver.lie_deg == pytest.approx(56.0)

    def test_loft_ladder_is_monotonic_driver_through_lob_wedge(self) -> None:
        lofts = [CLUB_LIBRARY[name].loft_deg for name in club_names()[:-2]]
        assert lofts == sorted(lofts)

    def test_woods_are_curved_irons_are_flat(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            if spec.club_type in (ClubType.DRIVER, ClubType.WOOD, ClubType.HYBRID):
                assert spec.has_curved_face, name
            else:
                assert not spec.has_curved_face, name

    def test_unknown_club_rejected(self) -> None:
        with pytest.raises(PreconditionError, match="unknown club"):
            get_club("2-Iron")


class TestInertia:
    def test_hand_computed_composition(self) -> None:
        """1 m club, 200 g head, 100 g shaft, 50 g grip.

        total = 0.35 kg
        balance = (0.2·1 + 0.1·0.5 + 0.05·0.125) / 0.35 = 0.25625/0.35
        I_grip = 0.2·1² + 0.1/3 + 0.05·0.25²/3 = 0.2343750
        I_shaft = 5e-4 + 0.1·0.006² + 0.05·0.011² = 5.0965e-4
        """
        inertia = club_inertia(_HAND_SPEC, shaft_mass_kg=0.1, grip_mass_kg=0.05)
        assert inertia.total_mass_kg == pytest.approx(0.35)
        assert inertia.balance_point_m == pytest.approx(0.25625 / 0.35)
        assert inertia.moi_about_grip_kg_m2 == pytest.approx(0.234375)
        assert inertia.moi_about_shaft_kg_m2 == pytest.approx(5.0965e-4)

    def test_parallel_axis_terms_use_documented_constants(self) -> None:
        inertia = club_inertia(_HAND_SPEC, shaft_mass_kg=0.1, grip_mass_kg=0.05)
        assert inertia.moi_about_shaft_kg_m2 == pytest.approx(
            5.0e-4 + 0.1 * SHAFT_TUBE_RADIUS_M**2 + 0.05 * GRIP_TUBE_RADIUS_M**2
        )
        assert inertia.balance_point_m == pytest.approx(
            (0.2 * 1.0 + 0.1 * 0.5 + 0.05 * GRIP_LENGTH_M / 2.0) / 0.35
        )

    def test_component_masses_validated(self) -> None:
        with pytest.raises(PreconditionError):
            club_inertia(_HAND_SPEC, shaft_mass_kg=0.0)
        with pytest.raises(PreconditionError):
            club_inertia(_HAND_SPEC, grip_mass_kg=0.5)

    def test_balance_point_sits_below_the_head_for_every_club(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            inertia = club_inertia(spec)
            assert 0.5 * spec.length_m < inertia.balance_point_m < spec.length_m, name


class TestFaceCurvature:
    def test_sagitta_matches_circle_formula_at_toe_offset(self) -> None:
        """s = R - sqrt(R² - t²) for the driver's 0.30 m bulge at 20 mm."""
        driver = get_club(_DRIVER)
        expected = 0.30 - math.sqrt(0.30**2 - 0.020**2)
        assert face_sagitta(driver, 0.020, 0.0) == pytest.approx(expected, rel=1e-12)

    def test_sagitta_sums_bulge_and_roll(self) -> None:
        driver = get_club(_DRIVER)
        expected = (0.30 - math.sqrt(0.30**2 - 0.020**2)) + (
            0.28 - math.sqrt(0.28**2 - 0.010**2)
        )
        assert face_sagitta(driver, 0.020, 0.010) == pytest.approx(expected, rel=1e-12)

    def test_flat_face_has_zero_sagitta(self) -> None:
        assert face_sagitta(get_club("7-Iron"), 0.020, 0.010) == 0.0

    def test_center_normal_is_pure_loft(self) -> None:
        for name in (_DRIVER, "7-Iron"):
            spec = get_club(name)
            normal = face_normal_at_offset(spec, 0.0, 0.0)
            lam = math.radians(spec.loft_deg)
            assert normal == pytest.approx((math.cos(lam), math.sin(lam), 0.0))

    def test_flat_face_normal_ignores_offset(self) -> None:
        iron = get_club("7-Iron")
        assert face_normal_at_offset(iron, 15.0, 8.0) == pytest.approx(
            face_normal_at_offset(iron, 0.0, 0.0)
        )

    def test_bulge_opens_the_face_toward_the_toe(self) -> None:
        driver = get_club(_DRIVER)
        toe = face_normal_at_offset(driver, 20.0, 0.0)
        heel = face_normal_at_offset(driver, -20.0, 0.0)
        assert toe[2] > 0.0 > heel[2]
        assert toe[2] == pytest.approx(-heel[2])

    def test_normals_are_unit_length(self) -> None:
        driver = get_club(_DRIVER)
        for toe, high in ((0.0, 0.0), (20.0, 10.0), (-15.0, -8.0)):
            normal = face_normal_at_offset(driver, toe, high)
            assert np.linalg.norm(normal) == pytest.approx(1.0, rel=1e-12)

    def test_offset_outside_radius_rejected(self) -> None:
        driver = get_club(_DRIVER)
        with pytest.raises(PreconditionError):
            face_normal_at_offset(driver, 400.0, 0.0)


class TestParametricHead:
    def test_mesh_is_closed_and_deterministic(self) -> None:
        driver = get_club(_DRIVER)
        first = build_parametric_head(driver)
        second = build_parametric_head(driver)
        assert first.shape == (12 * RING_POINTS, 3, 3)
        assert np.array_equal(first, second)

    def test_envelope_scales_with_head_mass(self) -> None:
        """Constant-density scaling: cbrt(m / 0.200 kg) on every axis."""
        wood = get_club("3-Wood")
        scale = (wood.head_mass_kg / REFERENCE_HEAD_MASS_KG) ** (1.0 / 3.0)
        flat = build_parametric_head(wood).reshape(-1, 3)
        assert flat[:, 2].max() - flat[:, 2].min() == pytest.approx(0.124 * scale)
        assert flat[:, 1].max() - flat[:, 1].min() == pytest.approx(0.062 * scale)

    def test_face_vertex_honors_bulge_sagitta(self) -> None:
        """The zero-loft toe vertex sits back by the circle sagitta."""
        spec = replace(get_club(_DRIVER), loft_deg=0.0, face_roll_radius_m=None)
        flat = build_parametric_head(spec).reshape(-1, 3)
        toe_face = flat[np.isclose(flat[:, 2], 0.058) & np.isclose(flat[:, 1], 0.0)]
        assert toe_face.shape[0] > 0
        expected_x = 0.055 - (0.30 - math.sqrt(0.30**2 - 0.058**2))
        assert toe_face[:, 0] == pytest.approx(expected_x, rel=1e-12)

    def test_flat_zero_loft_head_matches_reference_envelope(self) -> None:
        spec = replace(
            get_club(_DRIVER),
            loft_deg=0.0,
            face_bulge_radius_m=None,
            face_roll_radius_m=None,
        )
        flat = build_parametric_head(spec).reshape(-1, 3)
        extents = flat.max(axis=0) - flat.min(axis=0)
        np.testing.assert_allclose(extents, [0.11, 0.062, 0.124], atol=1e-12)

    def test_loft_tilts_face_triangle_normals_upward(self) -> None:
        """Face-patch normals average to ~(cos loft, sin loft, 0)."""
        spec = replace(
            get_club(_DRIVER), face_bulge_radius_m=None, face_roll_radius_m=None
        )
        mesh = parametric_head_mesh(spec)
        centroids = mesh.triangles.mean(axis=1)
        face = centroids[:, 0] > 0.045
        mean = mesh.normals[face].mean(axis=0)
        mean /= np.linalg.norm(mean)
        lam = math.radians(spec.loft_deg)
        np.testing.assert_allclose(mean, [math.cos(lam), math.sin(lam), 0.0], atol=1e-9)

    def test_head_mesh_carries_unit_normals(self) -> None:
        mesh = parametric_head_mesh(get_club("3-Wood"))
        norms = np.linalg.norm(mesh.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    def test_ring_helper_validates_inputs(self) -> None:
        with pytest.raises(PreconditionError):
            superellipse_ring(0.0, -0.01, 0.05)
        with pytest.raises(PreconditionError):
            superellipse_ring(0.0, 0.01, 0.05, points=2)


class TestParity:
    """Pinned numbers mirrored verbatim in web/src/model/club.test.ts."""

    def test_driver_inertia_pinned(self) -> None:
        inertia = club_inertia(get_club(_DRIVER))
        assert inertia.total_mass_kg == pytest.approx(0.325, rel=1e-12)
        assert inertia.balance_point_m == pytest.approx(0.863780769230769, rel=1e-12)
        assert inertia.moi_about_grip_kg_m2 == pytest.approx(
            0.301561226916667, rel=1e-12
        )
        assert inertia.moi_about_shaft_kg_m2 == pytest.approx(5.2875e-4, rel=1e-12)

    def test_driver_face_normal_pinned(self) -> None:
        normal = face_normal_at_offset(get_club(_DRIVER), 20.0, 10.0)
        assert normal[0] == pytest.approx(0.973950411287592, rel=1e-12)
        assert normal[1] == pytest.approx(0.216752844685502, rel=1e-12)
        assert normal[2] == pytest.approx(0.066624324938218, rel=1e-12)

    def test_driver_face_sagitta_pinned(self) -> None:
        driver = get_club(_DRIVER)
        assert face_sagitta(driver, 0.020, 0.0) == pytest.approx(
            6.6740905808465589e-4, rel=1e-12
        )
        assert face_sagitta(driver, 0.020, 0.010) == pytest.approx(
            8.4603746542022407e-4, rel=1e-12
        )

    def test_driver_mesh_pinned_vertex_and_extent(self) -> None:
        flat = build_parametric_head(get_club(_DRIVER)).reshape(-1, 3)
        assert flat[:, 0].max() == pytest.approx(0.058722579135751, rel=1e-12)
        target = np.array([0.049434717761548, -0.001031464094849, 0.058])
        assert (np.abs(flat - target).sum(axis=1) < 1e-12).any()
