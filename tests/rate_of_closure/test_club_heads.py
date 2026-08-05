"""Tests for type-specific heads, hosel points, and volumetrics (H1, #4125).

The numeric pins in :class:`TestVolumetricsParity` are mirrored
verbatim by the vitest suites (``web/src/model/heads.test.ts`` and
``web/src/model/volumetrics.test.ts``).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure._contracts import PreconditionError
from rate_of_closure.club import (
    CLUB_LIBRARY,
    ClubType,
    HeadStyle,
    build_parametric_head,
    face_center_point,
    get_club,
    head_cog,
    hosel_point,
    is_watertight,
    mesh_volume_centroid,
)
from rate_of_closure.club.head_profiles import (
    PLUMBER_NECK_OFFSET_M,
    mass_scale,
    profile_for,
    resolved_style,
)

pytestmark = pytest.mark.unit

_BLADE = "Blade Putter"
_MALLET = "Mallet Putter"


def _extents(name: str) -> np.ndarray:
    flat = build_parametric_head(CLUB_LIBRARY[name]).reshape(-1, 3)
    return np.asarray(flat.max(axis=0) - flat.min(axis=0))


def _cube_mesh(side: float = 1.0, center: float = 5.0) -> np.ndarray:
    """A watertight outward-wound cube, centered off-origin."""
    h = side / 2.0
    corners = np.array(
        [[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)]
    )
    # Each face as two triangles, outward winding (right-hand rule).
    faces = (
        (0, 1, 3, 2, (-1, 0, 0)),
        (4, 6, 7, 5, (1, 0, 0)),
        (0, 4, 5, 1, (0, -1, 0)),
        (2, 3, 7, 6, (0, 1, 0)),
        (0, 2, 6, 4, (0, 0, -1)),
        (1, 5, 7, 3, (0, 0, 1)),
    )
    triangles = []
    for a, b, c, d, normal in faces:
        for tri in ((a, b, c), (a, c, d)):
            pts = corners[list(tri)]
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if float(n @ np.asarray(normal)) < 0.0:
                pts = pts[::-1]
            triangles.append(pts)
    return np.asarray(triangles) + center


def _sphere_mesh(radius: float = 1.0, bands: int = 48) -> np.ndarray:
    """A UV-sphere triangulation, outward-wound and bit-exact at seams.

    Vertices are built once per (ring, meridian) index — with the
    meridian wrapped modulo — so shared edges reuse identical bits and
    the watertightness check holds.
    """
    grid = [
        [
            radius
            * np.array(
                [
                    math.sin(math.pi * i / bands) * math.cos(math.pi * j / bands),
                    math.sin(math.pi * i / bands) * math.sin(math.pi * j / bands),
                    math.cos(math.pi * i / bands),
                ]
            )
            for j in range(2 * bands)
        ]
        for i in range(bands + 1)
    ]
    pole_top = radius * np.array([0.0, 0.0, 1.0])
    pole_bot = radius * np.array([0.0, 0.0, -1.0])
    triangles = []
    for j in range(2 * bands):
        k = (j + 1) % (2 * bands)
        triangles.append(np.array([pole_top, grid[1][j], grid[1][k]]))
        triangles.append(np.array([grid[bands - 1][j], pole_bot, grid[bands - 1][k]]))
        for i in range(1, bands - 1):
            a, b = grid[i][j], grid[i + 1][j]
            c, d = grid[i + 1][k], grid[i][k]
            triangles.append(np.array([a, b, c]))
            triangles.append(np.array([a, c, d]))
    return np.asarray(triangles)


class TestTypeProportions:
    def test_iron_depth_much_less_than_wood_depth(self) -> None:
        assert _extents("7-Iron")[0] < 0.4 * _extents("3-Wood")[0]

    def test_hybrid_is_intermediate_between_iron_and_wood(self) -> None:
        iron, hybrid, wood = (_extents(n)[0] for n in ("7-Iron", "3-Hybrid", "3-Wood"))
        assert iron < hybrid < wood

    def test_blade_putter_shallower_than_mallet(self) -> None:
        assert _extents(_BLADE)[0] < 0.5 * _extents(_MALLET)[0]

    def test_putters_are_low_wide_bodies(self) -> None:
        for name in (_BLADE, _MALLET):
            depth, height, width = _extents(name)
            assert height < 0.035, name  # low profile vs a 62 mm driver
            assert width > 2.5 * height, name

    def test_iron_blade_is_wide_and_thin(self) -> None:
        depth, height, width = _extents("7-Iron")
        assert width > 2.0 * depth  # heel-toe blade, not a deep body

    def test_wedge_carries_higher_loft_than_iron(self) -> None:
        assert get_club("Sand Wedge").loft_deg > get_club("9-Iron").loft_deg

    def test_every_library_head_is_watertight_and_deterministic(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            first = build_parametric_head(spec)
            second = build_parametric_head(spec)
            assert np.array_equal(first, second), name
            assert is_watertight(first), name

    def test_putter_styles_resolve_to_distinct_profiles(self) -> None:
        blade, mallet = CLUB_LIBRARY[_BLADE], CLUB_LIBRARY[_MALLET]
        assert resolved_style(blade) is HeadStyle.BLADE
        assert resolved_style(mallet) is HeadStyle.MALLET
        assert profile_for(blade) is not profile_for(mallet)
        auto = replace(blade, head_style=HeadStyle.AUTO)
        assert profile_for(auto) is profile_for(blade)


class TestHosel:
    def test_hosel_is_on_the_heel_side_for_every_club(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            x, y, z = hosel_point(spec)
            assert z < 0.0, name

    def test_hosel_sits_on_the_head_envelope(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            flat = build_parametric_head(spec).reshape(-1, 3)
            low, high = flat.min(axis=0), flat.max(axis=0)
            x, y, z = hosel_point(spec)
            margin = 0.012  # a hosel may stand slightly proud of the crown
            assert low[0] - margin <= x <= high[0] + margin, name
            assert low[1] - margin <= y <= high[1] + margin, name
            assert low[2] - margin <= z <= high[2] + margin, name

    def test_blade_putter_hosel_has_plumbers_neck_setback(self) -> None:
        blade = CLUB_LIBRARY[_BLADE]
        face_x = face_center_point(blade)[0]
        setback = face_x - hosel_point(blade)[0]
        scale = mass_scale(blade)
        assert setback == pytest.approx(PLUMBER_NECK_OFFSET_M * scale, rel=1e-12)

    def test_woods_attach_at_the_heel_crown_transition(self) -> None:
        wood = CLUB_LIBRARY["3-Wood"]
        x, y, z = hosel_point(wood)
        flat = build_parametric_head(wood).reshape(-1, 3)
        assert y > 0.8 * flat[:, 1].max()  # near the crown
        assert x > 0.0  # forward half, near the face

    def test_hosel_scales_with_head_mass(self) -> None:
        wood = CLUB_LIBRARY["3-Wood"]
        heavy = replace(wood, head_mass_kg=wood.head_mass_kg * 2.0)
        ratio = mass_scale(heavy) / mass_scale(wood)
        assert np.allclose(
            np.asarray(hosel_point(heavy)), np.asarray(hosel_point(wood)) * ratio
        )


class TestVolumetrics:
    def test_cube_volume_and_centroid_exact(self) -> None:
        volume, centroid = mesh_volume_centroid(_cube_mesh(side=2.0, center=5.0))
        assert volume == pytest.approx(8.0, rel=1e-12)
        assert centroid == pytest.approx([5.0, 5.0, 5.0], rel=1e-12)

    def test_sphere_volume_and_centroid_within_one_percent(self) -> None:
        volume, centroid = mesh_volume_centroid(_sphere_mesh(radius=0.5))
        assert volume == pytest.approx(4.0 / 3.0 * math.pi * 0.5**3, rel=0.01)
        assert np.allclose(centroid, 0.0, atol=1e-9)

    def test_open_mesh_rejected(self) -> None:
        open_mesh = _cube_mesh()[:-1]
        assert not is_watertight(open_mesh)
        with pytest.raises(PreconditionError, match="watertight"):
            mesh_volume_centroid(open_mesh)

    def test_inward_winding_rejected(self) -> None:
        inverted = _cube_mesh()[:, ::-1, :]
        with pytest.raises(Exception, match="volume must be positive"):
            mesh_volume_centroid(inverted)


class TestCogReconciliation:
    #: Plausible per-type bands for CG depth/height [m]: wide enough to
    #: hold both the published-typical spec values and the geometric
    #: centroid of the uniform-density envelope (a solid's centroid
    #: sits deeper than a hollow head's CG), tight enough that a wood
    #: COG could never pass as an iron's.
    DEPTH_BANDS = {
        ClubType.DRIVER: (0.015, 0.065),
        ClubType.WOOD: (0.015, 0.065),
        ClubType.HYBRID: (0.012, 0.045),
        ClubType.IRON: (0.008, 0.025),
        ClubType.WEDGE: (0.007, 0.032),
    }
    HEIGHT_BANDS = {
        ClubType.DRIVER: (0.020, 0.035),
        ClubType.WOOD: (0.020, 0.035),
        ClubType.HYBRID: (0.015, 0.030),
        ClubType.IRON: (0.010, 0.022),
        ClubType.WEDGE: (0.010, 0.020),
    }
    PUTTER_DEPTH_BANDS = {
        HeadStyle.BLADE: (0.005, 0.020),
        HeadStyle.MALLET: (0.020, 0.050),
    }

    def test_generated_cog_lands_in_the_spec_band_per_type(self) -> None:
        for name, spec in CLUB_LIBRARY.items():
            report = head_cog(spec)
            if spec.club_type is ClubType.PUTTER:
                low, high = self.PUTTER_DEPTH_BANDS[resolved_style(spec)]
                height_band = (0.005, 0.018)
            else:
                low, high = self.DEPTH_BANDS[spec.club_type]
                height_band = self.HEIGHT_BANDS[spec.club_type]
            for value in (report.cg_depth_m, report.spec_cg_depth_m):
                assert low <= value <= high, (name, "depth", value)
            for value in (report.cg_height_m, report.spec_cg_height_m):
                assert height_band[0] <= value <= height_band[1], (
                    name,
                    "height",
                    value,
                )

    def test_report_carries_both_geometric_and_spec_values(self) -> None:
        report = head_cog(get_club("Driver 10.5°"))
        assert report.spec_cg_depth_m == pytest.approx(0.025)
        assert report.spec_cg_height_m == pytest.approx(0.028)
        assert report.volume_m3 > 0.0
        assert report.cg_depth_m > 0.0

    def test_cog_near_centerline_for_symmetric_heads(self) -> None:
        for name in ("Driver 10.5°", _MALLET):
            report = head_cog(CLUB_LIBRARY[name])
            assert abs(report.cog[2]) < 1e-4, name  # symmetric in z


class TestVolumetricsParity:
    """Pinned numbers mirrored in web/src/model/volumetrics.test.ts."""

    def test_driver_head_volume_and_cog_pinned(self) -> None:
        volume, centroid = mesh_volume_centroid(
            build_parametric_head(get_club("Driver 10.5°"))
        )
        assert volume == pytest.approx(5.713864825174431e-4, rel=1e-12)
        assert centroid[0] == pytest.approx(6.867533458253036e-3, rel=1e-12)
        assert centroid[1] == pytest.approx(-6.598125640208495e-4, rel=1e-12)
        assert centroid[2] == pytest.approx(-1.035457116425388e-6, rel=1e-9)

    def test_blade_putter_volume_and_cog_pinned(self) -> None:
        volume, centroid = mesh_volume_centroid(build_parametric_head(get_club(_BLADE)))
        assert volume == pytest.approx(4.601486227358806e-5, rel=1e-12)
        assert centroid[0] == pytest.approx(1.025352642901628e-3, rel=1e-12)
        assert centroid[1] == pytest.approx(-2.017195987009723e-3, rel=1e-12)

    def test_blade_putter_hosel_pinned(self) -> None:
        x, y, z = hosel_point(get_club(_BLADE))
        assert x == pytest.approx(0.0025, rel=1e-12)
        assert y == pytest.approx(0.0125, rel=1e-12)
        assert z == pytest.approx(-0.046, rel=1e-12)
