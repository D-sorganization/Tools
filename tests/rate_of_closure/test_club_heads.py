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
    ClubSpec,
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
from rate_of_closure.club.geometry import RING_POINTS
from rate_of_closure.club.head_profiles import (
    IRON_PROFILE,
    PLUMBER_NECK_OFFSET_M,
    WEDGE_PROFILE,
    leading_edge_height,
    lean_point,
    mass_scale,
    profile_for,
    resolved_style,
)
from rate_of_closure.club.parametric_head import face_sagitta

pytestmark = pytest.mark.unit

_BLADE = "Blade Putter"
_MALLET = "Mallet Putter"

#: Every library club by name — the G1/G2 gates (#4799) run over all 16.
_ALL_CLUBS = list(CLUB_LIBRARY)
_BLADES = [
    name
    for name, spec in CLUB_LIBRARY.items()
    if spec.club_type in (ClubType.IRON, ClubType.WEDGE)
]
_DRIVERS = [
    name for name, spec in CLUB_LIBRARY.items() if spec.club_type is ClubType.DRIVER
]
_IRONS = [
    name for name, spec in CLUB_LIBRARY.items() if spec.club_type is ClubType.IRON
]
_WEDGES = [
    name for name, spec in CLUB_LIBRARY.items() if spec.club_type is ClubType.WEDGE
]


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
        # #4799 G1: the x extent now includes the topline's H*sin(loft)
        # lean-back, so the blade ratio is judged against the leaned
        # envelope (G3 re-authors the silhouettes themselves).
        depth, height, width = _extents("7-Iron")
        assert width > 1.85 * depth  # heel-toe blade, not a deep body

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


class TestLeadingEdgeLean:
    """G1 gates (#4799): loft leans the head about the leading edge.

    The mesh is built unlofted, then every vertex is sheared about the
    ``y = y_le`` leading-edge line: ``x' = x - dy sin(loft)``,
    ``y' = y_le + dy cos(loft)``, ``z' = z``. Mirrored test-for-test in
    ``web/src/model/heads.test.ts``.
    """

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_leading_edge_station_is_loft_invariant(self, name: str) -> None:
        """Onset eliminated: the leading edge stays at the authored face
        station, and loft never pushes the head past the unlofted extent."""
        spec = CLUB_LIBRARY[name]
        scale = mass_scale(spec)
        x0, hh, _hw, _yc = (c * scale for c in profile_for(spec).sections[0])
        expected_le = np.array(
            [x0 - face_sagitta(spec, 0.0, -hh), leading_edge_height(spec)]
        )
        lofted = build_parametric_head(spec).reshape(-1, 3)
        near = np.abs(lofted[:, :2] - expected_le).sum(axis=1) < 1e-6
        assert near.any(), "no mesh vertex at the authored leading edge"
        unlofted = build_parametric_head(replace(spec, loft_deg=0.0)).reshape(-1, 3)
        assert lofted[:, 0].max() <= unlofted[:, 0].max() + 1e-9

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_topline_sets_back_by_slant_height_times_sin_loft(self, name: str) -> None:
        """The authored face height H becomes slant height: the face-top
        vertex sits H*sin(loft) behind the leading edge (+-1%)."""
        spec = CLUB_LIBRARY[name]
        scale = mass_scale(spec)
        x0, hh, _hw, yc = (c * scale for c in profile_for(spec).sections[0])
        lam = math.radians(spec.loft_deg)
        height = 2.0 * hh
        expected_top = np.array(
            [
                x0 - face_sagitta(spec, 0.0, hh) - height * math.sin(lam),
                leading_edge_height(spec) + height * math.cos(lam),
            ]
        )
        flat = build_parametric_head(spec).reshape(-1, 3)
        near = np.abs(flat[:, :2] - expected_top).sum(axis=1) < 1e-6
        assert near.any(), "no mesh vertex at the expected face top"
        setback = float(flat[:, 0].max() - flat[near][:, 0].min())
        assert setback == pytest.approx(height * math.sin(lam), rel=0.01, abs=1e-9)

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_sole_height_is_loft_invariant(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        lofted = build_parametric_head(spec).reshape(-1, 3)
        unlofted = build_parametric_head(replace(spec, loft_deg=0.0)).reshape(-1, 3)
        assert abs(lofted[:, 1].min() - unlofted[:, 1].min()) <= 5.0e-4

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_vertical_extent_compresses_by_cos_loft(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        lofted = build_parametric_head(spec).reshape(-1, 3)
        unlofted = build_parametric_head(replace(spec, loft_deg=0.0)).reshape(-1, 3)
        expected = (unlofted[:, 1].max() - unlofted[:, 1].min()) * math.cos(
            math.radians(spec.loft_deg)
        )
        assert lofted[:, 1].max() - lofted[:, 1].min() == pytest.approx(
            expected, rel=1e-9
        )

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_z_extent_is_untouched_by_loft(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        lofted = build_parametric_head(spec).reshape(-1, 3)
        unlofted = build_parametric_head(replace(spec, loft_deg=0.0)).reshape(-1, 3)
        assert np.array_equal(lofted[:, 2], unlofted[:, 2])

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_lean_preserves_watertightness_and_positive_volume(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        triangles = build_parametric_head(spec)
        assert is_watertight(triangles)
        volume, _centroid = mesh_volume_centroid(triangles)
        assert volume > 0.0

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_triangle_count_is_loft_invariant(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        sections = 3 * (len(profile_for(spec).sections) - 1) + 1
        expected = (2 * (sections - 1) + 2 * 4 + 2) * RING_POINTS
        assert build_parametric_head(spec).shape == (expected, 3, 3)
        unlofted = build_parametric_head(replace(spec, loft_deg=0.0))
        assert unlofted.shape == (expected, 3, 3)

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_mesh_face_cap_center_is_the_leaned_face_center(self, name: str) -> None:
        """`face_center_point` and the mesh's face fan agree post-lean."""
        spec = CLUB_LIBRARY[name]
        flat = build_parametric_head(spec).reshape(-1, 3)
        center = np.asarray(face_center_point(spec))
        assert (np.abs(flat - center).sum(axis=1) < 1e-12).any()


class TestHoselAnchors:
    """G2 gates (#4799): the shaft lands even with the leading edge.

    Blades get a loft-aware anchor (a few mm of real offset, never
    onset); woods, hybrids, and putters lean the authored anchor.
    Mirrored test-for-test in ``web/src/model/heads.test.ts``.
    """

    @pytest.mark.parametrize("name", _BLADES)
    def test_blade_hosel_is_offset_never_onset(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        hosel_x = hosel_point(spec)[0]
        le_x = float(build_parametric_head(spec).reshape(-1, 3)[:, 0].max())
        assert hosel_x <= le_x, "onset: hosel ahead of the leading edge"
        assert le_x - hosel_x <= 8.0e-3

    @pytest.mark.parametrize("name", _DRIVERS)
    def test_driver_leading_edge_sits_ahead_of_hosel(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        hosel_x = hosel_point(spec)[0]
        le_x = float(build_parametric_head(spec).reshape(-1, 3)[:, 0].max())
        assert 20.0e-3 <= le_x - hosel_x <= 40.0e-3

    @pytest.mark.parametrize("name", _ALL_CLUBS)
    def test_non_blade_hosel_is_the_leaned_authored_anchor(self, name: str) -> None:
        spec = CLUB_LIBRARY[name]
        if spec.club_type in (ClubType.IRON, ClubType.WEDGE):
            pytest.skip("blades use the loft-aware anchor rule")
        profile = profile_for(spec)
        scale = mass_scale(spec)
        ax, ay, az = profile.hosel_anchor
        expected = lean_point(spec, (ax * scale, ay * scale, az * scale))
        assert hosel_point(spec) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("name", _BLADES)
    def test_blade_hosel_enters_at_the_heel_face_height(self, name: str) -> None:
        """y = y_le + f*H*cos(loft) with f = 0.58 (heel-short face)."""
        spec = CLUB_LIBRARY[name]
        scale = mass_scale(spec)
        hh = profile_for(spec).sections[0][1] * scale
        lam = math.radians(spec.loft_deg)
        expected_y = leading_edge_height(spec) + 0.58 * 2.0 * hh * math.cos(lam)
        assert hosel_point(spec)[1] == pytest.approx(expected_y, rel=1e-12)


def _sole_depth_reference_mm(spec: ClubSpec) -> float:
    """Front-to-back extent [mm] of the post-lean sole band, at reference.

    The sole band is every mesh vertex within 1 mm (at reference scale)
    of the lowest point; the leading-edge lean keeps ``y = y_le`` fibers
    fixed, so the band's x span is the sole's face-to-back width. The
    result is divided by the mass scale so gates read in reference-scale
    millimeters across the whole library.
    """
    scale = mass_scale(spec)
    flat = build_parametric_head(spec).reshape(-1, 3)
    band = flat[:, 1] <= flat[:, 1].min() + 1.0e-3 * scale
    return float((flat[band, 0].max() - flat[band, 0].min()) / scale * 1000.0)


def _refined_authored_sections(spec: ClubSpec) -> list[tuple[float, ...]]:
    """The generator's refined (mass-scaled) stations — same subdivision."""
    profile = profile_for(spec)
    scale = mass_scale(spec)
    authored = [tuple(c * scale for c in section) for section in profile.sections]
    refined: list[tuple[float, ...]] = []
    for first, second in zip(authored[:-1], authored[1:], strict=True):
        for step in range(3):
            fraction = step / 3
            refined.append(
                tuple(
                    a + fraction * (b - a) for a, b in zip(first, second, strict=True)
                )
            )
    refined.append(authored[-1])
    return refined


def _sole_slab_areas_m2(spec: ClubSpec) -> tuple[float, float]:
    """(front, rear) side-view areas of the post-lean sole slab.

    The slab is the bottom quarter of the leaned head's height; its
    side-view (z = 0) area between the sole curve and the slab line is
    integrated by trapezoid over the leaned station bottoms and split
    at the sole's x midpoint. Rear >= front is the muscle/bounce mass
    bias of a real wedge sole (#4803).
    """
    refined = _refined_authored_sections(spec)
    bottoms = [lean_point(spec, (x, yc - hh, 0.0)) for x, hh, _hw, yc in refined]
    tops = [lean_point(spec, (x, yc + hh, 0.0)) for x, hh, _hw, yc in refined]
    y_min = min(p[1] for p in bottoms)
    y_max = max(p[1] for p in tops)
    line = y_min + 0.25 * (y_max - y_min)
    pts = sorted((p[0], max(0.0, line - p[1])) for p in bottoms)
    xs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    x_mid = 0.5 * (xs[0] + xs[-1])
    c_mid = float(np.interp(x_mid, xs, cs))

    def _trapezoid(x: list[float], c: list[float]) -> float:
        return sum(
            0.5 * (c[i] + c[i + 1]) * (x[i + 1] - x[i]) for i in range(len(x) - 1)
        )

    rear = _trapezoid(
        [x for x in xs if x <= x_mid] + [x_mid],
        [c for x, c in zip(xs, cs, strict=True) if x <= x_mid] + [c_mid],
    )
    front = _trapezoid(
        [x_mid] + [x for x in xs if x > x_mid],
        [c_mid] + [c for x, c in zip(xs, cs, strict=True) if x > x_mid],
    )
    return front, rear


class TestBladeSilhouettes:
    """G3 gates (#4803): real blade silhouettes — sole depth, muscle,
    bounce hint. Judged post-lean on the generated mesh at reference
    scale. Published typical dimension spans (no brand geometry):
    iron sole widths ~18-24 mm, wedge sole widths ~26-32 mm. Mirrored
    test-for-test in ``web/src/model/heads.test.ts``.
    """

    @pytest.mark.parametrize("name", _WEDGES)
    def test_wedge_sole_depth_at_reference(self, name: str) -> None:
        assert 26.0 <= _sole_depth_reference_mm(CLUB_LIBRARY[name]) <= 32.0

    @pytest.mark.parametrize("name", _IRONS)
    def test_iron_sole_depth_at_reference(self, name: str) -> None:
        assert 18.0 <= _sole_depth_reference_mm(CLUB_LIBRARY[name]) <= 24.0

    @pytest.mark.parametrize("name", _WEDGES)
    def test_wedge_rear_sole_area_at_least_front(self, name: str) -> None:
        """Muscle bias: the rear half of the sole slab carries at least
        as much side-view material as the front half."""
        front, rear = _sole_slab_areas_m2(CLUB_LIBRARY[name])
        assert front > 0.0
        assert rear >= front

    @pytest.mark.parametrize("name", _WEDGES)
    def test_wedge_bounce_hint_low_point_behind_leading_edge(self, name: str) -> None:
        """A bounce hint: the sole's lowest fiber dips 0.2-1.0 mm below
        the leading edge and sits well behind it."""
        spec = CLUB_LIBRARY[name]
        scale = mass_scale(spec)
        flat = build_parametric_head(spec).reshape(-1, 3)
        dip = leading_edge_height(spec) - float(flat[:, 1].min())
        assert 0.2e-3 * scale <= dip <= 1.0e-3 * scale
        low_x = float(flat[np.argmin(flat[:, 1]), 0])
        x_le = profile_for(spec).sections[0][0] * scale
        assert low_x <= x_le - 2.0e-3 * scale

    @pytest.mark.parametrize("name", _IRONS)
    def test_iron_sole_stays_on_the_leading_edge_line(self, name: str) -> None:
        """Irons keep a flat sole: the lowest fiber is the leading edge."""
        spec = CLUB_LIBRARY[name]
        flat = build_parametric_head(spec).reshape(-1, 3)
        assert float(flat[:, 1].min()) == pytest.approx(
            leading_edge_height(spec), abs=1e-12
        )

    def test_cavity_recess_on_irons_only(self) -> None:
        assert IRON_PROFILE.rear_recess_m > 0.0
        assert WEDGE_PROFILE.rear_recess_m == 0.0

    @pytest.mark.parametrize("name", _BLADES)
    def test_tail_cap_center_realizes_the_recess(self, name: str) -> None:
        """The mesh's tail-fan center sits at the (possibly recessed)
        authored tail station under the lean — the cavity on irons,
        the plain tail cap on wedges."""
        spec = CLUB_LIBRARY[name]
        profile = profile_for(spec)
        scale = mass_scale(spec)
        tail_x, _hh, _hw, tail_yc = profile.sections[-1]
        expected = np.asarray(
            lean_point(
                spec,
                ((tail_x + profile.rear_recess_m) * scale, tail_yc * scale, 0.0),
            )
        )
        flat = build_parametric_head(spec).reshape(-1, 3)
        assert (np.abs(flat - expected).sum(axis=1) < 1e-12).any()


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
        # #4799 G1/G2: face center and hosel are both leaned about the
        # leading edge; the hosel top rides one face half-height above
        # the face center, so the x gap grows by hh*sin(loft) on top of
        # the authored plumber's neck.
        blade = CLUB_LIBRARY[_BLADE]
        face_x = face_center_point(blade)[0]
        setback = face_x - hosel_point(blade)[0]
        scale = mass_scale(blade)
        hh = profile_for(blade).sections[0][1] * scale
        expected = PLUMBER_NECK_OFFSET_M * scale + hh * math.sin(
            math.radians(blade.loft_deg)
        )
        assert setback == pytest.approx(expected, rel=1e-12)

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
    #: COG could never pass as an iron's. The iron bands widened for
    #: #4799 G1: the leading-edge lean pulls the leaned topline (and the
    #: solid centroid) up and back relative to the old center-pivot
    #: loft; G3 re-authors the blade silhouettes.
    DEPTH_BANDS = {
        ClubType.DRIVER: (0.015, 0.065),
        ClubType.WOOD: (0.015, 0.065),
        ClubType.HYBRID: (0.012, 0.045),
        ClubType.IRON: (0.008, 0.028),
        ClubType.WEDGE: (0.007, 0.032),
    }
    HEIGHT_BANDS = {
        ClubType.DRIVER: (0.020, 0.035),
        ClubType.WOOD: (0.020, 0.035),
        ClubType.HYBRID: (0.015, 0.030),
        ClubType.IRON: (0.010, 0.024),
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
        # Repinned for the leading-edge loft lean (#4799 G1).
        volume, centroid = mesh_volume_centroid(
            build_parametric_head(get_club("Driver 10.5°"))
        )
        assert volume == pytest.approx(5.704668123824279e-4, rel=1e-12)
        assert centroid[0] == pytest.approx(1.9271184917634455e-3, rel=1e-12)
        assert centroid[1] == pytest.approx(-4.688625881900988e-4, rel=1e-12)
        assert centroid[2] == pytest.approx(0.0, abs=1e-12)

    def test_blade_putter_volume_and_cog_pinned(self) -> None:
        # Repinned for the leading-edge loft lean (#4799 G1).
        volume, centroid = mesh_volume_centroid(build_parametric_head(get_club(_BLADE)))
        assert volume == pytest.approx(4.634707753682171e-5, rel=1e-12)
        assert centroid[0] == pytest.approx(4.6947209615312937e-4, rel=1e-12)
        assert centroid[1] == pytest.approx(-1.9102497659159376e-3, rel=1e-12)

    def test_blade_putter_hosel_pinned(self) -> None:
        # Repinned for the leaned authored anchor (#4799 G2).
        x, y, z = hosel_point(get_club(_BLADE))
        assert x == pytest.approx(1.1916010939264044e-3, rel=1e-12)
        assert y == pytest.approx(1.2465738368864346e-2, rel=1e-12)
        assert z == pytest.approx(-0.046, rel=1e-12)
