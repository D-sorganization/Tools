"""UpstreamDrift topography adapter + cross-engine gates (#4800 P9).

Format tests run against ``fixtures/ud_green_topography.json`` — a
document synthesized field-for-field to match UpstreamDrift's
``_surface_io._load_json_topography`` schema exactly (UD ships no
canned topography JSON to copy; the schema keys are ``contours``
``[{x, y, elevation}, ...]`` plus ``hole_position``).

The cross-engine consistency gates cover only what both roll models
share (see the ``ud_adapter`` module docstring): the ``-g * grad h``
gravity law, the constant-deceleration ``v^2 / (2 mu g)`` roll-out
form, signs, monotonicity, and the flat-green straight line. The mu
laws legitimately differ (UD ``0.196/stimp`` vs Tools
``~0.559/stimp``) and are pinned as a documented constant ratio, not
reconciled.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import (
    GridGreenSurface,
    PlanarGreenSurface,
    PuttLaunch,
    UdGreenTopography,
    green_surface_from_ud_json,
    green_surface_to_ud_json,
    roll_out_distance,
    simulate_putt_on_surface,
    solve_skid,
    stimp_to_rolling_mu,
)
from shared.python.swing_sim.putting.roll import GRAVITY_M_S2

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_FIXTURE = Path(__file__).parent / "fixtures" / "ud_green_topography.json"

#: The fixture's plane: heights = -0.03125 * x (binary-exact 3.125 %
#: downgrade toward +x), 5 x 5 nodes at 1 m spacing, hole at (3, 2).
_FIXTURE_GRADE_PERCENT = 3.125
_FIXTURE_HOLE = (3.0, 2.0)


def _ud_rolling_mu(stimp_ft: float) -> float:
    """UD's rolling-mu law, reproduced analytically (UD never imported).

    ``turf_properties.rolling_friction_coefficient`` with bent grass at
    its 3.0 mm default height of cut (height factor 1.0), NORMAL
    condition (multiplier 1.0), and no grain.
    """
    return 0.196 / stimp_ft


def _launch(speed_mps: float) -> PuttLaunch:
    return PuttLaunch(
        ball_speed_mps=speed_mps,
        launch_angle_deg=0.0,
        horizontal_speed_mps=speed_mps,
        spin_rad_s=0.0,
        effective_loft_deg=0.0,
    )


def _doc(points: list[tuple[float, float, float]], **extra: object) -> str:
    payload: dict[str, object] = {
        "contours": [{"x": x, "y": y, "elevation": h} for x, y, h in points]
    }
    payload.update(extra)
    return json.dumps(payload)


def _grid_points(
    xs: tuple[float, ...], ys: tuple[float, ...]
) -> list[tuple[float, float, float]]:
    return [(x, y, 0.0) for y in ys for x in xs]


def _imported_grid(height_fn: Callable[[float, float], float]) -> GridGreenSurface:
    """A 12 x 12 m green (y spans [-6, 6]) pushed through the UD wire."""
    spacing = 0.5
    heights = tuple(
        tuple(height_fn(i * spacing, -6.0 + j * spacing) for i in range(25))
        for j in range(25)
    )
    grid = GridGreenSurface(origin_m=(0.0, -6.0), spacing_m=spacing, heights_m=heights)
    return green_surface_from_ud_json(green_surface_to_ud_json(grid)).surface


class TestFixtureImport:
    """The synthesized-from-UD-schema fixture parses losslessly."""

    def test_fixture_imports_as_grid(self) -> None:
        parsed = green_surface_from_ud_json(_FIXTURE.read_text(encoding="utf-8"))
        assert isinstance(parsed, UdGreenTopography)
        surface = parsed.surface
        assert isinstance(surface, GridGreenSurface)
        assert surface.origin_m == (0.0, 0.0)
        assert surface.spacing_m == 1.0
        expected_row = (0.0, -0.03125, -0.0625, -0.09375, -0.125)
        assert surface.heights_m == (expected_row,) * 5
        assert parsed.hole_position_m == _FIXTURE_HOLE

    def test_fixture_gravity_matches_shared_law_bitwise(self) -> None:
        """Both engines use -g * grad h; on the fixture plane the grid
        gradient is binary-exact, so the match is bitwise."""
        surface = green_surface_from_ud_json(
            _FIXTURE.read_text(encoding="utf-8")
        ).surface
        plane = PlanarGreenSurface(grade_percent=_FIXTURE_GRADE_PERCENT, aspect_deg=0.0)
        for x, y in ((0.7, 1.3), (2.5, 3.1), (3.9, 0.4)):
            gx, gy = surface.gravity_inplane_mps2(x, y)
            px, py = plane.gravity_inplane_mps2(x, y)
            assert gx == px
            assert gy == py
            assert surface.height_m(x, y) == pytest.approx(
                plane.height_m(x, y), abs=1e-15
            )

    def test_fixture_roundtrip_is_lossless_and_byte_deterministic(self) -> None:
        parsed = green_surface_from_ud_json(_FIXTURE.read_text(encoding="utf-8"))
        text = green_surface_to_ud_json(
            parsed.surface, hole_position_m=parsed.hole_position_m
        )
        again = green_surface_from_ud_json(text)
        assert again == parsed
        assert (
            green_surface_to_ud_json(
                again.surface, hole_position_m=again.hole_position_m
            )
            == text
        )

    def test_export_loads_back_into_ud_key_shape(self) -> None:
        """The export uses exactly the keys UD's loader reads."""
        parsed = green_surface_from_ud_json(_FIXTURE.read_text(encoding="utf-8"))
        data = json.loads(
            green_surface_to_ud_json(
                parsed.surface, hole_position_m=parsed.hole_position_m
            )
        )
        assert set(data) == {"contours", "hole_position"}
        assert all(set(point) == {"x", "y", "elevation"} for point in data["contours"])
        assert len(data["contours"]) == 25


class TestPlanarExport:
    def test_planar_export_reimports_equivalently(self) -> None:
        plane = PlanarGreenSurface(grade_percent=2.0, aspect_deg=33.0)
        text = green_surface_to_ud_json(plane, extent_m=(4.0, 4.0), spacing_m=0.5)
        surface = green_surface_from_ud_json(text).surface
        # Bilinear grids reproduce planes exactly; only fp rounding of
        # the sampled node heights separates the two forms.
        for x, y in ((0.0, 0.0), (0.25, 3.6), (1.7, 2.2), (4.0, 4.0)):
            assert surface.height_m(x, y) == pytest.approx(
                plane.height_m(x, y), abs=1e-12
            )
            gx, gy = surface.gravity_inplane_mps2(x, y)
            px, py = plane.gravity_inplane_mps2(x, y)
            assert gx == pytest.approx(px, abs=1e-12)
            assert gy == pytest.approx(py, abs=1e-12)

    def test_planar_export_requires_extent_and_spacing(self) -> None:
        plane = PlanarGreenSurface(grade_percent=1.0, aspect_deg=0.0)
        with pytest.raises(ValueError):
            green_surface_to_ud_json(plane)
        with pytest.raises(ValueError):
            green_surface_to_ud_json(plane, extent_m=(4.0, 4.0))

    def test_grid_export_refuses_extent(self) -> None:
        grid = GridGreenSurface(
            origin_m=(0.0, 0.0), spacing_m=1.0, heights_m=((0.0, 0.0), (0.0, 0.0))
        )
        with pytest.raises(ValueError):
            green_surface_to_ud_json(grid, extent_m=(2.0, 2.0), spacing_m=1.0)

    def test_planar_extent_must_fit_the_lattice(self) -> None:
        plane = PlanarGreenSurface(grade_percent=1.0, aspect_deg=0.0)
        with pytest.raises(ValueError):
            green_surface_to_ud_json(plane, extent_m=(4.3, 4.0), spacing_m=0.5)

    def test_export_refuses_non_surface(self) -> None:
        with pytest.raises(TypeError):
            green_surface_to_ud_json(3.0)  # type: ignore[arg-type]

    def test_export_refuses_bad_hole(self) -> None:
        grid = GridGreenSurface(
            origin_m=(0.0, 0.0), spacing_m=1.0, heights_m=((0.0, 0.0), (0.0, 0.0))
        )
        with pytest.raises(ValueError):
            green_surface_to_ud_json(grid, hole_position_m=(1.0,))  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            green_surface_to_ud_json(grid, hole_position_m=(math.nan, 1.0))


class TestFailClosedImport:
    def test_refuses_slopes(self) -> None:
        text = _doc(
            _grid_points((0.0, 1.0), (0.0, 1.0)),
            slopes=[
                {
                    "center": [0.5, 0.5],
                    "radius": 1.0,
                    "direction": [1.0, 0.0],
                    "magnitude": 0.02,
                }
            ],
        )
        with pytest.raises(ValueError, match="slope"):
            green_surface_from_ud_json(text)

    def test_refuses_unknown_fields(self) -> None:
        text = _doc(_grid_points((0.0, 1.0), (0.0, 1.0)), turf={"stimp_rating": 10})
        with pytest.raises(ValueError, match="unknown"):
            green_surface_from_ud_json(text)

    def test_refuses_missing_contours(self) -> None:
        with pytest.raises(ValueError, match="contours"):
            green_surface_from_ud_json(json.dumps({"hole_position": [1.0, 1.0]}))

    def test_refuses_non_object_document(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_ud_json("[]")

    def test_refuses_bad_contour_fields(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_ud_json(
                json.dumps({"contours": [{"x": 0.0, "y": 0.0, "z": 0.0}]})
            )
        with pytest.raises(ValueError):
            green_surface_from_ud_json(
                json.dumps(
                    {"contours": [{"x": 0.0, "y": 0.0, "elevation": 0.0, "extra": 1}]}
                )
            )

    def test_refuses_non_finite_elevation(self) -> None:
        points = _grid_points((0.0, 1.0), (0.0, 1.0))
        points[0] = (0.0, 0.0, math.nan)
        # json.dumps writes the NaN literal, which json.loads accepts —
        # the adapter itself must refuse it.
        with pytest.raises(ValueError, match="finite"):
            green_surface_from_ud_json(_doc(points))

    def test_refuses_bool_numbers(self) -> None:
        with pytest.raises(TypeError):
            green_surface_from_ud_json(
                json.dumps(
                    {
                        "contours": [
                            {"x": True, "y": 0.0, "elevation": 0.0},
                            {"x": 1.0, "y": 0.0, "elevation": 0.0},
                            {"x": 0.0, "y": 1.0, "elevation": 0.0},
                            {"x": 1.0, "y": 1.0, "elevation": 0.0},
                        ]
                    }
                )
            )

    def test_refuses_scattered_points(self) -> None:
        """UD would RBF-interpolate these; runtime-free parsing refuses."""
        text = _doc([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)])
        with pytest.raises(ValueError, match="complete regular grid"):
            green_surface_from_ud_json(text)

    def test_refuses_duplicate_node(self) -> None:
        points = _grid_points((0.0, 1.0), (0.0, 1.0))
        points[3] = points[0]
        with pytest.raises(ValueError, match="duplicate"):
            green_surface_from_ud_json(_doc(points))

    def test_refuses_anisotropic_spacing(self) -> None:
        text = _doc(_grid_points((0.0, 1.0, 2.0), (0.0, 0.5, 1.0)))
        with pytest.raises(ValueError, match="one spacing"):
            green_surface_from_ud_json(text)

    def test_refuses_irregular_axis(self) -> None:
        text = _doc(_grid_points((0.0, 1.0, 3.0), (0.0, 1.0, 2.0)))
        with pytest.raises(ValueError, match="evenly spaced"):
            green_surface_from_ud_json(text)

    def test_refuses_single_row_or_column(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_ud_json(_doc(_grid_points((0.0, 1.0), (0.0,))))
        with pytest.raises(ValueError):
            green_surface_from_ud_json(_doc(_grid_points((0.0,), (0.0, 1.0))))

    def test_refuses_bad_hole_position(self) -> None:
        points = _grid_points((0.0, 1.0), (0.0, 1.0))
        with pytest.raises(ValueError):
            green_surface_from_ud_json(_doc(points, hole_position=[1.0]))
        with pytest.raises(TypeError):
            green_surface_from_ud_json(_doc(points, hole_position=["a", 1.0]))


class TestCrossEngineConsistency:
    """Gates on what both roll models share (module docstring)."""

    def test_mu_laws_share_the_inverse_stimp_form(self) -> None:
        """Both mu laws are c/stimp; the ratio is a stimp-independent
        constant (~2.854, i.e. UD rolls ~2.85x farther) — documented,
        not reconciled."""
        ratios = [
            stimp_to_rolling_mu(stimp) / _ud_rolling_mu(stimp)
            for stimp in (7.0, 10.0, 13.0)
        ]
        assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)
        assert ratios[1] == pytest.approx(ratios[2], rel=1e-12)
        assert ratios[1] == pytest.approx(2.854, abs=2e-3)

    def test_flat_green_rolls_straight(self) -> None:
        surface = _imported_grid(lambda x, y: 0.0)
        result = simulate_putt_on_surface(
            _launch(2.0), surface, stimp_ft=10.0, hole_distance_m=30.0
        )
        assert all(y == 0.0 for y in result.path_y_m)
        assert result.break_m == 0.0
        assert not result.holed

    def test_flat_green_rollout_matches_the_shared_law(self) -> None:
        """Constant-deceleration roll-out v^2/(2 mu g) — the same form
        in both engines — predicts the integrated distance."""
        surface = _imported_grid(lambda x, y: 0.0)
        speed = 2.0
        result = simulate_putt_on_surface(
            _launch(speed), surface, stimp_ft=10.0, hole_distance_m=30.0
        )
        skid = solve_skid(speed, 0.0, GOLF_BALL_RADIUS_M)
        tools_roll = roll_out_distance(skid.exit_speed_mps, stimp_to_rolling_mu(10.0))
        assert result.total_distance_m == pytest.approx(
            skid.distance_m + tools_roll, rel=2e-2
        )
        # The same closed form with UD's mu: identical law, larger
        # roll-out by exactly the mu ratio (documented difference).
        ud_roll = skid.exit_speed_mps**2 / (2.0 * _ud_rolling_mu(10.0) * GRAVITY_M_S2)
        mu_ratio = stimp_to_rolling_mu(10.0) / _ud_rolling_mu(10.0)
        assert ud_roll == pytest.approx(tools_roll * mu_ratio, rel=1e-12)
        assert ud_roll > tools_roll

    def test_uphill_downhill_rollout_asymmetry(self) -> None:
        downhill = _imported_grid(lambda x, y: -0.02 * x)
        uphill = _imported_grid(lambda x, y: 0.02 * x)
        flat = _imported_grid(lambda x, y: 0.0)
        distances = [
            simulate_putt_on_surface(
                _launch(1.5), surface, stimp_ft=10.0, hole_distance_m=30.0
            ).total_distance_m
            for surface in (downhill, flat, uphill)
        ]
        assert distances[0] > distances[1] > distances[2]

    def test_break_falls_toward_the_cross_slope_downhill_side(self) -> None:
        downhill_left = _imported_grid(lambda x, y: -0.02 * y)
        downhill_right = _imported_grid(lambda x, y: 0.02 * y)
        left = simulate_putt_on_surface(
            _launch(1.5), downhill_left, stimp_ft=10.0, hole_distance_m=30.0
        )
        right = simulate_putt_on_surface(
            _launch(1.5), downhill_right, stimp_ft=10.0, hole_distance_m=30.0
        )
        assert left.break_m > 0.0
        assert right.break_m < 0.0

    def test_rollout_monotone_in_launch_speed(self) -> None:
        surface = _imported_grid(lambda x, y: 0.0)
        distances = [
            simulate_putt_on_surface(
                _launch(speed), surface, stimp_ft=10.0, hole_distance_m=30.0
            ).total_distance_m
            for speed in (1.0, 1.5, 2.0)
        ]
        assert distances[0] < distances[1] < distances[2]

    def test_rollout_monotone_in_stimp(self) -> None:
        """Both mu laws decrease in stimp, so faster greens roll out
        farther in both engines."""
        surface = _imported_grid(lambda x, y: 0.0)
        distances = [
            simulate_putt_on_surface(
                _launch(1.5), surface, stimp_ft=stimp, hole_distance_m=30.0
            ).total_distance_m
            for stimp in (8.0, 10.0, 12.0)
        ]
        assert distances[0] < distances[1] < distances[2]
        assert _ud_rolling_mu(8.0) > _ud_rolling_mu(10.0) > _ud_rolling_mu(12.0)
