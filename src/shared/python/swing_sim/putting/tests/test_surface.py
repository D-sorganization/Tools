"""Green surface, 2-D surface roll, capture model (#4800 P2).

Analytic gates first (flat -> straight, break sign vs cross-slope,
uphill/downhill asymmetry, capture window monotone in speed), then the
CRITICAL planar-limit regression: the legacy :func:`simulate_putt`
delegates to the surface integrator and must stay **bit-identical**.
The reference pins are mirrored value-for-value by the vitest suite in
``web/src/model/puttingGreen.test.ts``.
"""

from __future__ import annotations

import math

import pytest

from shared.python.swing_sim.putting import (
    HOLE_RADIUS_M,
    MINIMAL_PUTTERS,
    GreenConditions,
    GridGreenSurface,
    PlanarGreenSurface,
    capture_speed_mps,
    effective_hole_radius_m,
    green_surface_from_json,
    green_surface_to_json,
    simulate_putt,
    simulate_putt_on_surface,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

BLADE = MINIMAL_PUTTERS["Blade Putter"]
FLAT = PlanarGreenSurface(grade_percent=0.0, aspect_deg=0.0)


def _plane_grid(
    grade_percent: float,
    aspect_deg: float,
    origin: tuple[float, float] = (-2.0, -4.0),
    spacing: float = 0.5,
    nx: int = 41,
    ny: int = 17,
) -> GridGreenSurface:
    """Grid heightfield sampled from the parametric plane."""
    plane = PlanarGreenSurface(grade_percent=grade_percent, aspect_deg=aspect_deg)
    heights = tuple(
        tuple(
            plane.height_m(origin[0] + i * spacing, origin[1] + j * spacing)
            for i in range(nx)
        )
        for j in range(ny)
    )
    return GridGreenSurface(origin_m=origin, spacing_m=spacing, heights_m=heights)


class TestPlanarLimitRegression:
    """CRITICAL gate: the planar limit reproduces #4125 H3 exactly."""

    @pytest.mark.parametrize(
        ("clubhead", "green", "hole"),
        [
            (1.6, GreenConditions(stimp_ft=10.0), 3.0),
            (3.2, GreenConditions(stimp_ft=10.0), 3.0),
            (
                1.8,
                GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0),
                3.0,
            ),
            (
                1.6,
                GreenConditions(stimp_ft=13.0, grade_percent=2.0, aspect_deg=180.0),
                20.0,
            ),
            (
                2.0,
                GreenConditions(stimp_ft=8.0, grade_percent=3.0, aspect_deg=-45.0),
                10.0,
            ),
        ],
    )
    def test_legacy_api_equals_surface_integrator_bitwise(
        self, clubhead: float, green: GreenConditions, hole: float
    ) -> None:
        launch = strike(BLADE, clubhead)
        legacy = simulate_putt(launch, green, hole)
        surface = PlanarGreenSurface(
            grade_percent=green.grade_percent, aspect_deg=green.aspect_deg
        )
        direct = simulate_putt_on_surface(
            launch,
            surface,
            stimp_ft=green.stimp_ft,
            hole_distance_m=hole,
            mu_slide=green.mu_slide,
            capture_model="speed_threshold",
        )
        # Frozen-dataclass equality compares every trajectory sample —
        # a bit-level assertion, not a tolerance.
        assert legacy == direct

    def test_pre_4800_reference_pins_are_unchanged(self) -> None:
        """The #4125 H3 pinned reference putts, verbatim."""
        launch = strike(BLADE, 1.8)
        green = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0)
        result = simulate_putt(launch, green, 3.0)
        assert result.total_distance_m == pytest.approx(4.562853055205739, rel=1e-12)
        assert result.skid_distance_m == pytest.approx(0.49083593692889005, rel=1e-12)
        assert result.break_m == pytest.approx(0.8476231786308811, rel=1e-12)
        assert result.miss_distance_m == pytest.approx(1.6335909610224524, rel=1e-12)
        holed = simulate_putt(strike(BLADE, 1.6), GreenConditions(stimp_ft=10.0), 3.0)
        assert holed.holed
        assert holed.speed_at_hole_mps == pytest.approx(0.6746829587276968, rel=1e-12)
        assert holed.margin_mps == pytest.approx(0.1439566926681971, rel=1e-12)


class TestFlatStraight:
    def test_flat_planar_rolls_dead_straight(self) -> None:
        result = simulate_putt_on_surface(
            strike(BLADE, 2.0), FLAT, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert all(y == 0.0 for y in result.path_y_m)
        assert result.break_m == 0.0

    def test_flat_grid_matches_flat_planar_bitwise(self) -> None:
        grid = GridGreenSurface(
            origin_m=(-2.0, -2.0),
            spacing_m=1.0,
            heights_m=tuple(tuple(0.0 for _ in range(15)) for _ in range(5)),
        )
        launch = strike(BLADE, 2.0)
        on_grid = simulate_putt_on_surface(
            launch, grid, stimp_ft=10.0, hole_distance_m=10.0
        )
        on_plane = simulate_putt_on_surface(
            launch, FLAT, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert on_grid == on_plane
        assert all(y == 0.0 for y in on_grid.path_y_m)


class TestGridMatchesPlane:
    """Bilinear interpolation reproduces any plane (analytic identity)."""

    @pytest.mark.parametrize("aspect_deg", [90.0, -90.0, 0.0, 180.0, 37.0])
    def test_grid_sampled_from_plane_matches_parametric(
        self, aspect_deg: float
    ) -> None:
        launch = strike(BLADE, 2.0)
        plane = PlanarGreenSurface(grade_percent=2.0, aspect_deg=aspect_deg)
        grid = _plane_grid(2.0, aspect_deg)
        a = simulate_putt_on_surface(launch, plane, stimp_ft=10.0, hole_distance_m=10.0)
        b = simulate_putt_on_surface(launch, grid, stimp_ft=10.0, hole_distance_m=10.0)
        assert b.break_m == pytest.approx(a.break_m, abs=1e-9)
        assert b.total_distance_m == pytest.approx(a.total_distance_m, rel=1e-9)
        assert b.time_s == pytest.approx(a.time_s, abs=2e-3)
        assert b.holed == a.holed


class TestGridAnalyticGates:
    def test_cross_slope_breaks_toward_the_low_side(self) -> None:
        left_low = _plane_grid(2.0, 90.0)  # height falls toward +y (left)
        result = simulate_putt_on_surface(
            strike(BLADE, 2.0), left_low, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert result.break_m > 0.01

    def test_mirrored_cross_slope_mirrors_the_break(self) -> None:
        left = _plane_grid(2.0, 90.0)
        right = _plane_grid(2.0, -90.0, origin=(-2.0, -12.0), ny=33)
        a = simulate_putt_on_surface(
            strike(BLADE, 2.0), left, stimp_ft=10.0, hole_distance_m=10.0
        )
        b = simulate_putt_on_surface(
            strike(BLADE, 2.0), right, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert a.break_m == pytest.approx(-b.break_m, rel=1e-9)
        assert a.total_distance_m == pytest.approx(b.total_distance_m, rel=1e-9)

    def test_uphill_rolls_shorter_than_downhill_for_equal_launch(self) -> None:
        launch = strike(BLADE, 1.6)
        uphill = _plane_grid(2.0, 180.0, origin=(-2.0, -2.0), ny=9)
        downhill = _plane_grid(2.0, 0.0, origin=(-2.0, -2.0), ny=9)
        up = simulate_putt_on_surface(
            launch, uphill, stimp_ft=10.0, hole_distance_m=20.0
        )
        down = simulate_putt_on_surface(
            launch, downhill, stimp_ft=10.0, hole_distance_m=20.0
        )
        assert up.total_distance_m < down.total_distance_m

    def test_surface_continues_flat_beyond_the_grid_hull(self) -> None:
        # A short downhill patch: once the ball leaves it, only friction
        # acts, so the putt terminates well before the time cap.
        patch = tuple(
            tuple(-0.02 * (0.0 + i * 0.5) for i in range(5)) for _ in range(5)
        )
        surface = GridGreenSurface(origin_m=(0.0, -1.0), spacing_m=0.5, heights_m=patch)
        result = simulate_putt_on_surface(
            strike(BLADE, 2.0), surface, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert result.total_distance_m > 2.0  # rolled past the patch
        assert result.time_s < 60.0  # friction stopped it (no runaway)

    def test_repeat_runs_are_identical(self) -> None:
        grid = _plane_grid(2.0, 90.0)
        a = simulate_putt_on_surface(
            strike(BLADE, 2.0), grid, stimp_ft=10.0, hole_distance_m=10.0
        )
        b = simulate_putt_on_surface(
            strike(BLADE, 2.0), grid, stimp_ft=10.0, hole_distance_m=10.0
        )
        assert a == b


class TestCaptureModel:
    def test_effective_radius_limits(self) -> None:
        v_c = capture_speed_mps()
        assert effective_hole_radius_m(0.0) == HOLE_RADIUS_M
        assert effective_hole_radius_m(v_c) == 0.0
        assert effective_hole_radius_m(2.0 * v_c) == 0.0

    def test_effective_radius_is_strictly_monotone_in_speed(self) -> None:
        v_c = capture_speed_mps()
        speeds = [k * v_c / 40.0 for k in range(41)]
        radii = [effective_hole_radius_m(v) for v in speeds]
        assert all(b < a for a, b in zip(radii, radii[1:], strict=False))

    def test_reference_radii_pinned(self) -> None:
        """Cross-runtime pins (mirrored in puttingGreen.test.ts)."""
        assert effective_hole_radius_m(0.5) == pytest.approx(
            0.04275766281973086, rel=1e-12
        )
        assert effective_hole_radius_m(0.8) == pytest.approx(
            0.011457634498570566, rel=1e-12
        )

    def test_capture_window_is_nested_within_the_threshold_window(self) -> None:
        """Monotone gate: every effective-model make also makes under
        the (weaker) legacy threshold; the fast edge shrinks."""
        effective = set()
        threshold = set()
        for k in range(80):
            clubhead = 1.40 + 0.005 * k
            launch = strike(BLADE, clubhead)
            if simulate_putt_on_surface(
                launch, FLAT, stimp_ft=10.0, hole_distance_m=3.0
            ).holed:
                effective.add(k)
            if simulate_putt_on_surface(
                launch,
                FLAT,
                stimp_ft=10.0,
                hole_distance_m=3.0,
                capture_model="speed_threshold",
            ).holed:
                threshold.add(k)
        assert effective, "the dying-pace window must exist"
        assert effective <= threshold
        assert len(effective) < len(threshold)

    def test_dying_putt_is_holed_under_both_models(self) -> None:
        launch = strike(BLADE, 1.6)
        assert simulate_putt_on_surface(
            launch, FLAT, stimp_ft=10.0, hole_distance_m=3.0
        ).holed
        assert simulate_putt_on_surface(
            launch,
            FLAT,
            stimp_ft=10.0,
            hole_distance_m=3.0,
            capture_model="speed_threshold",
        ).holed

    def test_firm_edge_pace_discriminates_the_models(self) -> None:
        """Near the capture bound the shrunken hole rejects the pass
        the flat threshold accepted."""
        launch = strike(BLADE, 1.66)
        effective = simulate_putt_on_surface(
            launch, FLAT, stimp_ft=10.0, hole_distance_m=3.0
        )
        legacy = simulate_putt_on_surface(
            launch,
            FLAT,
            stimp_ft=10.0,
            hole_distance_m=3.0,
            capture_model="speed_threshold",
        )
        assert legacy.holed
        assert not effective.holed

    def test_reference_effective_capture_pins(self) -> None:
        """Cross-runtime pins (mirrored in puttingGreen.test.ts)."""
        result = simulate_putt_on_surface(
            strike(BLADE, 1.6), FLAT, stimp_ft=10.0, hole_distance_m=3.0
        )
        assert result.holed
        assert result.speed_at_hole_mps == pytest.approx(0.6746829587276968, rel=1e-12)
        assert result.margin_mps == pytest.approx(0.1439566926681971, rel=1e-12)
        assert result.total_distance_m == pytest.approx(2.9687052346196463, rel=1e-9)
        assert result.time_s == pytest.approx(2.276, abs=2e-3)

    def test_reference_grid_cross_slope_pins(self) -> None:
        """Cross-runtime pins (mirrored in puttingGreen.test.ts)."""
        result = simulate_putt_on_surface(
            strike(BLADE, 2.0),
            _plane_grid(2.0, 90.0),
            stimp_ft=10.0,
            hole_distance_m=10.0,
        )
        assert not result.holed
        assert result.break_m == pytest.approx(1.0478455745462154, rel=1e-9)
        assert result.total_distance_m == pytest.approx(5.639840165062302, rel=1e-9)
        assert result.skid_distance_m == pytest.approx(0.6049655102874043, rel=1e-9)
        assert result.miss_distance_m == pytest.approx(4.684540514279886, rel=1e-9)

    def test_unknown_capture_model_is_refused(self) -> None:
        with pytest.raises(ValueError):
            simulate_putt_on_surface(
                strike(BLADE, 1.6),
                FLAT,
                stimp_ft=10.0,
                hole_distance_m=3.0,
                capture_model="lip_out",  # type: ignore[arg-type]
            )

    def test_negative_or_non_finite_speed_is_refused(self) -> None:
        with pytest.raises(ValueError):
            effective_hole_radius_m(-0.1)
        with pytest.raises(ValueError):
            effective_hole_radius_m(math.nan)


class TestSurfaceValidation:
    def test_planar_ranges(self) -> None:
        with pytest.raises(ValueError):
            PlanarGreenSurface(grade_percent=-0.1, aspect_deg=0.0)
        with pytest.raises(ValueError):
            PlanarGreenSurface(grade_percent=50.0, aspect_deg=0.0)
        with pytest.raises(ValueError):
            PlanarGreenSurface(grade_percent=2.0, aspect_deg=361.0)

    def test_grid_shape_and_values(self) -> None:
        flat_row = (0.0, 0.0, 0.0)
        with pytest.raises(ValueError):  # too few rows
            GridGreenSurface(origin_m=(0.0, 0.0), spacing_m=1.0, heights_m=(flat_row,))
        with pytest.raises(ValueError):  # ragged rows
            GridGreenSurface(
                origin_m=(0.0, 0.0),
                spacing_m=1.0,
                heights_m=(flat_row, (0.0, 0.0)),
            )
        with pytest.raises(ValueError):  # non-finite height
            GridGreenSurface(
                origin_m=(0.0, 0.0),
                spacing_m=1.0,
                heights_m=(flat_row, (0.0, math.nan, 0.0)),
            )
        with pytest.raises(ValueError):  # spacing out of range
            GridGreenSurface(
                origin_m=(0.0, 0.0), spacing_m=0.001, heights_m=(flat_row, flat_row)
            )
        with pytest.raises(ValueError):  # implausible 50 % local grade
            GridGreenSurface(
                origin_m=(0.0, 0.0),
                spacing_m=1.0,
                heights_m=(flat_row, (0.0, 0.5, 0.0)),
            )

    def test_surface_type_is_required(self) -> None:
        with pytest.raises(ValueError):
            simulate_putt_on_surface(
                strike(BLADE, 1.6),
                GreenConditions(stimp_ft=10.0),  # type: ignore[arg-type]
                stimp_ft=10.0,
                hole_distance_m=3.0,
            )


class TestWire:
    PLANAR_JSON = (
        '{"aspect_deg":90.0,"format":"swing_sim.green_surface/1",'
        '"grade_percent":2.5,"kind":"planar"}'
    )

    def test_planar_round_trip_is_byte_identical(self) -> None:
        surface = PlanarGreenSurface(grade_percent=2.5, aspect_deg=90.0)
        text = green_surface_to_json(surface)
        assert text == self.PLANAR_JSON  # sorted keys, compact, pinned
        parsed = green_surface_from_json(text)
        assert parsed == surface
        assert green_surface_to_json(parsed) == text

    def test_grid_round_trip_is_byte_identical(self) -> None:
        surface = GridGreenSurface(
            origin_m=(-1.0, -1.5),
            spacing_m=0.5,
            heights_m=((0.0, 0.01, 0.02), (0.0, 0.005, 0.01)),
        )
        text = green_surface_to_json(surface)
        parsed = green_surface_from_json(text)
        assert parsed == surface
        assert green_surface_to_json(parsed) == text

    def test_unknown_fields_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json(
                self.PLANAR_JSON.replace('"kind"', '"stimp_ft":10.0,"kind"')
            )

    def test_missing_fields_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json(
                '{"format":"swing_sim.green_surface/1","kind":"planar",'
                '"grade_percent":2.5}'
            )

    def test_cross_kind_fields_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json(
                '{"aspect_deg":90.0,"format":"swing_sim.green_surface/1",'
                '"grade_percent":2.5,"kind":"planar","spacing_m":0.5}'
            )

    def test_wrong_format_and_kind_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json(self.PLANAR_JSON.replace("/1", "/2"))
        with pytest.raises(ValueError):
            green_surface_from_json(self.PLANAR_JSON.replace('"planar"', '"mesh"'))

    def test_non_finite_and_boolean_numbers_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json(self.PLANAR_JSON.replace("2.5", "NaN"))
        with pytest.raises(TypeError):
            green_surface_from_json(self.PLANAR_JSON.replace("2.5", "true"))

    def test_non_object_payloads_are_refused(self) -> None:
        with pytest.raises(ValueError):
            green_surface_from_json("[1,2,3]")
