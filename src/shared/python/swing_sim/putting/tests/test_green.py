"""Green simulation tests: slope, break, capture, energy (#4125 H3)."""

from __future__ import annotations

import pytest

from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import (
    HOLE_RADIUS_M,
    MINIMAL_PUTTERS,
    GreenConditions,
    capture_speed_mps,
    simulate_putt,
    solve_skid,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

BLADE = MINIMAL_PUTTERS["Blade Putter"]
FLAT_10 = GreenConditions(stimp_ft=10.0)


def putt(speed: float, green: GreenConditions = FLAT_10, hole: float = 3.0):
    return simulate_putt(strike(BLADE, speed), green, hole)


class TestCaptureBound:
    def test_derived_bound_value(self) -> None:
        """R sqrt(g / 2r) with the USGA hole and ball radii."""
        expected = HOLE_RADIUS_M * (9.80665 / (2.0 * GOLF_BALL_RADIUS_M)) ** 0.5
        assert capture_speed_mps() == pytest.approx(expected, rel=1e-12)
        assert capture_speed_mps() == pytest.approx(0.82, abs=0.01)

    def test_dying_putt_is_holed(self) -> None:
        """A putt that barely reaches the hole drops."""
        result = putt(1.6)
        assert result.holed
        assert result.margin_mps is not None and result.margin_mps > 0.0
        assert result.miss_distance_m is None

    def test_slammed_putt_runs_past(self) -> None:
        """Crossing the hole above the bound is not captured."""
        result = putt(3.2)
        assert not result.holed
        assert result.speed_at_hole_mps is not None
        assert result.speed_at_hole_mps > capture_speed_mps()
        assert result.miss_distance_m is not None
        assert result.total_distance_m > 3.0 + HOLE_RADIUS_M

    def test_short_putt_misses_short(self) -> None:
        result = putt(1.2)
        assert not result.holed
        assert result.speed_at_hole_mps is None
        assert result.miss_distance_m is not None
        assert result.total_distance_m < 3.0


class TestSkidRollTransition:
    def test_skid_matches_the_closed_form_on_a_flat_green(self) -> None:
        launch = strike(BLADE, 2.2)
        result = simulate_putt(launch, FLAT_10, 12.0)
        closed = solve_skid(
            launch.horizontal_speed_mps,
            launch.spin_rad_s,
            GOLF_BALL_RADIUS_M,
            FLAT_10.mu_slide,
        )
        assert result.skid_distance_m == pytest.approx(closed.distance_m, rel=5e-3)
        assert 0 < result.skid_end_index < len(result.path_x_m) - 1
        # Speed at the recorded transition sample matches the 5/7 exit.
        v_at_transition = result.speeds_mps[result.skid_end_index]
        assert v_at_transition == pytest.approx(closed.exit_speed_mps, rel=5e-3)

    def test_skid_fraction_is_a_minor_share(self) -> None:
        result = putt(2.0, hole=10.0)
        assert 0.0 < result.skid_fraction < 0.35


class TestEnergyAndDeterminism:
    def test_speed_is_monotone_non_increasing_on_a_flat_green(self) -> None:
        result = putt(2.0, hole=10.0)
        speeds = result.speeds_mps
        assert all(b <= a + 1e-12 for a, b in zip(speeds, speeds[1:], strict=False))

    def test_repeat_runs_are_identical(self) -> None:
        a = putt(2.0)
        b = putt(2.0)
        assert a == b


class TestSlopeAndBreak:
    def test_flat_straight_putt_has_no_break(self) -> None:
        result = putt(2.0, hole=10.0)
        assert result.break_m == pytest.approx(0.0, abs=1e-12)

    def test_cross_slope_breaks_toward_the_low_side(self) -> None:
        left_low = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0)
        result = putt(2.0, green=left_low, hole=10.0)
        assert result.break_m > 0.01  # breaks left (+y), toward downhill

    def test_mirror_aspect_mirrors_the_break(self) -> None:
        left = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=90.0)
        right = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=-90.0)
        a = putt(2.0, green=left, hole=10.0)
        b = putt(2.0, green=right, hole=10.0)
        assert a.break_m == pytest.approx(-b.break_m, rel=1e-9)
        assert a.total_distance_m == pytest.approx(b.total_distance_m, rel=1e-9)

    def test_downhill_rolls_farther_than_uphill(self) -> None:
        downhill = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=0.0)
        uphill = GreenConditions(stimp_ft=10.0, grade_percent=2.0, aspect_deg=180.0)
        down = putt(1.6, green=downhill, hole=20.0)
        up = putt(1.6, green=uphill, hole=20.0)
        assert down.total_distance_m > up.total_distance_m

    def test_faster_green_rolls_farther(self) -> None:
        fast = putt(1.6, green=GreenConditions(stimp_ft=13.0), hole=20.0)
        slow = putt(1.6, green=GreenConditions(stimp_ft=8.0), hole=20.0)
        assert fast.total_distance_m > slow.total_distance_m

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            GreenConditions(stimp_ft=1.0)
        with pytest.raises(ValueError):
            GreenConditions(stimp_ft=10.0, grade_percent=50.0)
        with pytest.raises(ValueError):
            simulate_putt(strike(BLADE, 2.0), FLAT_10, 0.0)
