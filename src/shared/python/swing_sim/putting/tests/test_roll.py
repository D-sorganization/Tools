"""Skid/roll and stimpmeter tests (#4125 H3)."""

from __future__ import annotations

import pytest

from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import (
    STIMP_RELEASE_SPEED_MPS,
    roll_out_distance,
    roll_time_s,
    rolling_mu_to_stimp,
    solve_skid,
    stimp_to_rolling_mu,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

FOOT_M = 0.3048


class TestStimpmeter:
    def test_release_speed_matches_the_quoted_value(self) -> None:
        """USGA geometry derivation lands on the quoted ~1.83 m/s."""
        assert STIMP_RELEASE_SPEED_MPS == pytest.approx(1.83, abs=0.01)

    def test_stimp_round_trip_through_the_roll_model(self) -> None:
        """Release speed rolled out on a stimp-S green travels S feet."""
        for stimp in (6.0, 8.5, 10.0, 12.0, 14.0):
            mu = stimp_to_rolling_mu(stimp)
            distance_ft = roll_out_distance(STIMP_RELEASE_SPEED_MPS, mu) / FOOT_M
            assert distance_ft == pytest.approx(stimp, rel=1e-12)

    def test_mu_stimp_inverse_pair(self) -> None:
        for stimp in (7.0, 10.0, 13.0):
            assert rolling_mu_to_stimp(stimp_to_rolling_mu(stimp)) == pytest.approx(
                stimp, rel=1e-12
            )

    def test_faster_green_means_lower_mu(self) -> None:
        assert stimp_to_rolling_mu(13.0) < stimp_to_rolling_mu(8.0)

    def test_stimp_ten_mu_is_in_the_published_band(self) -> None:
        assert 0.05 <= stimp_to_rolling_mu(10.0) <= 0.07

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            stimp_to_rolling_mu(1.0)
        with pytest.raises(ValueError):
            rolling_mu_to_stimp(0.5)


class TestSkid:
    def test_transition_continuity_v_equals_omega_r(self) -> None:
        """At the skid exit, v(t*) from kinematics equals omega(t*) r."""
        v0, mu = 2.0, 0.4
        spin0 = -50.0  # backspin [rad/s]
        sol = solve_skid(v0, spin0, GOLF_BALL_RADIUS_M, mu)
        g = 9.80665
        v_end = v0 - mu * g * sol.duration_s
        omega_r_end = spin0 * GOLF_BALL_RADIUS_M + 2.5 * mu * g * sol.duration_s
        assert v_end == pytest.approx(omega_r_end, rel=1e-12)
        assert sol.exit_speed_mps == pytest.approx(v_end, rel=1e-12)

    def test_no_spin_gives_the_classic_five_sevenths(self) -> None:
        sol = solve_skid(2.1, 0.0, GOLF_BALL_RADIUS_M)
        assert sol.exit_speed_mps == pytest.approx(2.1 * 5.0 / 7.0, rel=1e-12)

    def test_already_rolling_skids_zero(self) -> None:
        omega = 2.0 / GOLF_BALL_RADIUS_M  # rolling exactly at v = omega r
        sol = solve_skid(2.0, omega, GOLF_BALL_RADIUS_M)
        assert sol.duration_s == 0.0
        assert sol.distance_m == 0.0
        assert sol.exit_speed_mps == 2.0

    def test_backspin_extends_the_skid(self) -> None:
        clean = solve_skid(2.0, 0.0, GOLF_BALL_RADIUS_M)
        spun = solve_skid(2.0, -80.0, GOLF_BALL_RADIUS_M)
        assert spun.duration_s > clean.duration_s
        assert spun.exit_speed_mps < clean.exit_speed_mps

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError):
            solve_skid(-1.0, 0.0, GOLF_BALL_RADIUS_M)
        with pytest.raises(ValueError):
            solve_skid(2.0, 0.0, 0.5)
        with pytest.raises(ValueError):
            solve_skid(2.0, 0.0, GOLF_BALL_RADIUS_M, mu_slide=0.0)


class TestPureRoll:
    def test_stopping_distance_quadratic_in_speed(self) -> None:
        mu = stimp_to_rolling_mu(10.0)
        assert roll_out_distance(2.0, mu) == pytest.approx(
            4.0 * roll_out_distance(1.0, mu), rel=1e-12
        )

    def test_time_and_distance_are_consistent(self) -> None:
        """d = v t / 2 for constant deceleration to rest."""
        mu = stimp_to_rolling_mu(11.0)
        v = 1.7
        assert roll_out_distance(v, mu) == pytest.approx(
            0.5 * v * roll_time_s(v, mu), rel=1e-12
        )
