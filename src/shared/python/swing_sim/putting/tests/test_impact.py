"""Putter-ball impact tests (#4125 H3)."""

from __future__ import annotations

import math

import pytest

from shared.python.swing_sim.impact import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)
from shared.python.swing_sim.putting import (
    DEFAULT_PUTTER_COR,
    MINIMAL_PUTTERS,
    PutterSpec,
    clubhead_speed_from_backstroke,
    strike,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

BLADE = MINIMAL_PUTTERS["Blade Putter"]


class TestPutterSpec:
    def test_minimal_putters_are_plausible(self) -> None:
        for spec in MINIMAL_PUTTERS.values():
            assert 0.3 <= spec.head_mass_kg <= 0.4
            assert 2.0 <= spec.loft_deg <= 4.0
            assert spec.cor == DEFAULT_PUTTER_COR

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=5.0, loft_deg=3.0)
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=0.35, loft_deg=45.0)
        with pytest.raises(ValueError):
            PutterSpec(name="x", head_mass_kg=0.35, loft_deg=3.0, cor=1.2)


class TestStrike:
    def test_normal_transfer_matches_impulse_momentum(self) -> None:
        """Zero loft: pure 1-D COR impulse, no spin, no launch."""
        flat = PutterSpec(name="Flat", head_mass_kg=0.350, loft_deg=0.0)
        launch = strike(flat, 2.0)
        expected = (
            2.0
            * (1.0 + flat.cor)
            * flat.head_mass_kg
            / (flat.head_mass_kg + GOLF_BALL_MASS_KG)
        )
        assert launch.ball_speed_mps == pytest.approx(expected, rel=1e-12)
        assert launch.launch_angle_deg == pytest.approx(0.0, abs=1e-12)
        assert launch.spin_rad_s == pytest.approx(0.0, abs=1e-12)

    def test_lofted_strike_launches_up_with_backspin(self) -> None:
        launch = strike(BLADE, 2.0)
        assert launch.launch_angle_deg > 0.0
        assert launch.launch_angle_deg < 2.0 * BLADE.loft_deg
        assert launch.spin_rad_s < 0.0  # backspin, topspin-positive sign
        # 2/7 cap: surface backspin speed is (5/7) * v * sin(loft).
        u = 2.0 * math.sin(math.radians(BLADE.loft_deg))
        assert -launch.spin_rad_s * GOLF_BALL_RADIUS_M == pytest.approx(
            (5.0 / 7.0) * u, rel=1e-12
        )

    def test_smash_factor_is_physical(self) -> None:
        launch = strike(BLADE, 1.5)
        smash = launch.ball_speed_mps / 1.5
        assert 1.4 < smash < 1.7  # typical published putter smash ~1.5-1.7

    def test_forward_press_reduces_launch(self) -> None:
        pressed = strike(BLADE, 2.0, shaft_lean_deg=-2.0)
        neutral = strike(BLADE, 2.0)
        assert pressed.launch_angle_deg < neutral.launch_angle_deg
        assert pressed.effective_loft_deg == pytest.approx(1.0)

    def test_rejects_bad_speeds(self) -> None:
        with pytest.raises(ValueError):
            strike(BLADE, 0.0)
        with pytest.raises(ValueError):
            strike(BLADE, 50.0)
        with pytest.raises(ValueError):
            strike(BLADE, 2.0, shaft_lean_deg=-30.0)


class TestBackstrokeProxy:
    def test_pendulum_speed_scales_linearly_with_amplitude(self) -> None:
        v1 = clubhead_speed_from_backstroke(0.2)
        v2 = clubhead_speed_from_backstroke(0.4)
        assert v2 == pytest.approx(2.0 * v1, rel=1e-12)

    def test_matches_simple_pendulum_formula(self) -> None:
        v = clubhead_speed_from_backstroke(0.3, putter_length_m=0.889)
        assert v == pytest.approx(0.3 * math.sqrt(9.80665 / 0.889), rel=1e-12)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            clubhead_speed_from_backstroke(0.0)
        with pytest.raises(ValueError):
            clubhead_speed_from_backstroke(0.3, putter_length_m=3.0)
