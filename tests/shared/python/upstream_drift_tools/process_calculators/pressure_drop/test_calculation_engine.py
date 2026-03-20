"""Tests for pressure drop calculation engine standalone functions.

Tests cover friction factor correlations (laminar, Colebrook, Swamee-Jain,
Churchill, Haaland), flow regime classification, Darcy-Weisbach pressure
drop, elevation pressure drop, erosional velocity, and expansion factor.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.engine.pressure_drop_calculation_engine import (
    calculate_elevation_pressure_drop,
    calculate_erosional_velocity,
    calculate_expansion_factor,
    calculate_frictional_pressure_drop,
    classify_flow_regime,
    friction_factor_churchill,
    friction_factor_colebrook,
    friction_factor_haaland,
    friction_factor_laminar,
    friction_factor_swamee_jain,
    select_friction_factor_method,
)

# ─── Laminar friction factor ─────────────────────────────────


class TestFrictionFactorLaminar:
    def test_re_1000(self) -> None:
        f = friction_factor_laminar(1000)
        assert abs(f - 0.064) < 0.001

    def test_re_2000(self) -> None:
        f = friction_factor_laminar(2000)
        assert abs(f - 0.032) < 0.001

    def test_inversely_proportional(self) -> None:
        f1 = friction_factor_laminar(500)
        f2 = friction_factor_laminar(1000)
        assert abs(f1 / f2 - 2.0) < 0.01


# ─── Turbulent friction factor correlations ──────────────────


class TestTurbulentCorrelations:
    """All turbulent correlations should agree to within ~5% for typical pipe."""

    re = 100000
    eps_d = 0.001  # relative roughness

    def test_colebrook_positive(self) -> None:
        f = friction_factor_colebrook(self.re, self.eps_d)
        assert f > 0

    def test_swamee_jain_positive(self) -> None:
        f = friction_factor_swamee_jain(self.re, self.eps_d)
        assert f > 0

    def test_churchill_positive(self) -> None:
        f = friction_factor_churchill(self.re, self.eps_d)
        assert f > 0

    def test_haaland_positive(self) -> None:
        f = friction_factor_haaland(self.re, self.eps_d)
        assert f > 0

    def test_swamee_jain_close_to_colebrook(self) -> None:
        f_cole = friction_factor_colebrook(self.re, self.eps_d)
        f_sj = friction_factor_swamee_jain(self.re, self.eps_d)
        assert abs(f_sj - f_cole) / f_cole < 0.02

    def test_haaland_close_to_colebrook(self) -> None:
        f_cole = friction_factor_colebrook(self.re, self.eps_d)
        f_haa = friction_factor_haaland(self.re, self.eps_d)
        assert abs(f_haa - f_cole) / f_cole < 0.02

    def test_rougher_pipe_higher_friction(self) -> None:
        f_smooth = friction_factor_colebrook(100000, 0.0001)
        f_rough = friction_factor_colebrook(100000, 0.01)
        assert f_rough > f_smooth

    def test_laminar_fallback(self) -> None:
        """All methods should return laminar friction for Re < 2300."""
        f_cole = friction_factor_colebrook(1000, 0.001)
        f_sj = friction_factor_swamee_jain(1000, 0.001)
        f_haa = friction_factor_haaland(1000, 0.001)
        expected = 64 / 1000
        assert abs(f_cole - expected) < 0.01
        assert abs(f_sj - expected) < 0.01
        assert abs(f_haa - expected) < 0.01


# ─── Churchill (all regimes) ─────────────────────────────────


class TestChurchill:
    def test_laminar_range(self) -> None:
        f = friction_factor_churchill(500, 0.001)
        expected = 64 / 500
        assert abs(f - expected) / expected < 0.15

    def test_turbulent_range(self) -> None:
        f = friction_factor_churchill(100000, 0.001)
        assert 0.01 < f < 0.1

    def test_very_low_re(self) -> None:
        f = friction_factor_churchill(0.5, 0.001)
        assert f == 64  # Returns LAMINAR_FRICTION_CONSTANT for Re < 1


# ─── select_friction_factor_method ────────────────────────────


class TestSelectMethod:
    def test_colebrook(self) -> None:
        f = select_friction_factor_method("colebrook", 100000, 0.001)
        assert f > 0

    def test_swamee_jain(self) -> None:
        f = select_friction_factor_method("swamee-jain", 100000, 0.001)
        assert f > 0

    def test_swamee_jain_underscore(self) -> None:
        f = select_friction_factor_method("swamee_jain", 100000, 0.001)
        assert f > 0

    def test_churchill(self) -> None:
        f = select_friction_factor_method("churchill", 100000, 0.001)
        assert f > 0

    def test_haaland(self) -> None:
        f = select_friction_factor_method("haaland", 100000, 0.001)
        assert f > 0

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            select_friction_factor_method("bogus", 100000, 0.001)

    def test_case_insensitive(self) -> None:
        f = select_friction_factor_method("COLEBROOK", 100000, 0.001)
        assert f > 0


# ─── Flow regime classification ──────────────────────────────


class TestClassifyFlowRegime:
    def test_laminar(self) -> None:
        assert classify_flow_regime(500) == "laminar"

    def test_transitional(self) -> None:
        assert classify_flow_regime(3000) == "transitional"

    def test_turbulent(self) -> None:
        assert classify_flow_regime(50000) == "turbulent"

    def test_boundary_laminar(self) -> None:
        assert classify_flow_regime(2299) == "laminar"


# ─── Darcy-Weisbach pressure drop ────────────────────────────


class TestFrictionalPressureDrop:
    def test_basic(self) -> None:
        # ΔP = f × (L/D) × (ρV²/2)
        dp = calculate_frictional_pressure_drop(
            friction_factor=0.02, length=100, diameter=0.1, density=1.2, velocity=10
        )
        expected = 0.02 * (100 / 0.1) * 0.5 * 1.2 * 100
        assert abs(dp - expected) < 0.1

    def test_proportional_to_length(self) -> None:
        dp1 = calculate_frictional_pressure_drop(0.02, 50, 0.1, 1.2, 10)
        dp2 = calculate_frictional_pressure_drop(0.02, 100, 0.1, 1.2, 10)
        assert abs(dp2 / dp1 - 2.0) < 0.01

    def test_proportional_to_velocity_squared(self) -> None:
        dp1 = calculate_frictional_pressure_drop(0.02, 100, 0.1, 1.2, 5)
        dp2 = calculate_frictional_pressure_drop(0.02, 100, 0.1, 1.2, 10)
        assert abs(dp2 / dp1 - 4.0) < 0.01


# ─── Elevation pressure drop ─────────────────────────────────


class TestElevationPressureDrop:
    def test_upward_positive(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, 10.0)
        assert dp > 0

    def test_downward_negative(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, -10.0)
        assert dp < 0

    def test_zero_elevation(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, 0.0)
        assert dp == 0.0


# ─── Erosional velocity ──────────────────────────────────────


class TestErosionalVelocity:
    def test_continuous(self) -> None:
        v = calculate_erosional_velocity(1.2, "continuous")
        assert v > 0

    def test_intermittent_higher(self) -> None:
        v_cont = calculate_erosional_velocity(1.2, "continuous")
        v_int = calculate_erosional_velocity(1.2, "intermittent")
        assert v_int > v_cont

    def test_higher_density_lower_velocity(self) -> None:
        v_low = calculate_erosional_velocity(1.0)
        v_high = calculate_erosional_velocity(10.0)
        assert v_high < v_low


# ─── Expansion factor ────────────────────────────────────────


class TestExpansionFactor:
    def test_low_dp_near_one(self) -> None:
        y = calculate_expansion_factor(100000, 100, 0.02, 100)
        assert abs(y - 1.0) < 0.01

    def test_choked_zero(self) -> None:
        y = calculate_expansion_factor(100000, 100000, 0.02, 100)
        assert y == 0.0

    def test_between_zero_and_one(self) -> None:
        y = calculate_expansion_factor(100000, 30000, 0.02, 100)
        assert 0 < y <= 1.0
