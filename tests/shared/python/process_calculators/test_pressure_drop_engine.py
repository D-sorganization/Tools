"""Tests for the PressureDropCalculationEngine and related utilities.

Covers friction factor correlations, gas property calculations,
fitting K-factors, and Darcy-Weisbach pressure drop physics.

These tests validate the "source of truth" for all pressure drop
calculations consumed by UpstreamDrift via the shared library.
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
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.fitting_loss_coefficients import (
    calculate_fitting_pressure_drop,
    calculate_two_k_factor,
    equivalent_length_to_k,
    get_fitting_k_factor,
    get_multiple_fittings_k,
    k_to_equivalent_length,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    GAS_DATABASE,
    calculate_compressibility_factor,
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_ideal_gas_density,
    calculate_mixture_cp,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_wilke,
    calculate_speed_of_sound,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def syngas_composition() -> dict[str, float]:
    """Typical syngas composition (mole fractions normalised to 1)."""
    return {"H2": 0.30, "CO": 0.35, "CO2": 0.15, "N2": 0.15, "H2O": 0.05}


@pytest.fixture
def air_composition() -> dict[str, float]:
    return {"N2": 0.79, "O2": 0.21}


# ============================================================================
# FRICTION FACTOR TESTS
# ============================================================================


class TestFrictionFactorLaminar:
    """Hagen-Poiseuille f = 64/Re."""

    def test_canonical_value(self) -> None:
        f = friction_factor_laminar(1600.0)
        assert f == pytest.approx(64.0 / 1600.0, rel=1e-6)

    def test_positive_for_valid_re(self) -> None:
        assert friction_factor_laminar(500.0) > 0
        assert friction_factor_laminar(2000.0) > 0

    def test_decreases_with_increasing_re(self) -> None:
        f_low = friction_factor_laminar(500.0)
        f_high = friction_factor_laminar(2000.0)
        assert f_low > f_high

    @pytest.mark.parametrize("reynolds_number", [0.0, -100.0])
    def test_nonpositive_re_raises(self, reynolds_number: float) -> None:
        with pytest.raises(ValueError, match="Reynolds number must be positive"):
            friction_factor_laminar(reynolds_number)


class TestFrictionFactorColebrook:
    """Colebrook-White implicit correlation."""

    @pytest.mark.parametrize(
        "method", ["colebrook", "swamee-jain", "churchill", "haaland"]
    )
    def test_all_methods_positive(self, method: str) -> None:
        f = select_friction_factor_method(method, 50_000, 0.0002)
        assert f > 0

    def test_colebrook_swamee_close_for_turbulent(self) -> None:
        """Swamee-Jain is accurate within ~1% of Colebrook for turbulent flow."""
        re, eps = 100_000, 0.0001
        f_cb = friction_factor_colebrook(re, eps)
        f_sj = friction_factor_swamee_jain(re, eps)
        assert abs(f_cb - f_sj) / f_cb < 0.015  # within 1.5%

    def test_colebrook_haaland_close(self) -> None:
        """Haaland is accurate within ~2% of Colebrook."""
        re, eps = 80_000, 0.0002
        f_cb = friction_factor_colebrook(re, eps)
        f_haa = friction_factor_haaland(re, eps)
        assert abs(f_cb - f_haa) / f_cb < 0.02

    def test_churchill_smooth_pipe_turbulent(self) -> None:
        """Churchill converges to ~0.0085 for smooth pipe at Re=1e5."""
        f = friction_factor_churchill(1e5, 0.0)
        assert 0.005 < f < 0.02

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown friction factor method"):
            select_friction_factor_method("magic", 50_000, 0.0002)

    def test_colebrook_laminar_Re_falls_back(self) -> None:
        """Colebrook calls laminar formula for Re < 2300."""
        f_cb = friction_factor_colebrook(1000.0, 0.001)
        f_lam = friction_factor_laminar(1000.0)
        assert f_cb == pytest.approx(f_lam, rel=1e-6)

    def test_rougher_pipe_higher_friction_factor(self) -> None:
        """Higher relative roughness → higher friction factor in turbulent flow."""
        re = 100_000
        f_smooth = friction_factor_colebrook(re, 1e-5)
        f_rough = friction_factor_colebrook(re, 0.01)
        assert f_rough > f_smooth

    def test_churchill_zero_reynolds_raises(self) -> None:
        """Re=0 is unphysical; Churchill must raise, not return a flat constant.

        Regression test for issue #3868: the prior `Re < 1: return
        LAMINAR_FRICTION_CONSTANT` shortcut silently returned 64 for any Re
        in (0, 1), which is wrong for every value except Re=1 (Churchill is
        documented to be valid for all Re > 0, so it should just compute).
        """
        with pytest.raises(ValueError, match="Reynolds number must be positive"):
            friction_factor_churchill(0.0, 0.0002)

    def test_churchill_negative_reynolds_raises(self) -> None:
        with pytest.raises(ValueError, match="Reynolds number must be positive"):
            friction_factor_churchill(-100.0, 0.0002)

    def test_churchill_small_positive_reynolds_computes_not_flat(self) -> None:
        """Re in (0, 1) must use the real Churchill formula, not a flat 64."""
        f_half = friction_factor_churchill(0.5, 0.0002)
        f_tenth = friction_factor_churchill(0.1, 0.0002)
        assert f_half != pytest.approx(64.0)
        assert f_tenth != pytest.approx(64.0)
        assert f_half != pytest.approx(f_tenth)

    def test_colebrook_non_convergence_raises(self) -> None:
        """Colebrook must raise, not silently return an unconverged iterate.

        Regression test for issue #3868: previously logged a warning and
        returned the last (unconverged) f, giving a wrong ΔP with no error.
        max_iterations=0 forces immediate non-convergence.
        """
        with pytest.raises(ValueError, match="did not converge"):
            friction_factor_colebrook(100_000.0, 0.0002, max_iterations=0)


class TestClassifyFlowRegime:
    def test_laminar_below_2300(self) -> None:
        assert classify_flow_regime(1000.0) == "laminar"
        assert classify_flow_regime(2200.0) == "laminar"

    def test_transitional(self) -> None:
        regime = classify_flow_regime(3000.0)
        assert regime == "transitional"

    def test_turbulent_above_4000(self) -> None:
        assert classify_flow_regime(5000.0) == "turbulent"
        assert classify_flow_regime(1_000_000.0) == "turbulent"


# ============================================================================
# DARCY-WEISBACH PRESSURE DROP
# ============================================================================


class TestFrictionalPressureDrop:
    """ΔP = f × (L/D) × (ρV²/2)"""

    def test_basic_positive_result(self) -> None:
        dp = calculate_frictional_pressure_drop(
            friction_factor=0.02,
            length=100.0,
            diameter=0.1,
            density=1.2,
            velocity=10.0,
        )
        assert dp > 0

    def test_doubles_with_doubled_velocity_squared(self) -> None:
        """dp ∝ V² → doubling V quadruples ΔP."""
        base = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1.2, 10.0)
        doubled = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1.2, 20.0)
        assert doubled == pytest.approx(4 * base, rel=1e-6)

    def test_proportional_to_length(self) -> None:
        dp1 = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1.2, 10.0)
        dp2 = calculate_frictional_pressure_drop(0.02, 200.0, 0.1, 1.2, 10.0)
        assert dp2 == pytest.approx(2 * dp1, rel=1e-6)

    def test_exact_value(self) -> None:
        """Manual: f=0.02, L=100, D=0.1, ρ=1.2, V=10 → ΔP = 0.02*(100/0.1)*(0.5*1.2*100) = 1200.0 Pa"""
        dp = calculate_frictional_pressure_drop(0.02, 100.0, 0.1, 1.2, 10.0)
        assert dp == pytest.approx(1200.0, rel=1e-6)


class TestElevationPressureDrop:
    """ΔP = ρgh"""

    def test_upward_flow_positive(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, 10.0)
        assert dp > 0

    def test_downward_flow_negative(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, -10.0)
        assert dp < 0

    def test_zero_elevation(self) -> None:
        dp = calculate_elevation_pressure_drop(1.2, 0.0)
        assert dp == pytest.approx(0.0, abs=1e-9)


class TestErosionalVelocity:
    """V_e = C / √ρ"""

    def test_continuous_lower_than_intermittent(self) -> None:
        v_cont = calculate_erosional_velocity(1.2, "continuous")
        v_int = calculate_erosional_velocity(1.2, "intermittent")
        assert v_int > v_cont

    def test_higher_density_lower_limit(self) -> None:
        v_light = calculate_erosional_velocity(0.5, "continuous")
        v_dense = calculate_erosional_velocity(5.0, "continuous")
        assert v_light > v_dense

    def test_unknown_type_defaults_safely(self) -> None:
        v = calculate_erosional_velocity(1.2, "unknown_type")
        assert v > 0


class TestExpansionFactor:
    """Gas expansion factor Y: 0 < Y ≤ 1"""

    def test_near_incompressible_approaches_one(self) -> None:
        Y = calculate_expansion_factor(100_000.0, 100.0, 0.02, 50.0)
        assert Y == pytest.approx(1.0, abs=0.05)

    def test_high_pressure_drop_reduces_Y(self) -> None:
        Y = calculate_expansion_factor(100_000.0, 50_000.0, 0.02, 50.0)
        assert 0.0 <= Y < 1.0

    def test_zero_inlet_pressure_returns_one(self) -> None:
        Y = calculate_expansion_factor(0.0, 100.0, 0.02, 50.0)
        assert Y == pytest.approx(1.0)


# ============================================================================
# FITTING K-FACTORS
# ============================================================================


class TestFittingKFactors:
    def test_known_fitting_returns_value(self) -> None:
        k = get_fitting_k_factor("90_elbow_std")
        assert k == pytest.approx(0.75)

    def test_unknown_fitting_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_fitting_k_factor("nonexistent_fitting")

    def test_multiple_fittings_sum(self) -> None:
        k_total = get_multiple_fittings_k({"90_elbow_std": 2, "gate_valve_open": 1})
        expected = 2 * 0.75 + 1 * 0.15
        assert k_total == pytest.approx(expected, rel=1e-6)

    def test_k_to_equivalent_length_roundtrip(self) -> None:
        k = 0.5
        f = 0.02
        ld = k_to_equivalent_length(k, f)
        k_back = equivalent_length_to_k(ld, f)
        assert k_back == pytest.approx(k, rel=1e-6)

    def test_negative_friction_factor_raises(self) -> None:
        with pytest.raises(ValueError):
            k_to_equivalent_length(0.5, 0.0)

    def test_fitting_pressure_drop_positive(self) -> None:
        dp = calculate_fitting_pressure_drop(1.0, 1.2, 10.0)
        assert dp == pytest.approx(0.5 * 1.2 * 100.0)


class TestTwoKMethod:
    def test_turbulent_large_pipe(self) -> None:
        k = calculate_two_k_factor("90_elbow_std_2k", 100_000, 4.0)
        assert k > 0

    def test_laminar_higher_k_than_turbulent(self) -> None:
        """In laminar flow (low Re), K1/Re term dominates → higher K."""
        k_lam = calculate_two_k_factor("90_elbow_std_2k", 100, 4.0)
        k_turb = calculate_two_k_factor("90_elbow_std_2k", 200_000, 4.0)
        assert k_lam > k_turb

    def test_unknown_two_k_raises(self) -> None:
        with pytest.raises(ValueError):
            calculate_two_k_factor("nonexistent_2k", 50_000, 4.0)


# ============================================================================
# GAS PROPERTIES CALCULATIONS
# ============================================================================


class TestGasDatabaseIntegrity:
    def test_all_components_have_positive_mw(self) -> None:
        for name, props in GAS_DATABASE.items():
            assert props.molecular_weight > 0, f"{name}: MW must be > 0"

    def test_all_critical_temps_positive(self) -> None:
        for name, props in GAS_DATABASE.items():
            assert props.critical_temp > 0, f"{name}: Tc must be > 0"

    def test_h2_lightest_component(self) -> None:
        h2_mw = GAS_DATABASE["H2"].molecular_weight
        for name, props in GAS_DATABASE.items():
            if name != "H2":
                assert props.molecular_weight > h2_mw


class TestShoemateCp:
    @pytest.mark.parametrize("component", ["H2", "CO", "CO2", "CH4", "N2", "H2O"])
    def test_cp_positive_at_standard_conditions(self, component: str) -> None:
        cp = calculate_ideal_gas_cp(component, 298.15)
        assert cp > 0, f"{component}: Cp must be positive at 298 K"

    def test_cp_unknown_component_fallback(self) -> None:
        cp = calculate_ideal_gas_cp("XenonGas123", 500.0)
        assert cp > 0  # Falls back to Air properties


class TestMixtureProperties:
    def test_mw_weighted_average(self, syngas_composition: dict) -> None:
        mw = calculate_mixture_molecular_weight(syngas_composition)
        assert 5.0 < mw < 40.0  # Syngas MW is typically 10-30 kg/kmol

    def test_cp_mix_positive(self, syngas_composition: dict) -> None:
        cp = calculate_mixture_cp(syngas_composition, 800.0)
        assert cp > 0

    def test_gamma_physical_range(self, syngas_composition: dict) -> None:
        gamma = calculate_heat_capacity_ratio(syngas_composition, 800.0)
        assert 1.0 < gamma < 1.7

    def test_ideal_gas_density(self) -> None:
        """ρ = PM/RT — check at standard conditions for N2."""
        # N2: MW=28 kg/kmol, T=300K, P=101325 Pa
        # ρ = 101325*28 / (8314*300) ≈ 1.136 kg/m³
        rho = calculate_ideal_gas_density(28.014, 300.0, 101_325.0)
        assert rho == pytest.approx(1.136, rel=0.01)

    def test_z_factor_near_one_at_low_pressure(self, syngas_composition: dict) -> None:
        """At low pressure, all gasses approach ideal (Z ≈ 1)."""
        z = calculate_compressibility_factor(syngas_composition, 300.0, 101_325.0)
        assert z == pytest.approx(1.0, abs=0.15)

    def test_z_factor_bounded(self, syngas_composition: dict) -> None:
        """Z-factor must remain in [0.1, 1.5] as enforced by the engine."""
        z = calculate_compressibility_factor(syngas_composition, 500.0, 10e6)
        assert 0.1 <= z <= 1.5


class TestSpeedOfSound:
    def test_nitrogen_speed_of_sound(self, air_composition: dict) -> None:
        """Speed of sound in N2 at 20°C ≈ 349 m/s (ideal gas)."""
        a = calculate_speed_of_sound(air_composition, 293.15)
        assert 300 < a < 400

    def test_speed_increases_with_temperature(self, syngas_composition: dict) -> None:
        """a ∝ √T — higher temperature → higher speed."""
        a_cold = calculate_speed_of_sound(syngas_composition, 300.0)
        a_hot = calculate_speed_of_sound(syngas_composition, 1000.0)
        assert a_hot > a_cold


class TestWilkeViscosity:
    def test_viscosity_positive(self, syngas_composition: dict) -> None:
        mu = calculate_mixture_viscosity_wilke(syngas_composition, 800.0, 1e5)
        assert mu > 0

    def test_viscosity_in_typical_range(self, syngas_composition: dict) -> None:
        """Hot syngas viscosity should be ~20-50 µPa·s."""
        mu = calculate_mixture_viscosity_wilke(syngas_composition, 800.0, 1e5)
        assert 1e-5 < mu < 1e-4  # 10-100 µPa·s

    def test_viscosity_increases_with_temperature(
        self, syngas_composition: dict
    ) -> None:
        """Gas viscosity increases with temperature (unlike liquids)."""
        mu_low = calculate_mixture_viscosity_wilke(syngas_composition, 300.0, 1e5)
        mu_high = calculate_mixture_viscosity_wilke(syngas_composition, 1000.0, 1e5)
        assert mu_high > mu_low
