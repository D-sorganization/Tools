"""Tests for gas_properties module — thermophysical property calculations.

Covers:
- Gas component database (properties lookup)
- Ideal gas heat capacity (Shomate equation)
- Mixture property calculations (molecular weight, Cp, density)
- Physical law validations (ideal gas, speed of sound)
- Domain helpers (temperature, pressure validation)
"""

from __future__ import annotations

from numpy.testing import assert_allclose
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_ideal_gas_density,
    calculate_mixture_cp,
    calculate_mixture_molecular_weight,
    calculate_speed_of_sound,
)

# ── Molecular Weight ─────────────────────────────────────────────────────


class TestMixtureMolecularWeight:
    """Test mixture molecular weight calculations."""

    def test_pure_h2(self) -> None:
        """Pure H2 should have MW ≈ 2.016."""
        mw = calculate_mixture_molecular_weight({"H2": 1.0})
        assert_allclose(mw, 2.016, rtol=0.01)

    def test_pure_n2(self) -> None:
        """Pure N2 should have MW ≈ 28.014."""
        mw = calculate_mixture_molecular_weight({"N2": 1.0})
        assert_allclose(mw, 28.014, rtol=0.01)

    def test_pure_co2(self) -> None:
        """Pure CO2 should have MW ≈ 44.01."""
        mw = calculate_mixture_molecular_weight({"CO2": 1.0})
        assert_allclose(mw, 44.01, rtol=0.01)

    def test_air_mixture(self) -> None:
        """Air (79% N2, 21% O2) should have MW ≈ 28.97."""
        mw = calculate_mixture_molecular_weight({"N2": 0.79, "O2": 0.21})
        assert_allclose(mw, 28.97, rtol=0.01)

    def test_syngas_typical(self) -> None:
        """Typical syngas mixture should give reasonable MW."""
        comp = {"H2": 0.3, "CO": 0.4, "CO2": 0.2, "N2": 0.1}
        mw = calculate_mixture_molecular_weight(comp)
        # MW should be between H2 (2) and CO2 (44)
        assert 10.0 < mw < 35.0

    def test_mole_fractions_sum_validation(self) -> None:
        """Mole fractions should approximately sum to 1."""
        comp = {"H2": 0.5, "CO": 0.5}
        mw = calculate_mixture_molecular_weight(comp)
        # H2 (2), CO (28): weighted = 0.5*2 + 0.5*28 = 15
        assert_allclose(mw, 15.0, rtol=0.05)


# ── Ideal Gas Heat Capacity ─────────────────────────────────────────────


class TestIdealGasCp:
    """Test heat capacity calculations."""

    def test_h2_cp_at_300k(self) -> None:
        """H2 Cp at 300 K should be approximately 28.8 J/(mol·K)."""
        cp = calculate_ideal_gas_cp("H2", 300.0)
        assert 25.0 < cp < 35.0

    def test_n2_cp_at_300k(self) -> None:
        """N2 Cp at 300 K should be approximately 29.1 J/(mol·K)."""
        cp = calculate_ideal_gas_cp("N2", 300.0)
        assert 28.0 < cp < 32.0

    def test_co2_cp_higher_than_n2(self) -> None:
        """CO2 should have higher Cp than N2 (more degrees of freedom)."""
        cp_co2 = calculate_ideal_gas_cp("CO2", 500.0)
        cp_n2 = calculate_ideal_gas_cp("N2", 500.0)
        assert cp_co2 > cp_n2

    def test_cp_increases_with_temperature(self) -> None:
        """Cp generally increases with temperature for polyatomic gases."""
        cp_300 = calculate_ideal_gas_cp("CO2", 300.0)
        cp_1000 = calculate_ideal_gas_cp("CO2", 1000.0)
        assert cp_1000 > cp_300


# ── Mixture Heat Capacity ───────────────────────────────────────────────


class TestMixtureCp:
    """Test mixture heat capacity calculations."""

    def test_pure_component_cp(self) -> None:
        """Mixture Cp of pure component should equal component Cp."""
        T = 500.0
        cp_mixture = calculate_mixture_cp({"N2": 1.0}, T)
        cp_pure = calculate_ideal_gas_cp("N2", T)
        assert_allclose(cp_mixture, cp_pure, rtol=1e-6)

    def test_mixture_cp_reasonable_range(self) -> None:
        """Syngas mixture Cp should be in reasonable range."""
        comp = {"H2": 0.3, "CO": 0.4, "CO2": 0.2, "N2": 0.1}
        cp = calculate_mixture_cp(comp, 500.0)
        assert 25.0 < cp < 50.0

    def test_mixture_cp_between_components(self) -> None:
        """Binary mixture Cp should be between the two pure Cp values."""
        T = 400.0
        cp_h2 = calculate_ideal_gas_cp("H2", T)
        cp_co2 = calculate_ideal_gas_cp("CO2", T)
        cp_mix = calculate_mixture_cp({"H2": 0.5, "CO2": 0.5}, T)

        low = min(cp_h2, cp_co2)
        high = max(cp_h2, cp_co2)
        assert low <= cp_mix <= high


# ── Heat Capacity Ratio (γ) ─────────────────────────────────────────────


class TestHeatCapacityRatio:
    """Test Cp/Cv ratio calculations."""

    def test_monatomic_gas_gamma(self) -> None:
        """Monatomic gases should have γ ≈ 5/3 ≈ 1.667.
        But gas_properties may not have monatomic gases. Test with N2 (diatomic).
        """
        gamma = calculate_heat_capacity_ratio({"N2": 1.0}, 300.0)
        # N2 at 300K: γ ≈ 1.4
        assert 1.3 < gamma < 1.5

    def test_gamma_physical_range(self) -> None:
        """γ should always be > 1 for any gas mixture."""
        comp = {"H2": 0.2, "CO": 0.3, "CO2": 0.3, "H2O": 0.2}
        gamma = calculate_heat_capacity_ratio(comp, 500.0)
        assert gamma > 1.0

    def test_gamma_decreases_with_complexity(self) -> None:
        """More complex molecules have lower γ (more degrees of freedom)."""
        gamma_n2 = calculate_heat_capacity_ratio({"N2": 1.0}, 300.0)
        gamma_co2 = calculate_heat_capacity_ratio({"CO2": 1.0}, 300.0)
        assert gamma_n2 > gamma_co2


# ── Ideal Gas Density ────────────────────────────────────────────────────


class TestIdealGasDensity:
    """Test ideal gas density calculations using PV=nRT."""

    def test_air_at_stp(self) -> None:
        """Air at STP: ρ ≈ 1.225 kg/m³."""
        # MW_air ≈ 28.97 kg/kmol, T=293.15 K, P=101325 Pa
        rho = calculate_ideal_gas_density(28.97, 293.15, 101325.0)
        assert_allclose(rho, 1.225, rtol=0.02)

    def test_density_increases_with_pressure(self) -> None:
        """Higher pressure → higher density (Boyle's law)."""
        rho_1atm = calculate_ideal_gas_density(28.97, 300.0, 101325.0)
        rho_2atm = calculate_ideal_gas_density(28.97, 300.0, 202650.0)
        assert_allclose(rho_2atm / rho_1atm, 2.0, rtol=0.01)

    def test_density_decreases_with_temperature(self) -> None:
        """Higher temperature → lower density (Charles's law)."""
        rho_300 = calculate_ideal_gas_density(28.97, 300.0, 101325.0)
        rho_600 = calculate_ideal_gas_density(28.97, 600.0, 101325.0)
        assert_allclose(rho_300 / rho_600, 2.0, rtol=0.01)

    def test_lighter_gas_lower_density(self) -> None:
        """H2 (MW=2) should be much less dense than N2 (MW=28)."""
        rho_h2 = calculate_ideal_gas_density(2.016, 300.0, 101325.0)
        rho_n2 = calculate_ideal_gas_density(28.014, 300.0, 101325.0)
        assert rho_n2 / rho_h2 > 10


# ── Speed of Sound ───────────────────────────────────────────────────────


class TestSpeedOfSound:
    """Test speed of sound calculations."""

    def test_air_at_room_temp(self) -> None:
        """Speed of sound in air at 293 K ≈ 343 m/s."""
        a = calculate_speed_of_sound({"N2": 0.79, "O2": 0.21}, 293.0)
        assert_allclose(a, 343.0, rtol=0.05)

    def test_speed_increases_with_temperature(self) -> None:
        """Speed of sound proportional to √T."""
        a_300 = calculate_speed_of_sound({"N2": 1.0}, 300.0)
        a_1200 = calculate_speed_of_sound({"N2": 1.0}, 1200.0)
        # Should approximately double (√4 = 2)
        assert_allclose(a_1200 / a_300, 2.0, rtol=0.1)

    def test_h2_faster_than_n2(self) -> None:
        """H2 (lighter) should have higher speed of sound than N2."""
        a_h2 = calculate_speed_of_sound({"H2": 1.0}, 300.0)
        a_n2 = calculate_speed_of_sound({"N2": 1.0}, 300.0)
        assert a_h2 > a_n2

    def test_speed_positive(self) -> None:
        """Speed of sound must always be positive."""
        comp = {"H2": 0.3, "CO": 0.4, "CO2": 0.3}
        a = calculate_speed_of_sound(comp, 500.0)
        assert a > 0
