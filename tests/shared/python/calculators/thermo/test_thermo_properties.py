"""Tests for upstream_drift_tools.calculators.thermo.thermo_properties module.

Covers:
- ThermoPropertiesCalculator.calculate for pure gases
- Mixture calculations with normalization
- Ideal gas law (PV=nRT) verification
- Thermodynamic identity G = H - TS
- Temperature/pressure effects on density
- Heat capacity ratio gamma
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.calculators.thermo.thermo_properties import (
    MOLAR_CP_298,
    MOLECULAR_WEIGHTS,
    R_GAS,
    ThermoPropertiesCalculator,
    ThermoResult,
)


@pytest.fixture
def calc() -> ThermoPropertiesCalculator:
    return ThermoPropertiesCalculator()


# ── Pure Gas Calculations ────────────────────────────────────────────────


class TestPureGas:
    """Test calculations for single-component gases."""

    def test_nitrogen_at_stp(self, calc: ThermoPropertiesCalculator) -> None:
        result = calc.calculate(
            temperature_c=0.0,
            pressure_kpa=101.325,
            composition={"N2": 1.0},
        )
        assert isinstance(result, ThermoResult)
        assert result.temperature_k == pytest.approx(273.15)
        assert result.pressure_pa == pytest.approx(101325.0)
        assert result.molecular_weight_g_mol == pytest.approx(28.014)

    def test_ideal_gas_density_at_stp(self, calc: ThermoPropertiesCalculator) -> None:
        """N2 at STP should be about 1.25 kg/m3."""
        result = calc.calculate(
            temperature_c=0.0,
            pressure_kpa=101.325,
            composition={"N2": 1.0},
        )
        assert result.density_kg_m3 == pytest.approx(1.25, rel=0.05)

    def test_molar_volume_ideal_gas(self, calc: ThermoPropertiesCalculator) -> None:
        """PV=nRT => V = RT/P at STP ~ 0.02271 m^3/mol."""
        result = calc.calculate(
            temperature_c=0.0,
            pressure_kpa=101.325,
            composition={"N2": 1.0},
        )
        expected = R_GAS * 273.15 / 101325.0
        assert result.molar_volume_m3_mol == pytest.approx(expected, rel=1e-6)


# ── Mixture Calculations ────────────────────────────────────────────────


class TestMixture:
    """Test mixture property calculations."""

    def test_air_molecular_weight(self, calc: ThermoPropertiesCalculator) -> None:
        """Air (79% N2, 21% O2) MW ~ 28.97 g/mol."""
        result = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"N2": 79, "O2": 21},
        )
        assert result.molecular_weight_g_mol == pytest.approx(28.85, rel=0.01)

    def test_composition_normalization(self, calc: ThermoPropertiesCalculator) -> None:
        """Unnormalized fractions should be normalized internally."""
        result_100 = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"N2": 79, "O2": 21},
        )
        result_10 = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"N2": 7.9, "O2": 2.1},
        )
        assert result_100.density_kg_m3 == pytest.approx(
            result_10.density_kg_m3, rel=1e-10
        )

    def test_ternary_mixture(self, calc: ThermoPropertiesCalculator) -> None:
        """Three-component mixture should compute without error."""
        result = calc.calculate(
            temperature_c=500.0,
            pressure_kpa=200.0,
            composition={"CO2": 10, "H2O": 20, "N2": 70},
        )
        assert result.density_kg_m3 > 0
        assert result.gamma > 1.0


# ── Thermodynamic Identities ────────────────────────────────────────────


class TestThermodynamicIdentities:
    """Test consistency of computed properties."""

    def test_gibbs_identity(self, calc: ThermoPropertiesCalculator) -> None:
        """G = H - T*S must hold."""
        result = calc.calculate(
            temperature_c=500.0,
            pressure_kpa=101.325,
            composition={"N2": 1.0},
        )
        g_computed = (
            result.enthalpy_j_mol - result.temperature_k * result.entropy_j_molk
        )
        assert result.gibbs_energy_j_mol == pytest.approx(g_computed, rel=1e-10)

    def test_cv_equals_cp_minus_R(self, calc: ThermoPropertiesCalculator) -> None:
        """For ideal gas: Cv = Cp - R."""
        result = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"O2": 1.0},
        )
        assert result.cv_j_molk == pytest.approx(result.cp_j_molk - R_GAS, rel=1e-10)

    def test_gamma_positive_and_gt_1(self, calc: ThermoPropertiesCalculator) -> None:
        """Gamma = Cp/Cv > 1 for ideal gas."""
        result = calc.calculate(
            temperature_c=25.0,
            pressure_kpa=101.325,
            composition={"H2": 1.0},
        )
        assert result.gamma > 1.0


# ── Temperature & Pressure Effects ──────────────────────────────────────


class TestTemperaturePressureEffects:
    """Test that density responds correctly to T and P."""

    def test_density_decreases_with_temperature(
        self, calc: ThermoPropertiesCalculator
    ) -> None:
        r_low = calc.calculate(0.0, 101.325, {"N2": 1.0})
        r_high = calc.calculate(500.0, 101.325, {"N2": 1.0})
        assert r_high.density_kg_m3 < r_low.density_kg_m3

    def test_density_increases_with_pressure(
        self, calc: ThermoPropertiesCalculator
    ) -> None:
        r_low = calc.calculate(25.0, 101.325, {"N2": 1.0})
        r_high = calc.calculate(25.0, 500.0, {"N2": 1.0})
        assert r_high.density_kg_m3 > r_low.density_kg_m3

    def test_enthalpy_increases_with_temperature(
        self, calc: ThermoPropertiesCalculator
    ) -> None:
        r_low = calc.calculate(25.0, 101.325, {"N2": 1.0})
        r_high = calc.calculate(500.0, 101.325, {"N2": 1.0})
        assert r_high.enthalpy_j_mol > r_low.enthalpy_j_mol


# ── Reference Data / Lookup Tables ──────────────────────────────────────


class TestReferenceData:
    """Test the reference data tables."""

    def test_molecular_weights_populated(self) -> None:
        assert len(MOLECULAR_WEIGHTS) >= 10
        assert "N2" in MOLECULAR_WEIGHTS
        assert "H2O" in MOLECULAR_WEIGHTS

    def test_cp_values_populated(self) -> None:
        assert len(MOLAR_CP_298) >= 10
        for species, cp in MOLAR_CP_298.items():
            assert cp > 0, f"{species} has non-positive Cp"

    def test_r_gas_value(self) -> None:
        assert R_GAS == pytest.approx(8.314, rel=1e-3)
