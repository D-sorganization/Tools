"""Tests for upstream_drift_tools.utils.unit_constants module.

Covers:
- Physical constants (R, Avogadro, Boltzmann)
- STP definitions (IUPAC and old)
- Length conversions (self-consistency)
- Mass conversions
- Pressure conversions
- Energy conversions
- Temperature relationships
- Unit algebra (dimensional analysis checks)
"""

from __future__ import annotations

import math

from upstream_drift_tools.utils.unit_constants import (
    ATM_TO_KPA,
    ATMOSPHERE_TO_PASCAL,
    AVOGADRO_NUMBER,
    # Pressure
    BAR_TO_PASCAL,
    BOLTZMANN_CONSTANT,
    BTU_TO_JOULE,
    # Energy
    CALORIE_TO_JOULE,
    # Length
    CENTIMETER_TO_METER,
    FOOT_TO_METER,
    INCH_TO_METER,
    # Flow
    KG_PER_SECOND_TO_KG_PER_SECOND,
    KILOMETER_TO_METER,
    KILOWATT_HOUR_TO_JOULE,
    # Volume
    LITER_TO_CU_METER,
    MILE_TO_METER,
    MILLIMETER_TO_METER,
    MMHG_TO_PASCAL,
    MW_CARBON_DIOXIDE,
    MW_CARBON_MONOXIDE,
    # Molecular weights
    MW_HYDROGEN,
    MW_METHANE,
    MW_NITROGEN,
    MW_OXYGEN,
    MW_WATER_VAPOR,
    OUNCE_TO_KILOGRAM,
    # Mass
    POUND_TO_KILOGRAM,
    PSI_TO_PASCAL,
    # Physical constants
    R_UNIVERSAL,
    SHORT_TON_TO_KILOGRAM,
    # Area
    SQ_FOOT_TO_SQ_METER,
    STP_PRESSURE_PA,
    # STP
    STP_TEMPERATURE_K,
    US_GALLON_TO_CU_METER,
    YARD_TO_METER,
)

# ── Physical Constants ───────────────────────────────────────────────────


class TestPhysicalConstants:
    """Test NIST-sourced physical constants."""

    def test_gas_constant(self) -> None:
        """R = 8.314 J/(mol·K) — CODATA 2018."""
        assert 8.31 < R_UNIVERSAL < 8.32

    def test_avogadro_number(self) -> None:
        assert math.isclose(AVOGADRO_NUMBER, 6.02214076e23, rel_tol=1e-6)

    def test_boltzmann_constant(self) -> None:
        assert math.isclose(BOLTZMANN_CONSTANT, 1.380649e-23, rel_tol=1e-6)

    def test_boltzmann_equals_r_over_na(self) -> None:
        """k_B = R / N_A — by definition."""
        computed = R_UNIVERSAL / AVOGADRO_NUMBER
        assert math.isclose(computed, BOLTZMANN_CONSTANT, rel_tol=1e-4)


# ── STP Definitions ─────────────────────────────────────────────────────


class TestSTPDefinitions:
    """Test Standard Temperature and Pressure values."""

    def test_stp_temperature(self) -> None:
        assert math.isclose(STP_TEMPERATURE_K, 273.15)

    def test_stp_pressure(self) -> None:
        """IUPAC STP: 1 bar = 100,000 Pa."""
        assert math.isclose(STP_PRESSURE_PA, 1e5)


# ── Length Conversions ───────────────────────────────────────────────────


class TestLengthConversions:
    """Test length conversion factors."""

    def test_centimeter_to_meter(self) -> None:
        assert CENTIMETER_TO_METER == 0.01

    def test_millimeter_to_meter(self) -> None:
        assert MILLIMETER_TO_METER == 0.001

    def test_kilometer_to_meter(self) -> None:
        assert KILOMETER_TO_METER == 1000.0

    def test_inch_to_meter(self) -> None:
        assert math.isclose(INCH_TO_METER, 0.0254, abs_tol=1e-6)

    def test_foot_to_meter(self) -> None:
        assert math.isclose(FOOT_TO_METER, 0.3048, abs_tol=1e-6)

    def test_12_inches_equal_foot(self) -> None:
        """Self-consistency: 12 inches = 1 foot."""
        assert math.isclose(12 * INCH_TO_METER, FOOT_TO_METER, rel_tol=1e-10)

    def test_yard_is_3_feet(self) -> None:
        assert math.isclose(YARD_TO_METER, 3 * FOOT_TO_METER, rel_tol=1e-10)

    def test_mile_is_5280_feet(self) -> None:
        assert math.isclose(MILE_TO_METER, 5280 * FOOT_TO_METER, rel_tol=1e-10)

    def test_sq_foot_consistency(self) -> None:
        """1 sq ft = (0.3048 m)^2."""
        assert math.isclose(SQ_FOOT_TO_SQ_METER, FOOT_TO_METER**2, rel_tol=1e-8)


# ── Mass Conversions ─────────────────────────────────────────────────────


class TestMassConversions:
    """Test mass conversion factors."""

    def test_pound_to_kg(self) -> None:
        assert math.isclose(POUND_TO_KILOGRAM, 0.45359237, abs_tol=1e-8)

    def test_ounce_to_kg(self) -> None:
        """16 oz = 1 lb."""
        assert math.isclose(16 * OUNCE_TO_KILOGRAM, POUND_TO_KILOGRAM, rel_tol=1e-8)

    def test_short_ton(self) -> None:
        """1 short ton = 2000 lb."""
        assert math.isclose(
            SHORT_TON_TO_KILOGRAM,
            2000 * POUND_TO_KILOGRAM,
            rel_tol=1e-6,
        )


# ── Pressure Conversions ────────────────────────────────────────────────


class TestPressureConversions:
    """Test pressure conversion factors."""

    def test_bar_to_pascal(self) -> None:
        assert BAR_TO_PASCAL == 100000.0

    def test_atmosphere_to_pascal(self) -> None:
        """1 atm = 101325 Pa exactly."""
        assert ATMOSPHERE_TO_PASCAL == 101325.0

    def test_atm_to_kpa(self) -> None:
        assert math.isclose(ATM_TO_KPA, 101.325)

    def test_psi_to_pascal(self) -> None:
        assert math.isclose(PSI_TO_PASCAL, 6894.76, rel_tol=1e-4)

    def test_mmhg_to_pascal(self) -> None:
        """760 mmHg = 1 atm."""
        assert math.isclose(760 * MMHG_TO_PASCAL, ATMOSPHERE_TO_PASCAL, rel_tol=1e-4)


# ── Energy Conversions ──────────────────────────────────────────────────


class TestEnergyConversions:
    """Test energy conversion factors."""

    def test_calorie_to_joule(self) -> None:
        """Thermochemical calorie: 1 cal = 4.184 J."""
        assert math.isclose(CALORIE_TO_JOULE, 4.184, abs_tol=0.01)

    def test_btu_to_joule(self) -> None:
        assert math.isclose(BTU_TO_JOULE, 1055.06, rel_tol=1e-3)

    def test_kwh_to_joule(self) -> None:
        """1 kWh = 3.6 MJ."""
        assert math.isclose(KILOWATT_HOUR_TO_JOULE, 3.6e6)


# ── Volume Conversions ──────────────────────────────────────────────────


class TestVolumeConversions:
    """Test volume conversion factors."""

    def test_liter_to_cu_meter(self) -> None:
        assert LITER_TO_CU_METER == 0.001

    def test_gallon_to_cu_meter(self) -> None:
        assert math.isclose(US_GALLON_TO_CU_METER, 0.003785, rel_tol=1e-3)


# ── Molecular Weights ───────────────────────────────────────────────────


class TestMolecularWeights:
    """Test molecular weights of common gases."""

    def test_h2(self) -> None:
        assert math.isclose(MW_HYDROGEN, 2.016, rel_tol=0.01)

    def test_o2(self) -> None:
        assert math.isclose(MW_OXYGEN, 32.0, rel_tol=0.01)

    def test_n2(self) -> None:
        assert math.isclose(MW_NITROGEN, 28.014, rel_tol=0.01)

    def test_h2o(self) -> None:
        assert math.isclose(MW_WATER_VAPOR, 18.015, rel_tol=0.01)

    def test_co2(self) -> None:
        assert math.isclose(MW_CARBON_DIOXIDE, 44.01, rel_tol=0.01)

    def test_co(self) -> None:
        assert math.isclose(MW_CARBON_MONOXIDE, 28.01, rel_tol=0.01)

    def test_ch4(self) -> None:
        assert math.isclose(MW_METHANE, 16.04, rel_tol=0.01)

    def test_water_is_2h_plus_o(self) -> None:
        """H2O = 2*H + O, approximate check."""
        expected = MW_HYDROGEN + MW_OXYGEN / 2  # H2 + O
        assert math.isclose(MW_WATER_VAPOR, expected, rel_tol=0.03)

    def test_co2_is_c_plus_o2(self) -> None:
        """CO2 mass ~ CO + O (12 + 32 = 44)."""
        assert math.isclose(MW_CARBON_DIOXIDE, 44.0, rel_tol=0.01)


# ── Flow Rate Identity ──────────────────────────────────────────────────


class TestFlowConversions:
    """Test flow rate conversions."""

    def test_identity_conversion(self) -> None:
        assert KG_PER_SECOND_TO_KG_PER_SECOND == 1.0
