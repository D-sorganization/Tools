#!/usr/bin/env python3
"""Comprehensive tests for flow_rate_converter module.

Tests all conversion functions for mass, molar, volumetric (actual and standard) flows.
This module provides 147 lines of testable logic.
"""

import pytest
from upstream_drift_tools.calculators.conversion.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    R_UNIVERSAL,
    STANDARD_CONDITIONS,
    acfm_to_scfm,
    convert_flow_rate_to_mass,
    mass_to_molar,
    mass_to_standard_volumetric,
    mass_to_volumetric_actual,
    molar_to_mass,
    molar_to_molar,
    scfm_to_acfm,
    standard_volumetric_to_mass,
    volumetric_actual_to_mass,
)


class TestMolarToMolar:
    """Test molar flow rate conversions."""

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            (100.0, "kmol/s", "kmol/s", 100.0),
            (50.0, "mol/h", "mol/h", 50.0),
            (1.0, "mol/s", "mol/h", 3600.0),
            (1.0, "kmol/h", "mol/s", 1000.0 / 3600.0),
            (1.0, "lbmol/h", "mol/s", 453.59237 / 3600.0),
        ],
        ids=[
            "same-kmol/s",
            "same-mol/h",
            "mol/s-to-mol/h",
            "kmol/h-to-mol/s",
            "lbmol/h-to-mol/s",
        ],
    )
    def test_molar_conversions(self, value, from_unit, to_unit, expected):
        """Test molar flow rate unit conversions."""
        result = molar_to_molar(value, from_unit, to_unit)
        assert result == pytest.approx(expected, rel=1e-5)


class TestMassToMolar:
    """Test mass to molar flow conversions."""

    def test_kg_per_s_to_mol_per_s_co2(self):
        """Test conversion from kg/s to mol/s for CO2."""
        mw_co2 = 44.01  # kg/kmol
        mass_flow = 44.01  # kg/s
        result = mass_to_molar(mass_flow, "kg/s", mw_co2, "mol/s")
        # 44.01 kg/s / 44.01 kg/kmol = 1 kmol/s = 1000 mol/s
        expected = 1000.0
        assert result == pytest.approx(expected, rel=1e-5)

    def test_lb_per_h_to_kmol_per_h_water(self):
        """Test conversion from lb/h to kmol/h for water."""
        mw_h2o = 18.015  # kg/kmol
        mass_flow_lb_h = 100.0
        result = mass_to_molar(mass_flow_lb_h, "lb/h", mw_h2o, "kmol/h")
        # Should be positive
        assert result > 0

    def test_zero_molecular_weight_raises_error(self):
        """Test that zero molecular weight raises ValueError (DbC precondition)."""
        with pytest.raises(ValueError, match="molecular_weight must be positive"):
            mass_to_molar(100.0, "kg/s", 0.0, "mol/s")

    def test_unknown_mass_unit_raises_error(self):
        """Test unknown source unit fails with ValueError instead of KeyError."""
        with pytest.raises(ValueError, match="Unknown mass flow unit"):
            mass_to_molar(100.0, "bad_unit", 44.0, "mol/s")

    def test_unknown_molar_target_unit_raises_error(self):
        """Test unknown target unit fails with ValueError instead of KeyError."""
        with pytest.raises(ValueError, match="Unknown molar flow unit"):
            mass_to_molar(100.0, "kg/s", 44.0, "bad_unit")


class TestMolarToMass:
    """Test molar to mass flow conversions."""

    def test_mol_per_s_to_kg_per_s_nitrogen(self):
        """Test conversion from mol/s to kg/s for N2."""
        mw_n2 = 28.014  # kg/kmol
        molar_flow = 1000.0  # mol/s = 1 kmol/s
        result = molar_to_mass(molar_flow, "mol/s", mw_n2, "kg/s")
        # 1 kmol/s * 28.014 kg/kmol = 28.014 kg/s
        expected = 28.014
        assert result == pytest.approx(expected, rel=1e-4)

    def test_kmol_per_h_to_lb_per_h_methane(self):
        """Test conversion from kmol/h to lb/h for CH4."""
        mw_ch4 = 16.04  # kg/kmol
        molar_flow_kmol_h = 10.0
        result = molar_to_mass(molar_flow_kmol_h, "kmol/h", mw_ch4, "lb/h")
        # Should be positive
        assert result > 0

    def test_unknown_molar_source_unit_raises_error(self):
        """Test unknown source unit fails with ValueError instead of KeyError."""
        with pytest.raises(ValueError, match="Unknown molar flow unit"):
            molar_to_mass(100.0, "bad_unit", 44.0, "kg/s")

    def test_unknown_mass_target_unit_raises_error(self):
        """Test unknown target unit fails with ValueError instead of KeyError."""
        with pytest.raises(ValueError, match="Unknown mass flow unit"):
            molar_to_mass(100.0, "mol/s", 44.0, "bad_unit")


class TestVolumetricActualToMass:
    """Test actual volumetric flow to mass conversions using density."""

    def test_m3_per_s_to_kg_per_s(self):
        """Test conversion from m³/s to kg/s."""
        vol_flow = 10.0  # m³/s
        density = 1.2  # kg/m³ (typical air)
        result = volumetric_actual_to_mass(vol_flow, "m3/s", density, "kg/s")
        expected = 10.0 * 1.2
        assert result == pytest.approx(expected)

    def test_cfm_to_kg_per_h(self):
        """Test conversion from CFM to kg/h."""
        vol_flow = 1000.0  # CFM
        density = 1.2  # kg/m³
        result = volumetric_actual_to_mass(vol_flow, "CFM", density, "kg/h")
        # Should be positive
        assert result > 0

    def test_higher_density_gives_higher_mass(self):
        """Test that higher density increases mass flow."""
        vol_flow = 100.0
        density_low = 1.0
        density_high = 2.0

        result_low = volumetric_actual_to_mass(vol_flow, "m3/h", density_low, "kg/h")
        result_high = volumetric_actual_to_mass(vol_flow, "m3/h", density_high, "kg/h")

        assert result_high == pytest.approx(result_low * 2.0, rel=0.01)


class TestMassToVolumetricActual:
    """Test mass to actual volumetric flow conversions."""

    def test_kg_per_s_to_m3_per_s(self):
        """Test conversion from kg/s to m³/s."""
        mass_flow = 12.0  # kg/s
        density = 1.2  # kg/m³
        result = mass_to_volumetric_actual(mass_flow, "kg/s", density, "m3/s")
        expected = 12.0 / 1.2
        assert result == pytest.approx(expected)

    def test_roundtrip_conversion(self):
        """Test that mass->volume->mass returns original value."""
        mass_flow = 5.0  # kg/s
        density = 1.5  # kg/m³

        vol_flow = mass_to_volumetric_actual(mass_flow, "kg/s", density, "m3/s")
        mass_back = volumetric_actual_to_mass(vol_flow, "m3/s", density, "kg/s")

        assert mass_back == pytest.approx(mass_flow, rel=1e-10)


class TestStandardVolumetricToMass:
    """Test standard volumetric flow to mass conversions."""

    def test_scfm_to_kg_per_s_air(self):
        """Test conversion from SCFM to kg/s for air."""
        mw_air = 28.964  # kg/kmol
        scfm = 1000.0
        result = standard_volumetric_to_mass(
            scfm, "SCFM", mw_air, standard="SCFM", mass_unit="kg/s"
        )
        # Should be positive and reasonable
        assert result > 0
        assert result < 100

    def test_nm3_per_h_to_kg_per_h_nitrogen(self):
        """Test conversion from Nm³/h to kg/h for nitrogen."""
        mw_n2 = 28.014
        nm3_h = 100.0
        result = standard_volumetric_to_mass(
            nm3_h, "Nm³/h", mw_n2, standard="STP", mass_unit="kg/h"
        )
        # Should be positive
        assert result > 0

    def test_different_standards_give_different_results(self):
        """Test that different standard conditions give different results."""
        mw = 44.01  # CO2
        vol_flow = 1000.0

        result_stp = standard_volumetric_to_mass(
            vol_flow, "SCFM", mw, standard="STP", mass_unit="kg/s"
        )
        result_ntp = standard_volumetric_to_mass(
            vol_flow, "SCFM", mw, standard="NTP", mass_unit="kg/s"
        )

        # Should be different but close (different T, same P)
        assert result_stp != result_ntp
        assert abs(result_stp - result_ntp) / result_stp < 0.1

    def test_unknown_mass_target_unit_raises_error(self):
        """Test unknown mass target unit reports domain error."""
        with pytest.raises(ValueError, match="Unknown mass flow unit"):
            standard_volumetric_to_mass(
                100.0, "SCFM", 29.0, standard="STP", mass_unit="bad_unit"
            )


class TestMassToStandardVolumetric:
    """Test mass to standard volumetric flow conversions."""

    def test_kg_per_s_to_scfm_air(self):
        """Test conversion from kg/s to SCFM for air."""
        mw_air = 28.964
        mass_flow = 1.0
        result = mass_to_standard_volumetric(
            mass_flow, "kg/s", mw_air, standard="SCFM", vol_unit="SCFM"
        )
        assert result > 0

    def test_roundtrip_standard_conversion(self):
        """Test that mass->std vol->mass returns original value."""
        mw = 18.015  # H2O
        mass_flow = 3.0

        std_vol = mass_to_standard_volumetric(
            mass_flow, "kg/s", mw, standard="STP", vol_unit="Nm³/h"
        )
        mass_back = standard_volumetric_to_mass(
            std_vol, "Nm³/h", mw, standard="STP", mass_unit="kg/s"
        )

        assert mass_back == pytest.approx(mass_flow, rel=1e-8)

    def test_unknown_mass_source_unit_raises_error(self):
        """Test unknown source unit reports domain error."""
        with pytest.raises(ValueError, match="Unknown mass flow unit"):
            mass_to_standard_volumetric(
                10.0, "bad_unit", 29.0, standard="STP", vol_unit="Nm3/h"
            )


class TestSCFMToACFM:
    """Test SCFM to ACFM conversions."""

    def test_scfm_to_acfm_at_standard_conditions(self):
        """Test that at standard conditions, SCFM ≈ ACFM."""
        scfm = 1000.0
        # At SCFM standard conditions (60°F, 1 atm)
        temp_k = 288.71
        pressure_pa = 101325.0
        result = scfm_to_acfm(scfm, temp_k, pressure_pa, standard="SCFM")
        # Should be very close to input
        assert result == pytest.approx(scfm, rel=0.01)

    def test_scfm_to_acfm_high_temp_increases_volume(self):
        """Test that higher temperature increases ACFM."""
        scfm = 1000.0
        pressure_pa = 101325.0
        temp_low = 273.15
        temp_high = 373.15

        acfm_low = scfm_to_acfm(scfm, temp_low, pressure_pa, standard="STP")
        acfm_high = scfm_to_acfm(scfm, temp_high, pressure_pa, standard="STP")

        assert acfm_high > acfm_low

    def test_scfm_to_acfm_high_pressure_decreases_volume(self):
        """Test that higher pressure decreases ACFM."""
        scfm = 1000.0
        temp_k = 300.0
        pressure_low = 101325.0
        pressure_high = 1013250.0

        acfm_low = scfm_to_acfm(scfm, temp_k, pressure_low, standard="STP")
        acfm_high = scfm_to_acfm(scfm, temp_k, pressure_high, standard="STP")

        assert acfm_high < acfm_low


class TestACFMToSCFM:
    """Test ACFM to SCFM conversions."""

    def test_acfm_to_scfm_roundtrip(self):
        """Test that SCFM->ACFM->SCFM returns original value."""
        scfm_orig = 500.0
        temp_k = 320.0
        pressure_pa = 150000.0

        acfm = scfm_to_acfm(scfm_orig, temp_k, pressure_pa, standard="STP")
        scfm_back = acfm_to_scfm(acfm, temp_k, pressure_pa, standard="STP")

        assert scfm_back == pytest.approx(scfm_orig, rel=1e-10)

    def test_acfm_to_scfm_at_elevated_conditions(self):
        """Test ACFM to SCFM at non-standard conditions."""
        acfm = 1000.0
        temp_k = 350.0  # Higher than standard
        pressure_pa = 200000.0  # Higher than standard

        scfm = acfm_to_scfm(acfm, temp_k, pressure_pa, standard="STP")
        # Higher P and T partially cancel, but SCFM should be different from ACFM
        assert scfm != acfm


class TestConvertFlowRateToMass:
    """Test the unified convert_flow_rate_to_mass function."""

    def test_molar_to_mass_conversion(self):
        """Test molar to mass conversion."""
        mw = 44.01  # CO2
        result = convert_flow_rate_to_mass(
            value=1.0, from_unit="kmol/h", molecular_weight=mw
        )
        # 1 kmol/h = 1000 mol/h = 1000/3600 mol/s
        mol_s = 1000.0 / 3600.0
        # mol/s * kg/kmol / 1000 mol/kmol = kg/s
        expected = mol_s * mw / 1000.0
        assert result == pytest.approx(expected, rel=1e-5)

    def test_volumetric_actual_to_mass_conversion(self):
        """Test actual volumetric to mass conversion."""
        density = 1.2  # kg/m³
        result = convert_flow_rate_to_mass(
            value=1000.0, from_unit="CFM", molecular_weight=28.964, density=density
        )
        assert result > 0

    def test_standard_volumetric_to_mass_conversion(self):
        """Test standard volumetric to mass conversion."""
        mw = 28.964  # Air
        result = convert_flow_rate_to_mass(
            value=1000.0, from_unit="SCFM", molecular_weight=mw, standard="SCFM"
        )
        assert result > 0


class TestConstants:
    """Test that conversion constants are properly defined."""

    def test_mass_flow_conversions_exist(self):
        """Test that mass flow conversion factors are defined."""
        assert "kg/s" in MASS_FLOW_CONVERSIONS
        assert "lb/h" in MASS_FLOW_CONVERSIONS
        assert MASS_FLOW_CONVERSIONS["kg/s"] == pytest.approx(1.0)

    def test_molar_flow_conversions_exist(self):
        """Test that molar flow conversion factors are defined."""
        assert "mol/s" in MOLAR_FLOW_CONVERSIONS
        assert "kmol/h" in MOLAR_FLOW_CONVERSIONS

    def test_standard_conditions_defined(self):
        """Test that standard conditions are defined."""
        assert "STP" in STANDARD_CONDITIONS
        assert "SCFM" in STANDARD_CONDITIONS
        assert "NTP" in STANDARD_CONDITIONS

    def test_r_universal_is_reasonable(self):
        """Test that universal gas constant is in expected range."""
        # R should be around 8314 J/(kmol·K)
        assert 8000 < R_UNIVERSAL < 9000
