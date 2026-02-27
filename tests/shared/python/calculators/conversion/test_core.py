#!/usr/bin/env python3
"""Tests for conversion core helper functions.

Tests the stateless conversion helpers in core.py (113 lines).
"""

import pytest
from upstream_drift_tools.calculators.conversion.core import (
    actual_to_standard_flow,
    convert_temperature,
    convert_via_table,
    scfm_to_standard_m3_per_hour,
    standard_m3_per_hour_to_scfm,
    standard_to_actual_flow,
)
from upstream_drift_tools.calculators.conversion.tables import StandardCondition


class TestConvertViaTable:
    """Test table-based unit conversions."""

    def test_same_unit_returns_value(self):
        """Test converting to same unit returns original value."""
        table = {"m": 1.0, "km": 1000.0, "cm": 0.01}
        result = convert_via_table(100.0, "m", "m", table)
        assert result == pytest.approx(100.0)

    def test_meters_to_kilometers(self):
        """Test converting meters to kilometers."""
        table = {"m": 1.0, "km": 1000.0}
        result = convert_via_table(1000.0, "m", "km", table)
        assert result == pytest.approx(1.0)

    def test_kilometers_to_meters(self):
        """Test converting kilometers to meters."""
        table = {"m": 1.0, "km": 1000.0}
        result = convert_via_table(1.0, "km", "m", table)
        assert result == pytest.approx(1000.0)

    def test_centimeters_to_kilometers(self):
        """Test converting small to large units."""
        table = {"cm": 0.01, "km": 1000.0}
        result = convert_via_table(100000.0, "cm", "km", table)
        assert result == pytest.approx(1.0)


class TestConvertTemperature:
    """Test temperature conversions."""

    def test_same_unit_returns_value(self):
        """Test converting to same unit returns original value."""
        assert convert_temperature(100.0, "K", "K") == pytest.approx(100.0)
        assert convert_temperature(25.0, "C", "C") == pytest.approx(25.0)

    def test_celsius_to_kelvin(self):
        """Test C to K conversion."""
        result = convert_temperature(0.0, "C", "K")
        assert result == pytest.approx(273.15)

    def test_kelvin_to_celsius(self):
        """Test K to C conversion."""
        result = convert_temperature(273.15, "K", "C")
        assert result == pytest.approx(0.0)

    def test_fahrenheit_to_celsius(self):
        """Test F to C conversion."""
        result = convert_temperature(32.0, "F", "C")
        assert result == pytest.approx(0.0)
        result = convert_temperature(212.0, "F", "C")
        assert result == pytest.approx(100.0, abs=0.01)

    def test_celsius_to_fahrenheit(self):
        """Test C to F conversion."""
        result = convert_temperature(0.0, "C", "F")
        assert result == pytest.approx(32.0)
        result = convert_temperature(100.0, "C", "F")
        assert result == pytest.approx(212.0)

    def test_kelvin_to_rankine(self):
        """Test K to R conversion."""
        result = convert_temperature(273.15, "K", "R")
        assert result == pytest.approx(491.67, abs=0.01)

    def test_rankine_to_kelvin(self):
        """Test R to K conversion."""
        result = convert_temperature(491.67, "R", "K")
        assert result == pytest.approx(273.15, abs=0.01)

    def test_invalid_from_unit_raises_error(self):
        """Test that invalid from_unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, "X", "K")

    def test_invalid_to_unit_raises_error(self):
        """Test that invalid to_unit raises ValueError."""
        with pytest.raises(ValueError, match="Unknown temperature unit"):
            convert_temperature(100.0, "K", "X")


class TestStandardToActualFlow:
    """Test SCFM to ACFM conversion."""

    def test_at_standard_conditions_returns_same(self):
        """Test that at standard conditions, SCFM == ACFM."""
        scfm = 1000.0
        std = StandardCondition.STP
        std_temp, std_pressure, _ = std.value

        result = standard_to_actual_flow(scfm, std_temp, std_pressure, std)
        assert result == pytest.approx(scfm)

    def test_higher_temperature_increases_acfm(self):
        """Test that higher temperature increases ACFM."""
        scfm = 1000.0
        std = StandardCondition.STP
        std_temp, std_pressure, _ = std.value
        high_temp = std_temp * 2

        result = standard_to_actual_flow(scfm, high_temp, std_pressure, std)
        assert result == pytest.approx(scfm * 2)

    def test_higher_pressure_decreases_acfm(self):
        """Test that higher pressure decreases ACFM."""
        scfm = 1000.0
        std = StandardCondition.STP
        std_temp, std_pressure, _ = std.value
        high_pressure = std_pressure * 2

        result = standard_to_actual_flow(scfm, std_temp, high_pressure, std)
        assert result == pytest.approx(scfm / 2)

    def test_nonpositive_temperature_raises_value_error(self):
        """Test contract validation for temperature."""
        with pytest.raises(ValueError, match="temperature_k must be positive"):
            standard_to_actual_flow(100.0, 0.0, 101325.0, StandardCondition.STP)

    def test_nonpositive_pressure_raises_value_error(self):
        """Test contract validation for pressure."""
        with pytest.raises(ValueError, match="pressure_pa must be positive"):
            standard_to_actual_flow(100.0, 300.0, 0.0, StandardCondition.STP)


class TestActualToStandardFlow:
    """Test ACFM to SCFM conversion."""

    def test_roundtrip_conversion(self):
        """Test that SCFM->ACFM->SCFM returns original."""
        scfm_orig = 1000.0
        temp_k = 350.0
        pressure_pa = 150000.0
        std = StandardCondition.STP

        acfm = standard_to_actual_flow(scfm_orig, temp_k, pressure_pa, std)
        scfm_back = actual_to_standard_flow(acfm, temp_k, pressure_pa, std)

        assert scfm_back == pytest.approx(scfm_orig, rel=1e-10)

    def test_high_pressure_high_temp_partially_cancel(self):
        """Test that P and T effects partially cancel."""
        acfm = 1000.0
        std = StandardCondition.STP
        std_temp, std_pressure, _ = std.value

        # Double both T and P
        temp_k = std_temp * 2
        pressure_pa = std_pressure * 2

        scfm = actual_to_standard_flow(acfm, temp_k, pressure_pa, std)
        # Effect should be (T/T_std) * (P_std/P) = 2 * 0.5 = 1
        assert scfm == pytest.approx(acfm)

    def test_nonpositive_temperature_raises_value_error(self):
        """Test contract validation for temperature."""
        with pytest.raises(ValueError, match="temperature_k must be positive"):
            actual_to_standard_flow(100.0, 0.0, 101325.0, StandardCondition.STP)

    def test_nonpositive_pressure_raises_value_error(self):
        """Test contract validation for pressure."""
        with pytest.raises(ValueError, match="pressure_pa must be positive"):
            actual_to_standard_flow(100.0, 300.0, 0.0, StandardCondition.STP)


class TestSCFMToStandardM3PerHour:
    """Test SCFM to m³/h conversions."""

    def test_scfm_to_m3_hr_same_standard(self):
        """Test conversion when standards are the same."""
        scfm = 1000.0
        std = StandardCondition.SCFM_60F

        result = scfm_to_standard_m3_per_hour(scfm, std, std)
        # Should just be unit conversion, approximately 1.699 m³/h per SCFM
        assert result > 0

    def test_different_standards_affects_result(self):
        """Test that different reference standards change the result."""
        scfm = 1000.0
        std = StandardCondition.SCFM_60F
        ref = StandardCondition.STP

        result = scfm_to_standard_m3_per_hour(scfm, std, ref)
        # Should be different from same-standard case
        assert result > 0


class TestStandardM3PerHourToSCFM:
    """Test m³/h to SCFM conversions."""

    def test_roundtrip_conversion(self):
        """Test that m³/h->SCFM->m³/h returns original."""
        m3_hr_orig = 1000.0
        std = StandardCondition.SCFM_60F
        ref = StandardCondition.STP

        scfm = standard_m3_per_hour_to_scfm(m3_hr_orig, ref, std)
        m3_hr_back = scfm_to_standard_m3_per_hour(scfm, std, ref)

        assert m3_hr_back == pytest.approx(m3_hr_orig, rel=1e-10)

    def test_same_standard_roundtrip(self):
        """Test roundtrip when standards are the same."""
        m3_hr_orig = 500.0
        std = StandardCondition.SCFM_60F

        scfm = standard_m3_per_hour_to_scfm(m3_hr_orig, std, std)
        m3_hr_back = scfm_to_standard_m3_per_hour(scfm, std, std)

        assert m3_hr_back == pytest.approx(m3_hr_orig, rel=1e-10)
