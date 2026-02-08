"""Tests for the Unit Converter Python backend."""

from __future__ import annotations

import math

import pytest

from web_applications.unit_converter.converter import (
    CONVERSION_FACTORS,
    UnitConverter,
)


@pytest.fixture
def converter() -> UnitConverter:
    return UnitConverter()


class TestGetCategories:
    def test_returns_all_categories(self, converter: UnitConverter) -> None:
        categories = converter.get_categories()
        assert "temperature" in categories
        assert "length" in categories
        assert "mass" in categories
        assert "pressure" in categories
        assert "energy" in categories
        assert "gas_flow" in categories
        assert "heating_value" in categories

    def test_returns_nonempty_list(self, converter: UnitConverter) -> None:
        assert len(converter.get_categories()) > 10


class TestGetUnitsForCategory:
    def test_temperature_units(self, converter: UnitConverter) -> None:
        units = converter.get_units_for_category("temperature")
        assert units == ["K", "C", "F", "R"]

    def test_length_units(self, converter: UnitConverter) -> None:
        units = converter.get_units_for_category("length")
        assert "m" in units
        assert "ft" in units
        assert "in" in units

    def test_unknown_category(self, converter: UnitConverter) -> None:
        assert converter.get_units_for_category("nonexistent") == []


class TestGetCategory:
    def test_temperature_units(self, converter: UnitConverter) -> None:
        assert converter.get_category("K") == "temperature"
        assert converter.get_category("C") == "temperature"
        assert converter.get_category("F") == "temperature"

    def test_length_units(self, converter: UnitConverter) -> None:
        assert converter.get_category("m") == "length"
        assert converter.get_category("ft") == "length"

    def test_unknown_unit(self, converter: UnitConverter) -> None:
        assert converter.get_category("xyz_unknown") is None


class TestTemperatureConversion:
    def test_celsius_to_fahrenheit(self, converter: UnitConverter) -> None:
        result = converter.convert(100, "C", "F")
        assert abs(result.result - 212.0) < 0.01

    def test_fahrenheit_to_celsius(self, converter: UnitConverter) -> None:
        result = converter.convert(32, "F", "C")
        assert abs(result.result - 0.0) < 0.01

    def test_celsius_to_kelvin(self, converter: UnitConverter) -> None:
        result = converter.convert(0, "C", "K")
        assert abs(result.result - 273.15) < 0.01

    def test_kelvin_to_rankine(self, converter: UnitConverter) -> None:
        result = converter.convert(273.15, "K", "R")
        assert abs(result.result - 491.67) < 0.01

    def test_same_unit(self, converter: UnitConverter) -> None:
        result = converter.convert(100, "C", "C")
        assert result.result == 100.0

    def test_absolute_zero(self, converter: UnitConverter) -> None:
        result = converter.convert(0, "K", "C")
        assert abs(result.result - (-273.15)) < 0.01


class TestLinearConversions:
    def test_meters_to_feet(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "m", "ft")
        assert abs(result.result - 3.28084) < 0.001

    def test_feet_to_meters(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "ft", "m")
        assert abs(result.result - 0.3048) < 0.0001

    def test_kg_to_lb(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "kg", "lb")
        assert abs(result.result - 2.20462) < 0.001

    def test_psi_to_kpa(self, converter: UnitConverter) -> None:
        result = converter.convert(14.696, "psi", "kPa")
        assert abs(result.result - 101.325) < 0.01

    def test_atm_to_psi(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "atm", "psi")
        assert abs(result.result - 14.696) < 0.01

    def test_joule_to_btu(self, converter: UnitConverter) -> None:
        result = converter.convert(1055.06, "J", "BTU")
        assert abs(result.result - 1.0) < 0.01

    def test_kw_to_hp(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "kW", "hp")
        assert abs(result.result - 1.34102) < 0.01

    def test_liter_to_gallon(self, converter: UnitConverter) -> None:
        result = converter.convert(3.78541, "L", "gal")
        assert abs(result.result - 1.0) < 0.001

    def test_same_unit(self, converter: UnitConverter) -> None:
        result = converter.convert(42, "m", "m")
        assert result.result == 42.0


class TestMassFlowConversions:
    def test_kg_per_s_to_lb_per_hr(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "kg/s", "lb/hr")
        expected = 3600.0 / 0.45359237
        assert abs(result.result - expected) < 0.1

    def test_g_per_s_to_kg_per_hr(self, converter: UnitConverter) -> None:
        result = converter.convert(1, "g/s", "kg/hr")
        assert abs(result.result - 3.6) < 0.01


class TestCrossCategoryError:
    def test_length_to_mass_fails(self, converter: UnitConverter) -> None:
        with pytest.raises(ValueError, match="Cannot convert"):
            converter.convert(1, "m", "kg")

    def test_unknown_from_unit(self, converter: UnitConverter) -> None:
        with pytest.raises(ValueError, match="Unknown unit"):
            converter.convert(1, "xyz", "m")

    def test_unknown_to_unit(self, converter: UnitConverter) -> None:
        with pytest.raises(ValueError, match="Unknown unit"):
            converter.convert(1, "m", "xyz")


class TestFormatNumber:
    def test_normal_number(self, converter: UnitConverter) -> None:
        assert converter.format_number(3.14159) == "3.14159"

    def test_large_number(self, converter: UnitConverter) -> None:
        formatted = converter.format_number(1.23456e12)
        assert "e" in formatted.lower()

    def test_small_number(self, converter: UnitConverter) -> None:
        formatted = converter.format_number(1.23e-8)
        assert "e" in formatted.lower()

    def test_zero(self, converter: UnitConverter) -> None:
        assert converter.format_number(0) == "0"

    def test_nan(self, converter: UnitConverter) -> None:
        assert converter.format_number(float("nan")) == "nan"


class TestConversionResult:
    def test_result_fields(self, converter: UnitConverter) -> None:
        result = converter.convert(100, "C", "F")
        assert result.value == 100
        assert result.from_unit == "C"
        assert result.to_unit == "F"
        assert result.category == "temperature"
        assert isinstance(result.result, float)


class TestAllCategoriesHaveUnits:
    """Verify all categories in CONVERSION_FACTORS are accessible."""

    def test_each_category(self, converter: UnitConverter) -> None:
        for category in CONVERSION_FACTORS:
            units = converter.get_units_for_category(category)
            assert len(units) > 0, f"Category '{category}' has no units"
