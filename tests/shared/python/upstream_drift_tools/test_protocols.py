"""Tests for upstream_drift_tools.protocols.

Covers CalculationResult, ValidationResult, InputValidator, and
the Calculator protocol conformance check.
"""

from __future__ import annotations

from typing import Any

import pytest

from upstream_drift_tools.protocols import (
    CalculationResult,
    Calculator,
    InputValidator,
    ValidationResult,
)


# ======================================================================
# CalculationResult
# ======================================================================


class TestCalculationResult:
    """Tests for the CalculationResult dataclass."""

    def test_default_construction(self) -> None:
        result = CalculationResult()
        assert result.values == {}
        assert result.units == {}
        assert result.warnings == []
        assert result.metadata == {}

    def test_construction_with_values(self) -> None:
        result = CalculationResult(
            values={"flow": 42.0, "pressure": 101325.0},
            units={"flow": "kg/s", "pressure": "Pa"},
            warnings=["Near choke condition"],
            metadata={"solver": "Newton-Raphson"},
        )
        assert result.values["flow"] == 42.0
        assert result.units["pressure"] == "Pa"
        assert len(result.warnings) == 1
        assert result.metadata["solver"] == "Newton-Raphson"

    def test_values_are_mutable(self) -> None:
        result = CalculationResult()
        result.values["temperature"] = 300.0
        assert result.values["temperature"] == 300.0

    def test_warnings_appendable(self) -> None:
        result = CalculationResult()
        result.warnings.append("first")
        result.warnings.append("second")
        assert len(result.warnings) == 2


# ======================================================================
# ValidationResult
# ======================================================================


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_default_is_valid(self) -> None:
        vr = ValidationResult()
        assert vr.valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_invalid_construction(self) -> None:
        vr = ValidationResult(valid=False, errors=["bad input"])
        assert vr.valid is False
        assert "bad input" in vr.errors

    def test_warnings_without_errors(self) -> None:
        vr = ValidationResult(warnings=["consider lower flow"])
        assert vr.valid is True
        assert len(vr.warnings) == 1


# ======================================================================
# InputValidator.require_positive
# ======================================================================


class TestRequirePositive:
    """Tests for InputValidator.require_positive."""

    def test_positive_value_passes(self) -> None:
        InputValidator.require_positive("flow", 1.0)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InputValidator.require_positive("flow", 0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            InputValidator.require_positive("flow", -5.0)


# ======================================================================
# InputValidator.require_in_range
# ======================================================================


class TestRequireInRange:
    """Tests for InputValidator.require_in_range."""

    def test_value_in_range_passes(self) -> None:
        InputValidator.require_in_range("temp", 300.0, 200.0, 500.0)

    def test_value_at_low_boundary_passes(self) -> None:
        InputValidator.require_in_range("temp", 200.0, 200.0, 500.0)

    def test_value_at_high_boundary_passes(self) -> None:
        InputValidator.require_in_range("temp", 500.0, 200.0, 500.0)

    def test_value_below_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in range"):
            InputValidator.require_in_range("temp", 100.0, 200.0, 500.0)

    def test_value_above_range_raises(self) -> None:
        with pytest.raises(ValueError, match="must be in range"):
            InputValidator.require_in_range("temp", 600.0, 200.0, 500.0)


# ======================================================================
# InputValidator.require_keys
# ======================================================================


class TestRequireKeys:
    """Tests for InputValidator.require_keys."""

    def test_all_keys_present_passes(self) -> None:
        inputs = {"a": 1, "b": 2, "c": 3}
        InputValidator.require_keys(inputs, {"a", "b"})

    def test_missing_keys_raises(self) -> None:
        inputs = {"a": 1}
        with pytest.raises(ValueError, match="Missing required keys"):
            InputValidator.require_keys(inputs, {"a", "b", "c"})

    def test_empty_required_passes(self) -> None:
        InputValidator.require_keys({"x": 1}, set())


# ======================================================================
# InputValidator.validate_temperature
# ======================================================================


class TestValidateTemperature:
    """Tests for InputValidator.validate_temperature."""

    def test_valid_temperature(self) -> None:
        InputValidator.validate_temperature(300.0)

    def test_zero_kelvin_raises(self) -> None:
        with pytest.raises(ValueError, match="Temperature must be > 0 K"):
            InputValidator.validate_temperature(0.0)

    def test_negative_kelvin_raises(self) -> None:
        with pytest.raises(ValueError, match="Temperature must be > 0 K"):
            InputValidator.validate_temperature(-10.0)


# ======================================================================
# InputValidator.validate_pressure
# ======================================================================


class TestValidatePressure:
    """Tests for InputValidator.validate_pressure."""

    def test_valid_pressure(self) -> None:
        InputValidator.validate_pressure(101325.0)

    def test_zero_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="Pressure must be > 0 Pa"):
            InputValidator.validate_pressure(0.0)

    def test_negative_pressure_raises(self) -> None:
        with pytest.raises(ValueError, match="Pressure must be > 0 Pa"):
            InputValidator.validate_pressure(-1.0)


# ======================================================================
# InputValidator.validate_composition
# ======================================================================


class TestValidateComposition:
    """Tests for InputValidator.validate_composition."""

    def test_valid_composition(self) -> None:
        comp = {"CH4": 0.7, "CO2": 0.2, "N2": 0.1}
        InputValidator.validate_composition(comp)

    def test_composition_not_summing_to_one_raises(self) -> None:
        comp = {"CH4": 0.5, "CO2": 0.2}
        with pytest.raises(ValueError, match="must sum to 1.0"):
            InputValidator.validate_composition(comp)

    def test_negative_fraction_raises(self) -> None:
        comp = {"CH4": 1.1, "CO2": -0.1}
        with pytest.raises(ValueError, match="negative"):
            InputValidator.validate_composition(comp)

    def test_tolerance_respected(self) -> None:
        # Sum is 1.0 + 1e-9, well within default tolerance of 1e-6
        comp = {"A": 0.5 + 5e-10, "B": 0.5 + 5e-10}
        InputValidator.validate_composition(comp)

    def test_custom_tolerance(self) -> None:
        comp = {"A": 0.5, "B": 0.49}  # sum = 0.99
        with pytest.raises(ValueError):
            InputValidator.validate_composition(comp, tolerance=1e-6)
        # With a generous tolerance it should pass
        InputValidator.validate_composition(comp, tolerance=0.02)


# ======================================================================
# Calculator protocol conformance
# ======================================================================


class _MockCalculator:
    """Concrete class that satisfies the Calculator protocol."""

    @property
    def name(self) -> str:
        return "MockCalc"

    @property
    def version(self) -> str:
        return "0.1.0"

    def calculate(self, inputs: dict[str, Any]) -> CalculationResult:
        return CalculationResult(values={"result": 1.0})

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        return ValidationResult()


class TestCalculatorProtocol:
    """Tests for the Calculator protocol runtime check."""

    def test_conforming_class_is_instance(self) -> None:
        calc = _MockCalculator()
        assert isinstance(calc, Calculator)

    def test_mock_calculator_calculate(self) -> None:
        calc = _MockCalculator()
        result = calc.calculate({"x": 1})
        assert result.values["result"] == 1.0

    def test_mock_calculator_validate(self) -> None:
        calc = _MockCalculator()
        vr = calc.validate_inputs({})
        assert vr.valid is True

    def test_nonconforming_object_fails(self) -> None:
        class _Bad:
            pass

        assert not isinstance(_Bad(), Calculator)
