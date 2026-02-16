"""Tests for upstream_drift_tools.process_calculators.water_vapor_pressure_calculator.

Covers:
- WaterVaporPressureCalculator init
- calculate_vapor_pressure at known temperatures
- Monotonicity: higher temperature → higher vapor pressure
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.water_vapor_pressure_calculator import (
    WaterVaporPressureCalculator,
)


class TestWaterVaporPressureCalculator:
    def test_instantiation(self) -> None:
        calc = WaterVaporPressureCalculator()
        assert calc.calculator is not None

    def test_at_100c_near_1atm(self) -> None:
        """At 100°C, water vapor pressure should be near 101325 Pa (1 atm).

        The Buck equation gives ~104 kPa which is within 3% of 1 atm.
        """
        calc = WaterVaporPressureCalculator()
        pressure = calc.calculate_vapor_pressure(100.0)
        assert pressure == pytest.approx(101325.0, rel=0.05)

    def test_at_0c_low_pressure(self) -> None:
        """At 0°C, vapor pressure ≈ 611 Pa."""
        calc = WaterVaporPressureCalculator()
        pressure = calc.calculate_vapor_pressure(0.0)
        assert pressure == pytest.approx(611.0, rel=0.05)

    def test_monotonically_increasing(self) -> None:
        """Vapor pressure should increase with temperature."""
        calc = WaterVaporPressureCalculator()
        temps = [0, 20, 40, 60, 80, 100]
        pressures = [calc.calculate_vapor_pressure(t) for t in temps]
        for i in range(len(pressures) - 1):
            assert pressures[i + 1] > pressures[i]

    def test_positive_pressure(self) -> None:
        calc = WaterVaporPressureCalculator()
        assert calc.calculate_vapor_pressure(25.0) > 0

    def test_returns_float(self) -> None:
        calc = WaterVaporPressureCalculator()
        result = calc.calculate_vapor_pressure(50.0)
        assert isinstance(result, float)
