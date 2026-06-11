"""Tests for upstream_drift_tools.process_calculators.electrode_advancement_calculator.

Covers:
- ElectrodeAdvancementCalculator init
- calculate_consumption with various inputs
- Edge cases (zero current, zero time)
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
    ElectrodeAdvancementCalculator,
)


class TestElectrodeAdvancementCalculator:
    def test_default_rate(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc.consumption_rate == 0.5

    def test_basic_consumption(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        # 10 kA for 2 hours at 0.5 in/kAh = 10.0 inches
        result = calc.calculate_consumption(current_ka=10.0, time_hrs=2.0)
        assert result == pytest.approx(10.0)

    def test_zero_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=0.0, time_hrs=5.0)
        assert result == pytest.approx(0.0)

    def test_zero_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=50.0, time_hrs=0.0)
        assert result == pytest.approx(0.0)

    def test_linearity_in_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        single = calc.calculate_consumption(current_ka=5.0, time_hrs=1.0)
        double = calc.calculate_consumption(current_ka=10.0, time_hrs=1.0)
        assert double == pytest.approx(2.0 * single)

    def test_linearity_in_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        single = calc.calculate_consumption(current_ka=5.0, time_hrs=1.0)
        triple = calc.calculate_consumption(current_ka=5.0, time_hrs=3.0)
        assert triple == pytest.approx(3.0 * single)
