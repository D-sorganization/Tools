"""Tests for ElectrodeAdvancementCalculator.

Covers construction, consumption calculation, and edge cases.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
    ElectrodeAdvancementCalculator,
)


class TestConstruction:
    def test_creates(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc is not None

    def test_default_consumption_rate(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc.consumption_rate == 0.5


class TestCalculateConsumption:
    def test_basic_calculation(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        # 10 kA * 2 hours * 0.5 in/kAh = 10 inches
        assert calc.calculate_consumption(10.0, 2.0) == pytest.approx(10.0)

    def test_zero_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc.calculate_consumption(0.0, 5.0) == 0.0

    def test_zero_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        assert calc.calculate_consumption(10.0, 0.0) == 0.0

    def test_custom_rate(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        calc.consumption_rate = 1.0
        assert calc.calculate_consumption(5.0, 3.0) == pytest.approx(15.0)

    def test_proportional_to_current(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        c1 = calc.calculate_consumption(10.0, 1.0)
        c2 = calc.calculate_consumption(20.0, 1.0)
        assert c2 == pytest.approx(2.0 * c1)

    def test_proportional_to_time(self) -> None:
        calc = ElectrodeAdvancementCalculator()
        c1 = calc.calculate_consumption(10.0, 1.0)
        c2 = calc.calculate_consumption(10.0, 3.0)
        assert c2 == pytest.approx(3.0 * c1)
