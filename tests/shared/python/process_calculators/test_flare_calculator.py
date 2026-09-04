"""Tests for the Flare Calculator.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy, testing the math internally within the Tools repository.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.constants import FLARE_MIN_HEIGHT
from upstream_drift_tools.process_calculators.flare_calculator import (
    FlareCalculator,
    FlareDesign,
)


@pytest.fixture
def calculator() -> FlareCalculator:
    """Return a fresh FlareCalculator."""
    return FlareCalculator()


class TestFlareCalculator:
    """Tests for core math in the Flare parameter calculation."""

    def test_flare_size_calculation(self, calculator: FlareCalculator) -> None:
        """Ensure safe defaults and calculations apply."""
        mix = {"CH4": 1.0}
        design = calculator.calculate_flare_size(
            total_flow=1000.0,  # kg/hr
            gas_composition=mix,
            temperature=300.0,
            pressure=1.01325,  # bar
        )

        assert isinstance(design, FlareDesign)
        assert design.height >= FLARE_MIN_HEIGHT
        assert design.diameter > 0.0
        assert design.heat_release > 0.0  # kW

    def test_flare_size_preconditions(self, calculator: FlareCalculator) -> None:
        """Verify preconditions block invalid states."""
        with pytest.raises((AssertionError, ValueError), match="must be positive"):
            calculator.calculate_flare_size(
                total_flow=-50.0,
                gas_composition={"CH4": 1.0},
                temperature=300.0,
                pressure=1.0,
            )

    def test_radiation_zones(self, calculator: FlareCalculator) -> None:
        """Radiation distance zones should falloff monotonically."""
        mix = {"CH4": 1.0}
        design = calculator.calculate_flare_size(
            total_flow=5000.0,
            gas_composition=mix,
            temperature=400.0,
            pressure=1.0,
        )

        zones = calculator.calculate_radiation_zones(design)
        assert "lethal" in zones
        assert "damage" in zones
        assert "safe" in zones
        assert "comfort" in zones

        # Lethal radiation is highest intensity, so distance requires closer proximity
        assert zones["lethal"] < zones["damage"]
        assert zones["damage"] < zones["safe"]
        assert zones["safe"] < zones["comfort"]

    def test_combustion_efficiency(self, calculator: FlareCalculator) -> None:
        """Efficiency should scale with composition factors properly."""
        mix_good = {"CH4": 1.0}
        eff_good = calculator.calculate_combustion_efficiency(
            gas_composition=mix_good,
            temperature=500.0,
            pressure=1.0,
        )
        assert eff_good > 0.0

        mix_bad = {"CO": 0.8, "H2S": 0.2}
        eff_bad = calculator.calculate_combustion_efficiency(
            gas_composition=mix_bad,
            temperature=200.0,  # cold penalty
            pressure=1.0,
        )
        assert eff_bad < eff_good
