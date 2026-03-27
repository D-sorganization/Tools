"""Tests for the WGS Reactor Calculator.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy, testing the math internally within the Tools repository.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)


@pytest.fixture
def engine() -> WGSReactorEngine:
    """Return a fresh WGSReactorEngine."""
    return WGSReactorEngine()


class TestWGSReactorEngine:
    """Tests for core math in the WGS Reactor engine."""

    def test_equilibrium_constant(self, engine: WGSReactorEngine) -> None:
        """K_eq should decrease with higher temperatures (exothermic reaction)."""
        k_300 = engine.calculate_equilibrium_constant(573.15)  # 300C
        k_400 = engine.calculate_equilibrium_constant(673.15)  # 400C

        assert k_300 > 0
        assert k_400 > 0
        assert k_300 > k_400  # Exothermic shifts left at higher temp

    def test_equilibrium_composition(self, engine: WGSReactorEngine) -> None:
        """Test Gibbs minimization solver."""
        inlet = {"CO": 25.0, "H2O": 25.0, "CO2": 25.0, "H2": 25.0}  # Even mix

        eq = engine.calculate_equilibrium_composition(
            inlet_composition=inlet,
            temperature=673.15,  # 400C
            pressure=25.0,
            steam_ratio=2.0,
        )

        assert "conversion" in eq
        assert "composition" in eq
        assert "h2_co_ratio" in eq
        assert "equilibrium_constant" in eq
        assert "heat_released" in eq

        assert (
            eq["composition"]["H2"] >= 25.0
        )  # H2 should increase or stay high depending on shift
        assert eq["composition"]["CO"] <= 25.0  # CO should decrease as it shifts to H2

    def test_size_wgs_reactor(self, engine: WGSReactorEngine) -> None:
        """Physical sizing should yield positive volumes and dimensions."""
        sizing = engine.size_wgs_reactor(
            feed_rate=2000.0, conversion=80.0, temperature=673.15, catalyst_type="HTS"
        )

        assert sizing["reactor_volume"] > 0
        assert sizing["catalyst_volume"] > 0
        assert sizing["diameter"] > 0
        assert sizing["length"] > 0
        assert sizing["heat_duty"] > 0

    def test_zero_feed(self, engine: WGSReactorEngine) -> None:
        """Gracefully handle empty feeds."""
        eq = engine.calculate_equilibrium_composition(
            inlet_composition={}, temperature=673.15, pressure=25.0, steam_ratio=2.0
        )
        assert eq["conversion"] == 0.0
