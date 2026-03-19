"""Comprehensive tests for FlareCalculator.

Tests cover flare sizing, radiation zones, combustion efficiency,
constructor, and DbC preconditions.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.flare_calculator import (
    FlareCalculator,
    FlareDesign,
)

# ─── Fixtures ─────────────────────────────────────────────────


def _typical_composition() -> dict[str, float]:
    """Typical syngas-like composition."""
    return {"H2": 20.0, "CO": 30.0, "CH4": 5.0, "CO2": 15.0, "N2": 30.0}


# ─── Constructor ──────────────────────────────────────────────


class TestFlareConstructor:
    def test_creates(self) -> None:
        calc = FlareCalculator()
        assert calc is not None

    def test_has_gas_properties(self) -> None:
        calc = FlareCalculator()
        assert "H2" in calc.gas_properties
        assert "CO" in calc.gas_properties
        assert "N2" in calc.gas_properties


# ─── FlareDesign dataclass ───────────────────────────────────


class TestFlareDesign:
    def test_construction(self) -> None:
        fd = FlareDesign(
            height=30.0,
            diameter=0.5,
            exit_velocity=60.0,
            heat_release=5000.0,
            radiation_intensity=1.6,
        )
        assert fd.height == 30.0
        assert fd.diameter == 0.5


# ─── calculate_flare_size ────────────────────────────────────


class TestCalculateFlareSize:
    def test_returns_flare_design(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(
            total_flow=500.0,
            gas_composition=_typical_composition(),
            temperature=400.0,
            pressure=1.5,
        )
        assert isinstance(result, FlareDesign)

    def test_height_positive(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        assert result.height > 0.0

    def test_diameter_positive(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        assert result.diameter > 0.0

    def test_heat_release_positive(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        assert result.heat_release > 0.0

    def test_higher_flow_taller_flare(self) -> None:
        calc = FlareCalculator()
        small = calc.calculate_flare_size(100.0, _typical_composition(), 400.0, 1.5)
        large = calc.calculate_flare_size(1000.0, _typical_composition(), 400.0, 1.5)
        assert large.height > small.height

    def test_higher_flow_larger_diameter(self) -> None:
        calc = FlareCalculator()
        small = calc.calculate_flare_size(100.0, _typical_composition(), 400.0, 1.5)
        large = calc.calculate_flare_size(1000.0, _typical_composition(), 400.0, 1.5)
        assert large.diameter > small.diameter

    def test_pure_inert_gas_low_heat(self) -> None:
        calc = FlareCalculator()
        result = calc.calculate_flare_size(100.0, {"N2": 100.0}, 400.0, 1.5)
        # Inert gas has no heating value
        assert result.heat_release == 0.0

    def test_negative_flow_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises(AssertionError, match="total_flow"):
            calc.calculate_flare_size(-10.0, _typical_composition(), 400.0, 1.5)

    def test_zero_flow_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises(AssertionError, match="total_flow"):
            calc.calculate_flare_size(0.0, _typical_composition(), 400.0, 1.5)

    def test_negative_temp_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises(AssertionError, match="temperature"):
            calc.calculate_flare_size(100.0, _typical_composition(), -100.0, 1.5)

    def test_negative_pressure_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises(AssertionError, match="pressure"):
            calc.calculate_flare_size(100.0, _typical_composition(), 400.0, -1.0)

    def test_empty_composition_raises(self) -> None:
        calc = FlareCalculator()
        with pytest.raises(AssertionError, match="gas_composition"):
            calc.calculate_flare_size(100.0, {}, 400.0, 1.5)


# ─── calculate_radiation_zones ────────────────────────────────


class TestRadiationZones:
    def test_returns_four_zones(self) -> None:
        calc = FlareCalculator()
        design = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        zones = calc.calculate_radiation_zones(design)
        assert "lethal" in zones
        assert "damage" in zones
        assert "safe" in zones
        assert "comfort" in zones

    def test_lethal_closer_than_comfort(self) -> None:
        calc = FlareCalculator()
        design = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        zones = calc.calculate_radiation_zones(design)
        assert zones["lethal"] < zones["comfort"]

    def test_damage_between_lethal_and_safe(self) -> None:
        calc = FlareCalculator()
        design = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        zones = calc.calculate_radiation_zones(design)
        assert zones["lethal"] < zones["damage"] < zones["safe"]

    def test_all_zones_positive(self) -> None:
        calc = FlareCalculator()
        design = calc.calculate_flare_size(500.0, _typical_composition(), 400.0, 1.5)
        zones = calc.calculate_radiation_zones(design)
        for zone, distance in zones.items():
            assert distance > 0.0, f"Zone {zone} should be positive"

    def test_larger_flare_wider_zones(self) -> None:
        calc = FlareCalculator()
        small = calc.calculate_flare_size(100.0, _typical_composition(), 400.0, 1.5)
        large = calc.calculate_flare_size(1000.0, _typical_composition(), 400.0, 1.5)
        small_zones = calc.calculate_radiation_zones(small)
        large_zones = calc.calculate_radiation_zones(large)
        assert large_zones["safe"] > small_zones["safe"]


# ─── calculate_combustion_efficiency ──────────────────────────


class TestCombustionEfficiency:
    def test_returns_float(self) -> None:
        calc = FlareCalculator()
        eff = calc.calculate_combustion_efficiency(_typical_composition(), 400.0, 1.5)
        assert isinstance(eff, float)

    def test_between_zero_and_one(self) -> None:
        calc = FlareCalculator()
        eff = calc.calculate_combustion_efficiency(_typical_composition(), 400.0, 1.5)
        assert 0.0 < eff <= 1.0

    def test_high_h2_boosts_efficiency(self) -> None:
        calc = FlareCalculator()
        low_h2 = {"H2": 5.0, "CO": 30.0, "N2": 65.0}
        high_h2 = {"H2": 50.0, "CO": 30.0, "N2": 20.0}
        eff_low = calc.calculate_combustion_efficiency(low_h2, 400.0, 1.5)
        eff_high = calc.calculate_combustion_efficiency(high_h2, 400.0, 1.5)
        assert eff_high >= eff_low

    def test_high_h2s_reduces_efficiency(self) -> None:
        calc = FlareCalculator()
        clean = {"H2": 20.0, "CO": 30.0, "N2": 50.0}
        dirty = {"H2": 20.0, "CO": 30.0, "H2S": 20.0, "N2": 30.0}
        eff_clean = calc.calculate_combustion_efficiency(clean, 400.0, 1.5)
        eff_dirty = calc.calculate_combustion_efficiency(dirty, 400.0, 1.5)
        assert eff_dirty <= eff_clean

    def test_cold_temp_reduces_efficiency(self) -> None:
        calc = FlareCalculator()
        comp = _typical_composition()
        eff_cold = calc.calculate_combustion_efficiency(comp, 200.0, 1.5)
        eff_hot = calc.calculate_combustion_efficiency(comp, 600.0, 1.5)
        assert eff_hot >= eff_cold

    def test_efficiency_clamped_max(self) -> None:
        calc = FlareCalculator()
        # H2-rich gas at high temp — should hit max
        comp = {"H2": 90.0, "N2": 10.0}
        eff = calc.calculate_combustion_efficiency(comp, 1000.0, 1.5)
        assert eff <= 1.0
