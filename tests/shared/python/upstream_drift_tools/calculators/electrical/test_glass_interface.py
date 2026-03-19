"""Comprehensive tests for GlassPropertiesInterface.

Tests cover construction, conductivity model (Arrhenius), caching,
LRU eviction, external calculator, resistivity, and DbC preconditions.
"""

from __future__ import annotations

from upstream_drift_tools.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)

# ─── Constructor ──────────────────────────────────────────────


class TestGlassPropertiesConstructor:
    def test_creates_without_external(self) -> None:
        gpi = GlassPropertiesInterface()
        assert gpi is not None
        assert gpi.external_calculator is None

    def test_creates_with_external(self) -> None:
        def mock_calc(t: float, c: dict | None, p: float) -> float:
            return 5.0

        gpi = GlassPropertiesInterface(external_calculator=mock_calc)
        assert gpi.external_calculator is not None

    def test_custom_cache_size(self) -> None:
        gpi = GlassPropertiesInterface(cache_max_size=50)
        assert gpi._cache_max_size == 50


# ─── Default Conductivity Model ──────────────────────────────


class TestDefaultConductivityModel:
    def test_positive_at_high_temp(self) -> None:
        gpi = GlassPropertiesInterface()
        cond = gpi.get_conductivity(1200.0)
        assert cond > 0.0

    def test_increases_with_temperature(self) -> None:
        gpi = GlassPropertiesInterface()
        low = gpi.get_conductivity(800.0)
        high = gpi.get_conductivity(1400.0)
        assert high > low, "Arrhenius: conductivity increases with temperature"

    def test_reference_temp_gives_base_conductivity(self) -> None:
        gpi = GlassPropertiesInterface()
        # At reference temp (1200°C), result should be near base_conductivity (1.0)
        cond = gpi.get_conductivity(1200.0)
        assert abs(cond - 1.0) < 0.01

    def test_power_density_increases_conductivity(self) -> None:
        gpi = GlassPropertiesInterface()
        no_pd = gpi.get_conductivity(1000.0, power_density=0)
        with_pd = gpi.get_conductivity(1000.0, power_density=50000)
        assert with_pd > no_pd

    def test_metal_returns_high_conductivity(self) -> None:
        gpi = GlassPropertiesInterface()
        metal = gpi.get_conductivity(1000.0, is_metal=True)
        glass = gpi.get_conductivity(1000.0, is_metal=False)
        assert metal > glass * 100, "Metal conductivity >> glass"

    def test_metal_constant_value(self) -> None:
        gpi = GlassPropertiesInterface()
        assert gpi.get_conductivity(500.0, is_metal=True) == 10000.0
        assert gpi.get_conductivity(1500.0, is_metal=True) == 10000.0


# ─── Caching ─────────────────────────────────────────────────


class TestCaching:
    def test_cache_hit(self) -> None:
        gpi = GlassPropertiesInterface()
        first = gpi.get_conductivity(1000.0)
        second = gpi.get_conductivity(1000.0)
        assert first == second

    def test_cache_with_composition(self) -> None:
        gpi = GlassPropertiesInterface()
        comp = {"SiO2": 0.7, "Na2O": 0.15}
        first = gpi.get_conductivity(1000.0, composition=comp)
        second = gpi.get_conductivity(1000.0, composition=comp)
        assert first == second

    def test_lru_eviction(self) -> None:
        gpi = GlassPropertiesInterface(cache_max_size=5)
        # Fill cache beyond limit
        for t in range(10):
            gpi.get_conductivity(float(1000 + t * 10))
        assert len(gpi._temperature_dependent_data) <= 5

    def test_clear_cache(self) -> None:
        gpi = GlassPropertiesInterface()
        gpi.get_conductivity(1000.0)
        assert len(gpi._temperature_dependent_data) > 0
        gpi.clear_cache()
        assert len(gpi._temperature_dependent_data) == 0


# ─── External Calculator ─────────────────────────────────────


class TestExternalCalculator:
    def test_uses_external(self) -> None:
        def calc(t: float, c: dict | None, p: float) -> float:
            return 42.0

        gpi = GlassPropertiesInterface(external_calculator=calc)
        result = gpi.get_conductivity(1000.0)
        assert result == 42.0

    def test_fallback_on_external_error(self) -> None:
        def bad_calc(t: float, c: dict | None, p: float) -> float:
            raise ValueError("broken")

        gpi = GlassPropertiesInterface(external_calculator=bad_calc)
        result = gpi.get_conductivity(1000.0)
        assert result > 0.0, "Falls back to default model"

    def test_set_external_clears_cache(self) -> None:
        gpi = GlassPropertiesInterface()
        gpi.get_conductivity(1000.0)
        assert len(gpi._temperature_dependent_data) > 0

        def new_calc(t: float, c: dict | None, p: float) -> float:
            return 99.0

        gpi.set_external_calculator(new_calc)
        assert len(gpi._temperature_dependent_data) == 0


# ─── Properties ──────────────────────────────────────────────


class TestProperties:
    def test_update_and_get(self) -> None:
        gpi = GlassPropertiesInterface()
        gpi.update_properties({"viscosity": 100.0})
        props = gpi.get_current_properties()
        assert props["viscosity"] == 100.0

    def test_get_returns_copy(self) -> None:
        gpi = GlassPropertiesInterface()
        gpi.update_properties({"key": "val"})
        p1 = gpi.get_current_properties()
        p1["key"] = "modified"
        p2 = gpi.get_current_properties()
        assert p2["key"] == "val", "Modifying returned dict shouldn't affect internal"


# ─── Resistivity ─────────────────────────────────────────────


class TestResistivity:
    def test_inverse_of_conductivity(self) -> None:
        gpi = GlassPropertiesInterface()
        cond = gpi.get_conductivity(1000.0)
        res = gpi.get_resistivity(1000.0)
        assert abs(cond * res - 1.0) < 1e-10

    def test_metal_resistivity_very_low(self) -> None:
        gpi = GlassPropertiesInterface()
        res = gpi.get_resistivity(1000.0, is_metal=True)
        assert res < 0.001
