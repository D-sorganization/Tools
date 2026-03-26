"""Tests for glass_bath_fea.core.material_properties module.

Covers:
- GlassMaterialModel conductivity (Arrhenius)
- Resistivity (inverse of conductivity)
- Viscosity (Fulcher equation)
- Composition correction factor
- Metal conductivity
- Arrhenius parameter export
"""

from __future__ import annotations

import pytest
from glass_bath_fea.core.config import GlassComposition
from glass_bath_fea.core.material_properties import (
    DEFAULT_METAL_CONDUCTIVITY,
    GlassMaterialModel,
    get_metal_conductivity,
)


@pytest.fixture()
def default_model() -> GlassMaterialModel:
    """Create model with default soda-lime composition."""
    return GlassMaterialModel(GlassComposition())


class TestConductivity:
    """Tests for glass conductivity model."""

    def test_positive_conductivity(self, default_model: GlassMaterialModel) -> None:
        sigma = default_model.get_conductivity(1200.0)
        assert sigma > 0

    def test_conductivity_increases_with_temperature(
        self, default_model: GlassMaterialModel
    ) -> None:
        """Arrhenius: conductivity increases with T."""
        sigma_low = default_model.get_conductivity(1000.0)
        sigma_high = default_model.get_conductivity(1400.0)
        assert sigma_high > sigma_low

    def test_conductivity_at_reference_temp(self) -> None:
        """At reference temp, conductivity ≈ base * comp_factor."""
        comp = GlassComposition()
        model = GlassMaterialModel(comp)
        sigma = model.get_conductivity(1200.0)  # 1200°C = 1473.15 K ≈ Tref
        comp_factor = model._get_composition_factor()
        assert sigma == pytest.approx(1.0 * comp_factor, rel=0.05)

    def test_power_density_increases_conductivity(self, default_model: GlassMaterialModel) -> None:
        sigma_no_power = default_model.get_conductivity(1200.0, power_density=0)
        sigma_with_power = default_model.get_conductivity(1200.0, power_density=1e6)
        assert sigma_with_power >= sigma_no_power


class TestCompositionFactor:
    """Tests for composition correction factor."""

    def test_standard_composition_near_unity(self) -> None:
        """Default composition should give factor ≈ 1.0 * (1 + 0.5*0.1)."""
        model = GlassMaterialModel(GlassComposition())
        factor = model._get_composition_factor()
        assert factor > 0
        assert factor == pytest.approx(1.0 * (1.0 + 0.5 * 0.1), rel=0.01)

    def test_high_na2o_increases_factor(self) -> None:
        """More Na2O → higher ionic mobility → higher factor."""
        low_na = GlassMaterialModel(GlassComposition(na2o=10.0))
        high_na = GlassMaterialModel(GlassComposition(na2o=20.0))
        assert high_na._get_composition_factor() > low_na._get_composition_factor()

    def test_high_fe2o3_increases_factor(self) -> None:
        """More Fe2O3 → higher electronic conduction → higher factor."""
        low_fe = GlassMaterialModel(GlassComposition(fe2o3=0.0))
        high_fe = GlassMaterialModel(GlassComposition(fe2o3=1.0))
        assert high_fe._get_composition_factor() > low_fe._get_composition_factor()


class TestResistivity:
    """Tests for electrical resistivity."""

    def test_inverse_of_conductivity(self, default_model: GlassMaterialModel) -> None:
        sigma = default_model.get_conductivity(1200.0)
        rho = default_model.get_resistivity(1200.0)
        assert rho == pytest.approx(1.0 / sigma, rel=1e-6)


class TestViscosity:
    """Tests for Fulcher viscosity model."""

    def test_positive_viscosity(self, default_model: GlassMaterialModel) -> None:
        eta = default_model.get_viscosity(1200.0)
        assert eta > 0

    def test_viscosity_decreases_with_temperature(self, default_model: GlassMaterialModel) -> None:
        """Fulcher: viscosity decreases with increasing T."""
        eta_low = default_model.get_viscosity(1000.0)
        eta_high = default_model.get_viscosity(1400.0)
        assert eta_high < eta_low

    def test_typical_viscosity_range(self, default_model: GlassMaterialModel) -> None:
        """At 1200°C, molten glass viscosity is typically 1-1000 Pa·s."""
        eta = default_model.get_viscosity(1200.0)
        assert 0.001 < eta < 10000


class TestArrheniusParams:
    """Tests for Arrhenius parameter export."""

    def test_keys(self, default_model: GlassMaterialModel) -> None:
        params = default_model.get_arrhenius_params()
        assert "base_conductivity" in params
        assert "activation_energy" in params
        assert "reference_temp" in params
        assert "composition_factor" in params


class TestMetalConductivity:
    """Tests for metal conductivity function."""

    def test_at_reference_temp(self) -> None:
        sigma = get_metal_conductivity(1200.0)
        assert sigma == pytest.approx(DEFAULT_METAL_CONDUCTIVITY, rel=0.01)

    def test_much_higher_than_glass(self) -> None:
        model = GlassMaterialModel(GlassComposition())
        sigma_glass = model.get_conductivity(1200.0)
        sigma_metal = get_metal_conductivity(1200.0)
        assert sigma_metal > 100 * sigma_glass

    def test_decreases_slightly_with_temperature(self) -> None:
        sigma_low = get_metal_conductivity(1000.0)
        sigma_high = get_metal_conductivity(1500.0)
        assert sigma_low > sigma_high
