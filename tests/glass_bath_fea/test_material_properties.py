"""Tests for glass_bath_fea.core.material_properties module.

Covers:
- GlassMaterialModel: Arrhenius conductivity, composition correction,
  resistivity (1/σ), Fulcher viscosity
- Physical law verification:
  - Conductivity increases with temperature (Arrhenius)
  - Resistivity = 1/conductivity
  - Viscosity decreases with temperature (Fulcher)
  - Na2O increases conductivity
- get_metal_conductivity
- get_arrhenius_params
"""

from __future__ import annotations

import pytest
from numpy.testing import assert_allclose

from glass_bath_fea.core.config import GlassComposition
from glass_bath_fea.core.material_properties import (
    DEFAULT_ACTIVATION_ENERGY,
    DEFAULT_BASE_CONDUCTIVITY,
    DEFAULT_METAL_CONDUCTIVITY,
    DEFAULT_REFERENCE_TEMP_K,
    FULCHER_A,
    FULCHER_B,
    FULCHER_T0,
    GlassMaterialModel,
    get_metal_conductivity,
)

# ── GlassMaterialModel Construction ─────────────────────────────────────


class TestGlassMaterialModelConstruction:
    """Test GlassMaterialModel construction."""

    def test_default_construction(self) -> None:
        comp = GlassComposition()
        model = GlassMaterialModel(comp)
        assert model.composition == comp

    def test_custom_parameters(self) -> None:
        comp = GlassComposition()
        model = GlassMaterialModel(
            comp,
            base_conductivity=2.0,
            activation_energy=90000,
        )
        assert model._base_conductivity == 2.0
        assert model._activation_energy == 90000


# ── Arrhenius Conductivity ───────────────────────────────────────────────


class TestArrheniusConductivity:
    """Test Arrhenius equation-based conductivity calculations."""

    @pytest.fixture()
    def model(self) -> GlassMaterialModel:
        return GlassMaterialModel(GlassComposition())

    def test_conductivity_positive(self, model: GlassMaterialModel) -> None:
        sigma = model.get_conductivity(1200)
        assert sigma > 0

    def test_conductivity_increases_with_temperature(
        self, model: GlassMaterialModel
    ) -> None:
        """Arrhenius: conductivity increases exponentially with T."""
        sigma_1000 = model.get_conductivity(1000)
        sigma_1200 = model.get_conductivity(1200)
        sigma_1400 = model.get_conductivity(1400)
        assert sigma_1000 < sigma_1200 < sigma_1400

    def test_conductivity_at_reference_temp(self, model: GlassMaterialModel) -> None:
        """At reference temperature, should be close to base * composition factor."""
        ref_temp_c = DEFAULT_REFERENCE_TEMP_K - 273.15
        sigma = model.get_conductivity(ref_temp_c)
        comp_factor = model._get_composition_factor()
        expected = DEFAULT_BASE_CONDUCTIVITY * comp_factor
        # At ref temp, exp term = exp(0) = 1
        assert_allclose(sigma, expected, rtol=0.01)

    def test_power_density_increases_conductivity(
        self, model: GlassMaterialModel
    ) -> None:
        """Power density causes local heating, increasing conductivity."""
        sigma_no_power = model.get_conductivity(1200, power_density=0)
        sigma_with_power = model.get_conductivity(1200, power_density=10000)
        assert sigma_with_power > sigma_no_power


# ── Composition Effects ──────────────────────────────────────────────────


class TestCompositionEffects:
    """Test composition correction factor."""

    def test_higher_na2o_increases_conductivity(self) -> None:
        """Na2O increases ionic mobility."""
        comp_low = GlassComposition(na2o=10.0)
        comp_high = GlassComposition(na2o=16.0)
        model_low = GlassMaterialModel(comp_low)
        model_high = GlassMaterialModel(comp_high)
        assert model_low.get_conductivity(1200) < model_high.get_conductivity(1200)

    def test_higher_fe2o3_increases_conductivity(self) -> None:
        """Fe2O3 increases electronic conduction."""
        comp_low = GlassComposition(fe2o3=0.0)
        comp_high = GlassComposition(fe2o3=1.0)
        model_low = GlassMaterialModel(comp_low)
        model_high = GlassMaterialModel(comp_high)
        assert model_low.get_conductivity(1200) < model_high.get_conductivity(1200)

    def test_default_composition_factor_near_1(self) -> None:
        """Default soda-lime composition should have factor near 1."""
        model = GlassMaterialModel(GlassComposition())
        factor = model._get_composition_factor()
        # na_factor = 1.0 + 0.02*(13-13) = 1.0, fe_factor = 1.0 + 0.5*0.1 = 1.05
        assert_allclose(factor, 1.05, atol=0.01)


# ── Resistivity ──────────────────────────────────────────────────────────


class TestResistivity:
    """Test resistivity calculations."""

    def test_resistivity_is_inverse_conductivity(self) -> None:
        model = GlassMaterialModel(GlassComposition())
        sigma = model.get_conductivity(1200)
        rho = model.get_resistivity(1200)
        assert_allclose(sigma * rho, 1.0, atol=1e-10)

    def test_resistivity_decreases_with_temperature(self) -> None:
        model = GlassMaterialModel(GlassComposition())
        rho_1000 = model.get_resistivity(1000)
        rho_1400 = model.get_resistivity(1400)
        assert rho_1000 > rho_1400


# ── Fulcher Viscosity ────────────────────────────────────────────────────


class TestFulcherViscosity:
    """Test Fulcher equation viscosity calculations."""

    @pytest.fixture()
    def model(self) -> GlassMaterialModel:
        return GlassMaterialModel(GlassComposition())

    def test_viscosity_positive(self, model: GlassMaterialModel) -> None:
        eta = model.get_viscosity(1200)
        assert eta > 0

    def test_viscosity_decreases_with_temperature(
        self, model: GlassMaterialModel
    ) -> None:
        """Glass viscosity decreases significantly with temperature."""
        eta_1000 = model.get_viscosity(1000)
        eta_1200 = model.get_viscosity(1200)
        eta_1400 = model.get_viscosity(1400)
        assert eta_1000 > eta_1200 > eta_1400

    def test_viscosity_manual_fulcher(self, model: GlassMaterialModel) -> None:
        """Verify against manual Fulcher calculation."""
        temp_c = 1300.0
        temp_k = temp_c + 273.15
        log_eta = FULCHER_A + FULCHER_B / (temp_k - FULCHER_T0)
        expected = 10.0**log_eta
        actual = model.get_viscosity(temp_c)
        assert_allclose(actual, expected, rtol=1e-10)


# ── Arrhenius Parameters Export ──────────────────────────────────────────


class TestArrheniusParamsExport:
    """Test get_arrhenius_params for MATLAB export."""

    def test_returns_expected_keys(self) -> None:
        model = GlassMaterialModel(GlassComposition())
        params = model.get_arrhenius_params()
        assert "base_conductivity" in params
        assert "activation_energy" in params
        assert "reference_temp" in params
        assert "composition_factor" in params

    def test_values_match_model(self) -> None:
        model = GlassMaterialModel(GlassComposition())
        params = model.get_arrhenius_params()
        assert params["base_conductivity"] == DEFAULT_BASE_CONDUCTIVITY
        assert params["activation_energy"] == DEFAULT_ACTIVATION_ENERGY
        assert params["reference_temp"] == DEFAULT_REFERENCE_TEMP_K


# ── Metal Conductivity ──────────────────────────────────────────────────


class TestMetalConductivity:
    """Test get_metal_conductivity function."""

    def test_at_reference_temperature(self) -> None:
        sigma = get_metal_conductivity(1200.0)
        assert_allclose(sigma, DEFAULT_METAL_CONDUCTIVITY, rtol=0.01)

    def test_much_higher_than_glass(self) -> None:
        """Metal conductivity >> glass conductivity."""
        sigma_metal = get_metal_conductivity(1200.0)
        model = GlassMaterialModel(GlassComposition())
        sigma_glass = model.get_conductivity(1200.0)
        assert sigma_metal > 100 * sigma_glass

    def test_decreases_slightly_with_temperature(self) -> None:
        """Metals have slightly decreasing conductivity with temperature."""
        sigma_1000 = get_metal_conductivity(1000.0)
        sigma_1400 = get_metal_conductivity(1400.0)
        assert sigma_1000 > sigma_1400


# ── GlassComposition ────────────────────────────────────────────────────


class TestGlassComposition:
    """Test GlassComposition dataclass."""

    def test_defaults_sum_near_100(self) -> None:
        comp = GlassComposition()
        total = comp.total_percent()
        assert 99.0 <= total <= 101.0

    def test_validates_default(self) -> None:
        assert GlassComposition().validate() is True

    def test_invalid_composition(self) -> None:
        comp = GlassComposition(sio2=50.0, na2o=5.0, cao=0.0)
        assert comp.validate() is False

    def test_custom_composition(self) -> None:
        comp = GlassComposition(
            sio2=72.0, na2o=14.0, cao=10.0, mgo=3.0, al2o3=0.9, fe2o3=0.1
        )
        assert comp.validate() is True
