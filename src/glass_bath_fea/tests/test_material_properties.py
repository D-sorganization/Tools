"""Tests for Glass Bath FEA material property models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add paths for imports (when running tests directly)
TOOLS_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(TOOLS_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT / "src" / "shared" / "python"))


class TestGlassMaterialModel:
    """Tests for the glass material property model."""

    def test_conductivity_increases_with_temperature(
        self, soda_lime_composition, temperature_range
    ) -> None:
        """Arrhenius behavior: higher T -> higher conductivity."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        conductivities = [model.get_conductivity(t) for t in temperature_range]

        # Each value should be greater than the previous (increasing with T)
        for i in range(1, len(conductivities)):
            assert conductivities[i] > conductivities[i - 1]

    def test_conductivity_arrhenius_form(self, soda_lime_composition) -> None:
        """Test that conductivity follows Arrhenius equation."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        # At reference temperature, conductivity should be close to base value
        # Reference temp is typically 1200°C (1473 K)
        sigma_ref = model.get_conductivity(1200.0)

        # Base conductivity from electrode adviser is ~1.0 S/m
        # With composition factors, should still be in reasonable range
        assert 0.1 < sigma_ref < 100.0

    def test_conductivity_positive(
        self, soda_lime_composition, temperature_range
    ) -> None:
        """Conductivity must always be positive."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        for temp in temperature_range:
            sigma = model.get_conductivity(temp)
            assert sigma > 0

    def test_composition_effect_on_conductivity(
        self, soda_lime_composition, high_iron_composition
    ) -> None:
        """Higher iron content should increase conductivity."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model_soda = GlassMaterialModel(soda_lime_composition)
        model_iron = GlassMaterialModel(high_iron_composition)

        temp = 1300.0  # °C
        sigma_soda = model_soda.get_conductivity(temp)
        sigma_iron = model_iron.get_conductivity(temp)

        # High iron glass should have higher conductivity
        assert sigma_iron > sigma_soda

    def test_resistivity_is_inverse_of_conductivity(
        self, soda_lime_composition
    ) -> None:
        """Resistivity should be 1/conductivity."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)
        temp = 1300.0

        sigma = model.get_conductivity(temp)
        rho = model.get_resistivity(temp)

        assert rho == pytest.approx(1.0 / sigma, rel=1e-6)

    def test_viscosity_decreases_with_temperature(
        self, soda_lime_composition, temperature_range
    ) -> None:
        """Viscosity should decrease as temperature increases."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        viscosities = [model.get_viscosity(t) for t in temperature_range]

        # Each value should be less than the previous (decreasing with T)
        for i in range(1, len(viscosities)):
            assert viscosities[i] < viscosities[i - 1]

    def test_viscosity_positive(self, soda_lime_composition, temperature_range) -> None:
        """Viscosity must always be positive."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        for temp in temperature_range:
            eta = model.get_viscosity(temp)
            assert eta > 0

    def test_viscosity_fulcher_form(self, soda_lime_composition) -> None:
        """Test that viscosity follows Fulcher equation form."""
        from glass_bath_fea.core.material_properties import GlassMaterialModel

        model = GlassMaterialModel(soda_lime_composition)

        # At working temperature (~1100°C), viscosity should be ~10^2-10^4 Pa·s
        eta_working = model.get_viscosity(1100.0)
        assert 1e1 < eta_working < 1e5

        # At melting temperature (~1400°C), viscosity should be lower
        eta_melt = model.get_viscosity(1400.0)
        assert eta_melt < eta_working


class TestMetalConductivity:
    """Tests for metal layer conductivity."""

    def test_metal_conductivity_constant(self) -> None:
        """Metal conductivity should be approximately constant."""
        from glass_bath_fea.core.material_properties import get_metal_conductivity

        sigma_1200 = get_metal_conductivity(1200.0)
        sigma_1350 = get_metal_conductivity(1350.0)

        # Metal conductivity is much less temperature-dependent
        # Should be within 20% over this range
        ratio = sigma_1350 / sigma_1200
        assert 0.8 < ratio < 1.2

    def test_metal_much_higher_than_glass(self, soda_lime_composition) -> None:
        """Metal conductivity should be orders of magnitude higher than glass."""
        from glass_bath_fea.core.material_properties import (
            GlassMaterialModel,
            get_metal_conductivity,
        )

        model = GlassMaterialModel(soda_lime_composition)
        temp = 1300.0

        sigma_glass = model.get_conductivity(temp)
        sigma_metal = get_metal_conductivity(temp)

        # Metal should be at least 100x more conductive
        assert sigma_metal > 100 * sigma_glass


class TestMaterialDataExport:
    """Tests for exporting material data to MATLAB format."""

    def test_export_material_data(self, soda_lime_composition, tmp_path) -> None:
        """Test exporting material properties to .mat file."""
        from glass_bath_fea.core.material_properties import (
            GlassMaterialModel,
            export_material_data,
        )

        model = GlassMaterialModel(soda_lime_composition)
        output_path = tmp_path / "materials.mat"

        export_material_data(model, output_path)

        assert output_path.exists()

        # Verify file can be loaded
        from scipy.io import loadmat

        data = loadmat(output_path)
        assert "arrhenius_params" in data or "base_conductivity" in data

    def test_export_contains_required_fields(
        self, soda_lime_composition, tmp_path
    ) -> None:
        """Test that exported data contains all required fields for MATLAB."""
        from glass_bath_fea.core.material_properties import (
            GlassMaterialModel,
            export_material_data,
        )

        model = GlassMaterialModel(soda_lime_composition)
        output_path = tmp_path / "materials.mat"

        export_material_data(model, output_path)

        from scipy.io import loadmat

        data = loadmat(output_path)

        # Required fields for MATLAB solver
        required_fields = [
            "base_conductivity",
            "activation_energy",
            "reference_temp",
            "metal_conductivity",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
