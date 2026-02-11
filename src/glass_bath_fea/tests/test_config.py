"""Tests for Glass Bath FEA configuration dataclasses."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]

from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig, GlassComposition


class TestGlassBathFEAConfig:
    """Tests for the main FEA configuration dataclass."""

    def test_default_values(self) -> None:
        """Test that default configuration values are reasonable."""
        from glass_bath_fea.core.config import GlassBathFEAConfig

        config = GlassBathFEAConfig()

        # Verify default vessel dimensions (inches)
        assert config.bath_diameter == 120.0
        assert config.glass_depth == 15.0
        assert config.metal_layer_thickness == 2.0

        # Verify electrode configuration
        assert config.num_electrodes == 3
        assert config.electrode_spacing_degrees == 120.0

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        from glass_bath_fea.core.config import GlassBathFEAConfig

        config = GlassBathFEAConfig(
            bath_diameter=150.0,
            glass_depth=20.0,
            metal_layer_thickness=3.0,
            operating_temperature=1400.0,
        )

        assert config.bath_diameter == 150.0
        assert config.glass_depth == 20.0
        assert config.metal_layer_thickness == 3.0
        assert config.operating_temperature == 1400.0

    def test_total_height_property(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test that total height is correctly computed."""
        expected = (
            default_fea_config.glass_depth + default_fea_config.metal_layer_thickness
        )
        assert default_fea_config.total_height == expected

    def test_bath_radius_property(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that bath radius is correctly computed."""
        expected = default_fea_config.bath_diameter / 2.0
        assert default_fea_config.bath_radius == expected

    def test_dimensions_in_meters(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test conversion to SI units (meters)."""
        inches_to_meters = 0.0254

        dims = default_fea_config.get_dimensions_meters()

        assert dims["bath_diameter"] == pytest.approx(
            120.0 * inches_to_meters, rel=1e-6
        )
        assert dims["glass_depth"] == pytest.approx(15.0 * inches_to_meters, rel=1e-6)
        assert dims["metal_layer_thickness"] == pytest.approx(
            2.0 * inches_to_meters, rel=1e-6
        )

    def test_electrode_angles(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test that electrode angles are correctly computed."""
        angles = default_fea_config.get_electrode_angles_radians()

        assert len(angles) == 3
        # Should be 0°, 120°, 240° in radians
        assert angles[0] == pytest.approx(0.0, abs=1e-6)
        assert angles[1] == pytest.approx(2 * math.pi / 3, rel=1e-6)
        assert angles[2] == pytest.approx(4 * math.pi / 3, rel=1e-6)

    def test_phase_voltages_default(self) -> None:
        """Test default three-phase voltages."""
        from glass_bath_fea.core.config import GlassBathFEAConfig

        config = GlassBathFEAConfig()

        assert len(config.phase_voltages) == 3
        assert all(v == 100.0 for v in config.phase_voltages)


class TestGlassComposition:
    """Tests for glass composition specification."""

    def test_default_soda_lime(self, soda_lime_composition: GlassComposition) -> None:
        """Test default soda-lime glass composition."""
        assert soda_lime_composition.sio2 == 74.0
        assert soda_lime_composition.na2o == 13.0
        assert soda_lime_composition.cao == 10.5
        assert soda_lime_composition.fe2o3 == 0.1

    def test_composition_validation_valid(
        self, soda_lime_composition: GlassComposition
    ) -> None:
        """Test that valid composition passes validation."""
        assert soda_lime_composition.validate()

    def test_composition_validation_invalid_low(self) -> None:
        """Test that composition summing to less than 99% fails."""
        from glass_bath_fea.core.config import GlassComposition

        invalid = GlassComposition(
            sio2=50.0,
            na2o=10.0,
            cao=10.0,
            mgo=0.0,
            al2o3=0.0,
            fe2o3=0.0,
        )  # Total = 70%

        assert not invalid.validate()

    def test_composition_validation_invalid_high(self) -> None:
        """Test that composition summing to more than 101% fails."""
        from glass_bath_fea.core.config import GlassComposition

        invalid = GlassComposition(
            sio2=80.0,
            na2o=15.0,
            cao=10.0,
            mgo=5.0,
            al2o3=0.0,
            fe2o3=0.0,
        )  # Total = 110%

        assert not invalid.validate()

    def test_total_percent(self, soda_lime_composition: GlassComposition) -> None:
        """Test that total percentage is computed correctly."""
        total = soda_lime_composition.total_percent()
        # 74 + 13 + 10.5 + 0 + 1.5 + 0.1 = 99.1
        assert total == pytest.approx(99.1, rel=1e-6)


class TestMeshConfig:
    """Tests for mesh configuration."""

    def test_default_element_sizes(self) -> None:
        """Test default mesh element sizes."""
        from glass_bath_fea.core.config import MeshConfig

        config = MeshConfig()

        # Default sizes should follow: electrodes < metal < glass
        assert config.element_size_electrodes < config.element_size_metal
        assert config.element_size_metal < config.element_size_glass

    def test_mesh_order_validation(self) -> None:
        """Test that mesh order must be 1 or 2."""
        from glass_bath_fea.core.config import MeshConfig

        # Valid orders
        config1 = MeshConfig(mesh_order=1)
        assert config1.mesh_order == 1

        config2 = MeshConfig(mesh_order=2)
        assert config2.mesh_order == 2

    def test_algorithm_options(self) -> None:
        """Test available mesh algorithms."""
        from glass_bath_fea.core.config import MeshConfig

        config = MeshConfig(mesh_algorithm="delaunay")
        assert config.mesh_algorithm == "delaunay"

        config = MeshConfig(mesh_algorithm="frontal")
        assert config.mesh_algorithm == "frontal"

    def test_export_format_default(self) -> None:
        """Test default export format is MSH v2.2."""
        from glass_bath_fea.core.config import MeshConfig

        config = MeshConfig()
        assert config.export_format == "msh22"
