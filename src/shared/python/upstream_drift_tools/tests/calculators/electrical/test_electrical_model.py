"""test_electrical_model.py module."""

import numpy as np
import pytest
from upstream_drift_tools.calculators.electrical.config import ElectrodeConfig
from upstream_drift_tools.calculators.electrical.electrical_model import (
    ThreePhaseElectricalModelEnhanced,
)
from upstream_drift_tools.calculators.electrical.glass_interface import (
    GlassPropertiesInterface,
)


class TestElectricalModel:
    @pytest.fixture
    def model(self):
        config = ElectrodeConfig()
        glass = GlassPropertiesInterface()
        return ThreePhaseElectricalModelEnhanced(config, glass)

    def test_initialization(self, model):
        assert len(model.electrode_positions) == 3
        np.testing.assert_allclose(
            model.electrode_positions, [0, 2.094395, 4.18879], rtol=1e-4
        )

    def test_calculate_system_state(self, model):
        depths = np.array([10.0, 10.0, 10.0])
        voltages = np.array([100.0, 100.0, 100.0])

        result = model.calculate_system_state(
            depths=depths,
            bath_diameter=120.0,
            tip_diameter=24.0,
            metal_depth=2.0,
            k_factors={"K_tt": 1.0},
            bath_temperature=1200.0,
            voltages=voltages,
        )

        assert "resistances" in result
        assert "current_paths" in result
        assert "current_distribution" in result
        assert len(result["actual_currents"]) == 3

    def test_parallel_resistance(self, model):
        r1 = 100.0
        r2 = 100.0
        rp = model._parallel_resistance(r1, r2)
        assert rp == 50.0  # Parallel of two equal resistors is half

    def test_glass_conductivity(self):
        glass = GlassPropertiesInterface()

        # Test default Arrhenius behavior
        cond_1200 = glass.get_conductivity(1200.0)  # Base temp (1473.15 K)
        cond_1300 = glass.get_conductivity(1300.0)  # Higher temp

        assert cond_1300 > cond_1200  # Conductivity should increase with temp

        # Test metal conductivity
        cond_metal = glass.get_conductivity(1200.0, is_metal=True)
        assert cond_metal == 10000.0


class TestElectrodeConfigMethods:
    """Tests for ElectrodeConfig.status_color and scheme_color methods."""

    def test_status_color_ok(self):
        cfg = ElectrodeConfig()
        assert cfg.status_color("ok") == "#C8FFC8"

    def test_status_color_warn(self):
        cfg = ElectrodeConfig()
        assert cfg.status_color("warn") == "#FFFFB4"

    def test_status_color_error(self):
        cfg = ElectrodeConfig()
        assert cfg.status_color("error") == "#FF9696"

    def test_status_color_unknown_falls_back_to_ok(self):
        cfg = ElectrodeConfig()
        result = cfg.status_color("unknown_type")
        # Falls back to 'status_ok' key
        assert result == "#C8FFC8"

    def test_status_color_colors_none(self):
        """When colors dict is None, returns fallback."""
        cfg = ElectrodeConfig(colors=None)
        cfg.__post_init__()
        cfg.colors = None  # Force None after init
        assert cfg.status_color("ok") == "#C8FFC8"

    def test_scheme_color_default_scheme(self):
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("default", "direct_glass")
        assert color == "#4169E1"

    def test_scheme_color_missing_path_type_returns_lightblue(self):
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("default", "nonexistent_path")
        assert color == "lightblue"

    def test_scheme_color_missing_scheme_returns_lightblue(self):
        cfg = ElectrodeConfig()
        color = cfg.scheme_color("nonexistent_scheme", "direct_glass")
        assert color == "lightblue"

    def test_scheme_color_color_schemes_none(self):
        """When color_schemes is None, returns 'lightblue'."""
        cfg = ElectrodeConfig()
        cfg.__post_init__()
        cfg.color_schemes = None
        assert cfg.scheme_color("default", "direct_glass") == "lightblue"
