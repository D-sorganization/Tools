"""Tests for electrode adviser adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Bootstrap for test discovery
_REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)

if TYPE_CHECKING:
    from glass_bath_fea.core.config import GlassBathFEAConfig


class TestElectrodeAdapter:
    """Tests for the electrode adviser adapter interface."""

    def test_adapter_initialization(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test that adapter can be initialized with FEA config."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        assert adapter is not None
        assert adapter.config is not None

    def test_get_electrode_config(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test conversion to electrode adviser config format."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)
        electrode_config = adapter.get_electrode_config()

        # Should return compatible config object
        assert hasattr(electrode_config, "bath_diameter")
        assert hasattr(electrode_config, "glass_depth")

    def test_get_glass_interface(self, default_fea_config: GlassBathFEAConfig) -> None:
        """Test getting glass properties interface."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)
        glass_interface = adapter.get_glass_interface()

        # Should have conductivity method
        assert hasattr(glass_interface, "get_conductivity")

    def test_calculate_electrode_positions(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test using electrode adviser to calculate positions."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        # Default depths (10 inches each for 3 electrodes)
        depths = np.array(
            [
                default_fea_config.electrode_insertion_depth,
                default_fea_config.electrode_insertion_depth,
                default_fea_config.electrode_insertion_depth,
            ]
        )

        positions = adapter.calculate_electrode_positions(depths)

        assert len(positions) == 3
        for pos in positions:
            assert "tip" in pos
            assert "base" in pos

    def test_conductivity_from_adapter(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test getting conductivity through adapter."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        # Get conductivity at operating temperature
        temp = default_fea_config.operating_temperature
        sigma = adapter.get_glass_conductivity(temp)

        # Should be positive
        assert sigma > 0

    def test_resistivity_calculation(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test resistance calculation using electrode adviser model."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        # Calculate phase resistances at default configuration
        resistances = adapter.calculate_phase_resistances()

        # Should have 3 phase-to-phase resistances
        assert "1-2" in resistances or len(resistances) >= 3


class TestAdapterCompatibility:
    """Tests for compatibility with electrode adviser code."""

    def test_electrical_model_creation(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test creating electrical model through adapter."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)
        model = adapter.get_electrical_model()

        # Should have the calculate_system_state method
        assert hasattr(model, "calculate_system_state")

    def test_system_state_calculation(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test full system state calculation through adapter."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        results = adapter.calculate_system_state()

        # Should return dict with key results
        assert results is not None
        assert isinstance(results, dict)

    def test_temperature_effect_on_conductivity(
        self, default_fea_config: GlassBathFEAConfig
    ) -> None:
        """Test that temperature affects conductivity through adapter."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)

        sigma_low = adapter.get_glass_conductivity(1200.0)
        sigma_high = adapter.get_glass_conductivity(1400.0)

        # Higher temperature should give higher conductivity
        assert sigma_high > sigma_low


class TestDataExport:
    """Tests for exporting data to FEA format."""

    def test_export_boundary_conditions(
        self, default_fea_config: GlassBathFEAConfig, tmp_path: Path
    ) -> None:
        """Test exporting boundary condition data."""
        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)
        output_path = tmp_path / "boundary_conditions.mat"

        adapter.export_boundary_conditions(output_path)

        assert output_path.exists()

    def test_export_contains_voltages(
        self, default_fea_config: GlassBathFEAConfig, tmp_path: Path
    ) -> None:
        """Test that exported data contains electrode voltages."""
        from scipy.io import loadmat

        from glass_bath_fea.interfaces.electrode_adapter import ElectrodeAdapter

        adapter = ElectrodeAdapter(default_fea_config)
        output_path = tmp_path / "boundary_conditions.mat"

        adapter.export_boundary_conditions(output_path)

        data = loadmat(output_path)

        # Should have electrode voltages
        assert "electrode_voltages" in data or "phase_voltages" in data
