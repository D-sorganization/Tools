"""Tests for Syngas Compression Calculator GUI.

Tests the PyQt6 GUI launcher and its integration with the shared engine.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Set headless mode for testing
os.environ["HEADLESS"] = "true"


class TestSyngasCompressionEngine:
    """Test suite for Syngas Compression Engine integration."""

    def test_engine_imports(self) -> None:
        """Test that all engine imports work correctly."""
        from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
            CompressionStage,
            SyngasCompressionEngine,
        )

        assert SyngasCompressionEngine is not None
        assert CompressionStage is not None

    def test_compression_stage_creation(self) -> None:
        """Test that CompressionStage can be created."""
        from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
            CompressionStage,
        )

        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,  # 40°C in K
            efficiency=0.75,
            compression_type="isentropic",
        )

        assert stage.inlet_pressure == 1.0
        assert stage.outlet_pressure == 3.0
        assert stage.efficiency == 0.75
        assert stage.compression_type == "isentropic"

    def test_engine_calculation(self) -> None:
        """Test that the engine produces valid results."""
        from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()

        # Test mixture properties calculation
        composition = {
            "H2": 0.20,
            "CO": 0.25,
            "CO2": 0.15,
            "CH4": 0.05,
            "N2": 0.30,
            "H2O": 0.05,
        }

        mixture_props = engine.calculate_mixture_properties(composition)

        assert mixture_props is not None
        assert "molecular_weight" in mixture_props
        assert "gamma" in mixture_props
        assert mixture_props["molecular_weight"] > 0
        assert 1.2 < mixture_props["gamma"] < 1.5  # Reasonable gamma range

    def test_compression_work_calculation(self) -> None:
        """Test compression work calculation."""
        from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
            CompressionStage,
            SyngasCompressionEngine,
        )

        engine = SyngasCompressionEngine()

        composition = {
            "H2": 0.20,
            "CO": 0.25,
            "CO2": 0.15,
            "CH4": 0.05,
            "N2": 0.30,
            "H2O": 0.05,
        }

        mixture_props = engine.calculate_mixture_properties(composition)

        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.75,
            compression_type="isentropic",
        )

        result = engine.calculate_compression_work(
            stage=stage,
            flow_rate=100.0,  # kmol/h
            mixture_props=mixture_props,
        )

        assert result is not None
        assert "work" in result
        assert "power_hp" in result
        assert result["work"] > 0
        assert result["power_hp"] > 0


class TestSyngasCompressionGUI:
    """Test suite for Syngas Compression Calculator GUI."""

    @pytest.fixture
    def mock_qt_app(self):
        """Create a mock Qt application for testing."""
        with patch("PyQt6.QtWidgets.QApplication"):
            yield

    def test_launcher_dependencies(self) -> None:
        """Test that the launcher can check dependencies."""
        launcher_path = Path(__file__).resolve().parents[1] / "launch_pyqt6.py"
        spec = importlib.util.spec_from_file_location(
            "syngas_compression_launch_pyqt6", launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        missing = mod.check_dependencies()
        assert isinstance(missing, list)

    def test_module_imports(self) -> None:
        """Test that module imports work correctly."""
        from syngas_compression import CompressionStage, SyngasCompressionEngine

        assert CompressionStage is not None
        assert SyngasCompressionEngine is not None

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and sys.platform != "win32",
        reason="No display available",
    )
    def test_factory_function_exists(self, mock_qt_app) -> None:
        """Test that the factory function exists and is callable."""
        try:
            from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
                create_syngas_compression_calculator,
            )

            assert callable(create_syngas_compression_calculator)
        except ImportError as e:
            pytest.skip(f"Qt not available: {e}")


class TestGUIRegistration:
    """Test suite for GUI registration."""

    def test_registration_imports(self) -> None:
        """Test that registration module can be imported."""
        try:
            from gui_launcher import (
                GUIType,
                LaunchConfig,
                register_gui,
            )

            assert GUIType is not None
            assert LaunchConfig is not None
            assert register_gui is not None
        except ImportError as e:
            pytest.skip(f"GUI launcher not available: {e}")
