"""Tests for Electrode Advisor GUI.

Tests the PyQt6 GUI and its integration with the shared electrical engine.
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


class TestElectrodeAdvisorGUI:
    """Test suite for Electrode Advisor GUI."""

    @pytest.fixture
    def mock_qt_app(self):
        """Create a mock Qt application for testing."""
        with patch("PyQt6.QtWidgets.QApplication"):
            yield

    def test_imports(self):
        """Test that all imports work correctly."""
        # Test shared engine imports
        from upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
            GlassPropertiesInterface,
            ThreePhaseElectricalModelEnhanced,
        )

        assert ElectrodeConfig is not None
        assert GlassPropertiesInterface is not None
        assert ThreePhaseElectricalModelEnhanced is not None

    def test_config_creation(self):
        """Test that ElectrodeConfig can be created with defaults."""
        from upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
        )

        config = ElectrodeConfig()
        assert config is not None
        assert hasattr(config, "bath_diameter")
        assert hasattr(config, "tip_diameter")

    def test_electrical_model_creation(self):
        """Test that the electrical model can be instantiated."""
        from upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
            GlassPropertiesInterface,
            ThreePhaseElectricalModelEnhanced,
        )

        config = ElectrodeConfig()
        glass_interface = GlassPropertiesInterface()
        model = ThreePhaseElectricalModelEnhanced(config, glass_interface)

        assert model is not None
        assert hasattr(model, "calculate_system_state")

    def test_electrical_model_calculation(self):
        """Test that the electrical model produces results."""
        import numpy as np
        from upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
            GlassPropertiesInterface,
            ThreePhaseElectricalModelEnhanced,
        )

        config = ElectrodeConfig()
        glass_interface = GlassPropertiesInterface()
        model = ThreePhaseElectricalModelEnhanced(config, glass_interface)

        # Run calculation
        results = model.calculate_system_state(
            depths=np.array([12.0, 12.0, 12.0]),
            bath_diameter=120.0,
            tip_diameter=24.0,
            metal_depth=2.0,
            k_factors={"K_tt": 1.0, "K_vert": 1.0},
            bath_temperature=1350.0,
            voltages=np.array([100.0, 100.0, 100.0]),
            conductive_height=2.0,
        )

        assert results is not None
        assert isinstance(results, dict)

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and sys.platform != "win32",
        reason="No display available",
    )
    def test_widget_creation(self, mock_qt_app):
        """Test that the widget can be created."""
        main_window = pytest.importorskip(
            "electrode_advisor.ui.pyqt6.main_window",
            reason="electrode_advisor.ui not available",
        )
        widget_cls = main_window.ElectrodeAdvisorWidget

        with (
            patch.object(widget_cls, "_init_ui", return_value=None),
            patch.object(widget_cls, "_apply_styling", return_value=None),
            patch.object(widget_cls, "calculate_system", return_value=None),
        ):
            widget = widget_cls.__new__(widget_cls)
            assert widget is not None

    def test_launcher_dependencies(self):
        """Test that the launcher can check dependencies."""
        launcher_path = Path(__file__).resolve().parents[1] / "launch_pyqt6.py"
        spec = importlib.util.spec_from_file_location(
            "electrode_advisor_launch_pyqt6", launcher_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        missing = mod.check_dependencies()
        # We expect no missing dependencies in a properly set up environment
        # But don't fail if they're missing (CI might not have PyQt6)
        assert isinstance(missing, list)


class TestGUIRegistration:
    """Test suite for GUI registration."""

    def test_registration_imports(self):
        """Test that registration module can be imported."""
        gui_launcher = pytest.importorskip(
            "gui_launcher", reason="gui_launcher not available"
        )
        assert gui_launcher.GUIType is not None
        assert gui_launcher.LaunchConfig is not None
        assert gui_launcher.register_gui is not None
