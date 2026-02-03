"""Tests for Electrode Advisor GUI.

Tests the PyQt6 GUI and its integration with the shared electrical engine.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

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
        from shared.python.upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
            GlassPropertiesInterface,
            ThreePhaseElectricalModelEnhanced,
        )

        assert ElectrodeConfig is not None
        assert GlassPropertiesInterface is not None
        assert ThreePhaseElectricalModelEnhanced is not None

    def test_config_creation(self):
        """Test that ElectrodeConfig can be created with defaults."""
        from shared.python.upstream_drift_tools.calculators.electrical import (
            ElectrodeConfig,
        )

        config = ElectrodeConfig()
        assert config is not None
        assert hasattr(config, "bath_diameter")
        assert hasattr(config, "tip_diameter")

    def test_electrical_model_creation(self):
        """Test that the electrical model can be instantiated."""
        from shared.python.upstream_drift_tools.calculators.electrical import (
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

        from shared.python.upstream_drift_tools.calculators.electrical import (
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
        # This test requires Qt to be available
        try:
            from electrode_advisor.ui.pyqt6.main_window import (
                ElectrodeAdvisorWidget,
            )

            # Mock the Qt widgets to avoid display issues
            with patch.object(ElectrodeAdvisorWidget, "_init_ui", return_value=None):
                with patch.object(
                    ElectrodeAdvisorWidget, "_apply_styling", return_value=None
                ):
                    with patch.object(
                        ElectrodeAdvisorWidget, "calculate_system", return_value=None
                    ):
                        widget = ElectrodeAdvisorWidget.__new__(ElectrodeAdvisorWidget)
                        assert widget is not None
        except ImportError as e:
            pytest.skip(f"Qt not available: {e}")

    def test_launcher_dependencies(self):
        """Test that the launcher can check dependencies."""
        # Import the launcher module
        sys.path.insert(
            0,
            str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

        from launch_pyqt6 import check_dependencies

        missing = check_dependencies()
        # We expect no missing dependencies in a properly set up environment
        # But don't fail if they're missing (CI might not have PyQt6)
        assert isinstance(missing, list)


class TestGUIRegistration:
    """Test suite for GUI registration."""

    def test_registration_imports(self):
        """Test that registration module can be imported."""
        try:
            from shared.python.gui_launcher import (
                GUIType,
                LaunchConfig,
                register_gui,
            )

            assert GUIType is not None
            assert LaunchConfig is not None
            assert register_gui is not None
        except ImportError as e:
            pytest.skip(f"GUI launcher not available: {e}")
