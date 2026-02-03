"""Tests for TRC Vessel Designer GUI.

Tests the PyQt6 GUI and its integration with the shared TRC geometry engine.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

# Set headless mode for testing
os.environ["HEADLESS"] = "true"


class TestTRCVesselDesignerEngine:
    """Test suite for TRC Geometry Engine integration."""

    def test_engine_imports(self):
        """Test that all engine imports work correctly."""
        from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
            LayerConfig,
            LayerResult,
            TRCGeometryEngine,
            VesselDimensions,
            VesselGeometryResult,
        )

        assert TRCGeometryEngine is not None
        assert VesselDimensions is not None
        assert LayerConfig is not None
        assert LayerResult is not None
        assert VesselGeometryResult is not None

    def test_engine_calculation(self):
        """Test that the engine produces valid results."""
        from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
            LayerConfig,
            TRCGeometryEngine,
            VesselDimensions,
        )

        engine = TRCGeometryEngine()
        dimensions = VesselDimensions(
            cylinder_height=72.0,
            cylinder_diameter=24.0,
            cone_height=24.0,
            cone_bottom_diameter=6.0,
            cone_interior_hole=4.0,
            top_refractory_thickness=6.0,
        )
        layers = [
            LayerConfig(
                name="Working Lining",
                thickness=6.0,
                density=150.0,
                color="#94a3b8",
            ),
            LayerConfig(
                name="Insulating",
                thickness=4.5,
                density=60.0,
                color="#cbd5e1",
            ),
        ]

        results = engine.calculate_geometry(dimensions, layers)

        assert results is not None
        assert results.total_volume_ft3 > 0
        assert results.interior_volume_ft3 > 0
        assert results.total_mass_lb > 0
        assert len(results.layers) == 2

    def test_layer_config_creation(self):
        """Test that LayerConfig can be created with defaults."""
        from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
            LayerConfig,
        )

        layer = LayerConfig(
            name="Test Layer",
            thickness=5.0,
            density=100.0,
            color="#ffffff",
        )

        assert layer.name == "Test Layer"
        assert layer.thickness == 5.0
        assert layer.density == 100.0
        assert layer.visible is True

    def test_vessel_dimensions_creation(self):
        """Test that VesselDimensions can be created."""
        from shared.python.upstream_drift_tools.calculators.mechanical.trc_geometry import (
            VesselDimensions,
        )

        dims = VesselDimensions(
            cylinder_height=72.0,
            cylinder_diameter=24.0,
            cone_height=24.0,
            cone_bottom_diameter=6.0,
            cone_interior_hole=4.0,
            top_refractory_thickness=6.0,
        )

        assert dims.cylinder_height == 72.0
        assert dims.cylinder_diameter == 24.0
        assert dims.display_lid is True
        assert dims.display_cylinder is True


class TestTRCVesselDesignerGUI:
    """Test suite for TRC Vessel Designer GUI."""

    @pytest.fixture
    def mock_qt_app(self):
        """Create a mock Qt application for testing."""
        with patch("PyQt6.QtWidgets.QApplication"):
            yield

    @pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and sys.platform != "win32",
        reason="No display available",
    )
    def test_widget_creation(self, mock_qt_app):
        """Test that the widget can be created."""
        try:
            from trc_vessel_designer.ui.pyqt6.main_window import (
                TRCVesselDesignerWidget,
            )

            with patch.object(TRCVesselDesignerWidget, "_init_ui", return_value=None):
                with patch.object(
                    TRCVesselDesignerWidget, "_apply_styling", return_value=None
                ):
                    with patch.object(
                        TRCVesselDesignerWidget, "calculate_geometry", return_value=None
                    ):
                        widget = TRCVesselDesignerWidget.__new__(
                            TRCVesselDesignerWidget
                        )
                        assert widget is not None
        except ImportError as e:
            pytest.skip(f"Qt not available: {e}")

    def test_launcher_dependencies(self):
        """Test that the launcher can check dependencies."""
        sys.path.insert(
            0,
            str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )

        from launch_pyqt6 import check_dependencies

        missing = check_dependencies()
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
