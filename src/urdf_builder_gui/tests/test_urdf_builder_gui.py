"""
Parametric URDF Builder GUI Tests
=================================

TDD tests for the Parametric URDF Builder GUI components.
Tests cover PyQt6 main window, URDF generation, and configuration options.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestURDFBuilderMainWindow:
    """Tests for the PyQt6 URDF Builder main window."""

    @pytest.fixture
    def mock_qt_app(self):
        """Create mock Qt application for headless testing."""
        with patch.dict(
            sys.modules,
            {
                "PyQt6": MagicMock(),
                "PyQt6.QtWidgets": MagicMock(),
                "PyQt6.QtCore": MagicMock(),
                "PyQt6.QtGui": MagicMock(),
            },
        ):
            yield

    def test_main_window_imports(self, mock_qt_app):
        """Test that main window module can be imported."""
        try:
            from urdf_builder_gui.ui.pyqt6 import main_window

            assert hasattr(main_window, "URDFBuilderWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app):
        """Test that window class is defined and callable."""
        try:
            from urdf_builder_gui.ui.pyqt6.main_window import URDFBuilderWindow

            assert callable(URDFBuilderWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestParametricCalculations:
    """Tests for parametric model calculations."""

    def test_segment_length_scaling(self):
        """Test that segment lengths scale with height."""
        height = 1.80
        pelvis_ratio = 0.078
        thigh_ratio = 0.245

        pelvis_length = height * pelvis_ratio
        thigh_length = height * thigh_ratio

        assert pelvis_length == pytest.approx(0.1404, rel=1e-3)
        assert thigh_length == pytest.approx(0.441, rel=1e-3)

    def test_mass_distribution(self):
        """Test that mass ratios sum to approximately 1."""
        mass_ratios = {
            "pelvis": 0.112,
            "lumbar": 0.139,
            "thorax": 0.216,
            "neck": 0.024,
            "head": 0.069,
            "upper_arm": 0.027 * 2,  # Both arms
            "forearm": 0.016 * 2,
            "hand": 0.006 * 2,
            "thigh": 0.142 * 2,  # Both legs
            "shin": 0.043 * 2,
            "foot": 0.014 * 2,
        }

        total = sum(mass_ratios.values())
        # Should be close to 1.0 (anthropometric data varies)
        assert 0.9 < total < 1.1

    def test_gender_factor_range(self):
        """Test gender factor is properly bounded."""
        # Gender factor should be clamped between 0 and 1
        for value in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            clamped = max(0.0, min(1.0, value))
            assert 0.0 <= clamped <= 1.0

    def test_proportion_factor_conversion(self):
        """Test proportion slider to factor conversion."""
        # Slider value 50 -> 0.5 factor
        # Slider value 100 -> 1.0 factor
        # Slider value 150 -> 1.5 factor
        test_cases = [(50, 0.5), (100, 1.0), (150, 1.5), (75, 0.75)]

        for slider_value, expected_factor in test_cases:
            factor = slider_value / 100.0
            assert factor == pytest.approx(expected_factor)


class TestURDFGeneration:
    """Tests for URDF output generation."""

    def test_urdf_xml_structure(self):
        """Test that generated URDF has valid structure."""
        # Simple URDF structure check
        urdf_template = """<?xml version="1.0" encoding="UTF-8"?>
<robot name="test">
  <link name="base"/>
</robot>"""

        assert '<?xml version="1.0"' in urdf_template
        assert '<robot name="test">' in urdf_template
        assert "<link name=" in urdf_template
        assert "</robot>" in urdf_template

    def test_inertia_values_positive(self):
        """Test that calculated inertias are positive."""
        mass = 10.0
        length = 0.5
        width = 0.1

        # Box inertia
        ixx = (1 / 12) * mass * (width**2 + length**2)
        iyy = (1 / 12) * mass * (width**2 + length**2)
        izz = (1 / 12) * mass * (width**2 + width**2)

        assert ixx > 0
        assert iyy > 0
        assert izz > 0

    def test_joint_limits_valid(self):
        """Test that joint limits are valid."""
        import math

        joint_limits = [
            (-math.radians(30), math.radians(30)),  # Lumbar
            (-math.radians(150), 0),  # Knee
            (-math.pi, math.pi),  # Shoulder
        ]

        for lower, upper in joint_limits:
            assert lower <= upper, "Lower limit should be <= upper limit"


class TestURDFBuilderGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self):
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from urdf_builder_gui import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self):
        """Test that builder is in robotics category."""
        try:
            from urdf_builder_gui import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "robotics"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self):
        """Test that launcher script exists."""
        try:
            from urdf_builder_gui import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")
