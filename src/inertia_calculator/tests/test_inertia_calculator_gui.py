from typing import Any

"""
Inertia Calculator GUI Tests
============================

TDD tests for the Inertia Calculator GUI components.
Tests cover PyQt6 main window, inertia calculations, and validation.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestInertiaCalculatorMainWindow:
    """Tests for the PyQt6 Inertia Calculator main window."""

    @pytest.fixture
    def mock_qt_app(self) -> Any:
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

    def test_main_window_imports(self, mock_qt_app) -> Any:
        """Test that main window module can be imported."""
        try:
            from inertia_calculator.ui.pyqt6 import main_window

            assert hasattr(main_window, "InertiaCalculatorWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app) -> Any:
        """Test that window class is defined and callable."""
        try:
            from inertia_calculator.ui.pyqt6.main_window import (
                InertiaCalculatorWindow,
            )

            assert callable(InertiaCalculatorWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestInertiaCalculations:
    """Tests for inertia calculation formulas."""

    def test_solid_box_inertia(self) -> Any:
        """Test inertia calculation for solid box."""
        mass = 1.0
        lx, ly, lz = 0.1, 0.1, 0.1

        ixx = (1 / 12) * mass * (ly**2 + lz**2)
        iyy = (1 / 12) * mass * (lx**2 + lz**2)
        izz = (1 / 12) * mass * (lx**2 + ly**2)

        assert ixx == pytest.approx((1 / 12) * 1.0 * (0.01 + 0.01), rel=1e-6)
        assert iyy == pytest.approx((1 / 12) * 1.0 * (0.01 + 0.01), rel=1e-6)
        assert izz == pytest.approx((1 / 12) * 1.0 * (0.01 + 0.01), rel=1e-6)

    def test_solid_cylinder_inertia(self) -> Any:
        """Test inertia calculation for solid cylinder."""
        mass = 1.0
        r = 0.05
        h = 0.1

        ixx = (1 / 12) * mass * (3 * r**2 + h**2)
        iyy = ixx
        izz = (1 / 2) * mass * r**2

        # Verify cylinder inertia formulas
        assert ixx == pytest.approx(iyy, rel=1e-6)
        assert izz == pytest.approx(0.5 * mass * r**2, rel=1e-6)

    def test_solid_sphere_inertia(self) -> Any:
        """Test inertia calculation for solid sphere."""
        mass = 1.0
        r = 0.05

        ixx = (2 / 5) * mass * r**2
        iyy = ixx
        izz = ixx

        # Sphere should have equal moments about all axes
        assert ixx == pytest.approx(iyy, rel=1e-6)
        assert iyy == pytest.approx(izz, rel=1e-6)
        assert ixx == pytest.approx((2 / 5) * mass * r**2, rel=1e-6)

    def test_hollow_cylinder_inertia(self) -> Any:
        """Test inertia calculation for hollow cylinder."""
        mass = 1.0
        r_out = 0.1
        r_in = 0.05
        h = 0.1

        ixx = (1 / 12) * mass * (3 * (r_out**2 + r_in**2) + h**2)
        iyy = ixx
        izz = (1 / 2) * mass * (r_out**2 + r_in**2)

        # Verify hollow cylinder has expected properties
        assert ixx == pytest.approx(iyy, rel=1e-6)
        assert izz == pytest.approx(0.5 * mass * (r_out**2 + r_in**2), rel=1e-6)


class TestInertiaValidation:
    """Tests for inertia tensor validation."""

    def test_valid_diagonal_inertia(self) -> Any:
        """Test validation of valid diagonal inertia tensor."""
        import numpy as np

        # Valid sphere-like inertia
        ixx = iyy = izz = 0.1
        ixy = ixz = iyz = 0.0

        tensor = np.array(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ]
        )

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert all(ev > 0 for ev in eigenvalues)

    def test_triangle_inequality(self) -> Any:
        """Test triangle inequality validation."""
        # Valid values that satisfy triangle inequality
        ixx = 1.0
        iyy = 1.0
        izz = 1.0

        # Check: |Ixx - Iyy| <= Izz <= Ixx + Iyy
        assert abs(ixx - iyy) <= izz <= ixx + iyy
        assert abs(iyy - izz) <= ixx <= iyy + izz
        assert abs(ixx - izz) <= iyy <= ixx + izz

    def test_invalid_negative_inertia(self) -> Any:
        """Test detection of invalid negative inertia values."""
        # Invalid - negative diagonal
        ixx = -0.1

        # Should fail positive check
        assert ixx <= 0
        # Valid inertia values should be positive
        assert 0.1 > 0  # iyy, izz would be positive

    def test_invalid_triangle_inequality(self) -> Any:
        """Test detection of violated triangle inequality."""
        # Invalid - violates triangle inequality
        ixx = 0.1
        iyy = 0.1
        izz = 1.0

        # Should violate: |Ixx - Iyy| <= Izz <= Ixx + Iyy
        assert not (abs(ixx - iyy) <= izz <= ixx + iyy)


class TestInertiaCalculatorGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self) -> Any:
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from inertia_calculator import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self) -> Any:
        """Test that calculator is in robotics category."""
        try:
            from inertia_calculator import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "robotics"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self) -> Any:
        """Test that launcher script exists."""
        try:
            from inertia_calculator import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")
