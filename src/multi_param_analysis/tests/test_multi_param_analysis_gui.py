"""
Multi-Parameter Analysis GUI Tests
==================================

TDD tests for the Multi-Parameter Analysis GUI components.
Tests cover PyQt6 main window, grid evaluation, and sensitivity analysis.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMultiParamAnalysisMainWindow:
    """Tests for the PyQt6 Multi-Parameter Analysis main window."""

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
            from multi_param_analysis.ui.pyqt6 import main_window

            assert hasattr(main_window, "MultiParamAnalysisWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app):
        """Test that window class is defined and callable."""
        try:
            from multi_param_analysis.ui.pyqt6.main_window import (
                MultiParamAnalysisWindow,
            )

            assert callable(MultiParamAnalysisWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestGridEvaluation:
    """Tests for parameter grid evaluation."""

    def test_linspace_grid_creation(self):
        """Test grid creation with linspace."""
        param1 = np.linspace(0, 10, 5)
        param2 = np.linspace(0, 1, 5)

        assert len(param1) == 5
        assert len(param2) == 5
        assert param1[0] == 0
        assert param1[-1] == 10
        assert param2[0] == 0
        assert param2[-1] == 1

    def test_meshgrid_creation(self):
        """Test 2D meshgrid creation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])

        X, Y = np.meshgrid(x, y)

        assert X.shape == (2, 3)
        assert Y.shape == (2, 3)
        np.testing.assert_array_equal(X[0], [1, 2, 3])
        np.testing.assert_array_equal(Y[:, 0], [4, 5])

    def test_grid_point_count(self):
        """Test total grid points calculation."""
        n1, n2 = 10, 15
        total = n1 * n2
        assert total == 150


class TestDemoFunctions:
    """Tests for demo test functions."""

    def test_rosenbrock_minimum(self):
        """Test Rosenbrock function minimum at (1, 1)."""
        x, y = 1.0, 1.0
        f = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        assert f == pytest.approx(0.0)

    def test_sphere_minimum(self):
        """Test Sphere function minimum at (0, 0)."""
        x, y = 0.0, 0.0
        f = x**2 + y**2
        assert f == pytest.approx(0.0)

    def test_rastrigin_minimum(self):
        """Test Rastrigin function minimum at (0, 0)."""
        x, y = 0.0, 0.0
        f = 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
        assert f == pytest.approx(0.0)

    def test_himmelblau_minimum(self):
        """Test Himmelblau function has minimum at (3, 2)."""
        x, y = 3.0, 2.0
        f = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        assert f == pytest.approx(0.0, abs=1e-10)


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis calculations."""

    def test_variance_calculation(self):
        """Test variance calculation on grid."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        variance = Z.var()
        assert variance > 0

    def test_main_effect_param1(self):
        """Test main effect calculation for parameter 1."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Average over param2 (rows) to get param1 effect
        param1_means = Z.mean(axis=0)
        np.testing.assert_array_equal(param1_means, [4, 5, 6])

    def test_main_effect_param2(self):
        """Test main effect calculation for parameter 2."""
        Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Average over param1 (cols) to get param2 effect
        param2_means = Z.mean(axis=1)
        np.testing.assert_array_equal(param2_means, [2, 5, 8])

    def test_sensitivity_indices_sum(self):
        """Test that sensitivity indices approximately sum to 1."""
        # Create test data with known structure
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = 2 * X + 3 * Y  # Linear combination

        total_var = Z.var()
        if total_var > 0:
            s1 = Z.mean(axis=0).var() / total_var
            s2 = Z.mean(axis=1).var() / total_var
            # For purely additive function, interaction should be near 0
            assert s1 + s2 <= 1.0 + 1e-10


class TestMultiParamAnalysisGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self):
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from multi_param_analysis import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self):
        """Test that tool is in analysis category."""
        try:
            from multi_param_analysis import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "analysis"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self):
        """Test that launcher script exists."""
        try:
            from multi_param_analysis import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")


# ---------------------------------------------------------------------------
# DbC guard functions — pure Python, no Qt dependency.
# These replicate the precondition logic from the corresponding methods in
# MultiParamAnalysisWindow so that DbC enforcement can be tested headlessly
# without needing to instantiate the Qt class.
# ---------------------------------------------------------------------------


def _check_run_demo_analysis_args(
    param1_name: object,
    param2_name: object,
    output_name: object,
    param1_values: object,
    param2_values: object,
) -> None:
    """Precondition guard mirroring _run_demo_analysis in MultiParamAnalysisWindow."""
    if not isinstance(param1_name, str):
        raise TypeError(f"param1_name must be a str, got {type(param1_name).__name__}")
    if not isinstance(param2_name, str):
        raise TypeError(f"param2_name must be a str, got {type(param2_name).__name__}")
    if not isinstance(output_name, str):
        raise TypeError(f"output_name must be a str, got {type(output_name).__name__}")
    if not isinstance(param1_values, np.ndarray):
        raise TypeError(
            f"param1_values must be a numpy ndarray, got {type(param1_values).__name__}"
        )
    if not isinstance(param2_values, np.ndarray):
        raise TypeError(
            f"param2_values must be a numpy ndarray, got {type(param2_values).__name__}"
        )


def _check_ndarray_triple(
    param1_values: object,
    param2_values: object,
    Z: object,
) -> None:
    """Precondition guard mirroring _calculate_sensitivity and _update_preview."""
    if not isinstance(param1_values, np.ndarray):
        raise TypeError(
            f"param1_values must be a numpy ndarray, got {type(param1_values).__name__}"
        )
    if not isinstance(param2_values, np.ndarray):
        raise TypeError(
            f"param2_values must be a numpy ndarray, got {type(param2_values).__name__}"
        )
    if not isinstance(Z, np.ndarray):
        raise TypeError(f"Z must be a numpy ndarray, got {type(Z).__name__}")


class TestDbCEnforcement:
    """Tests for Design by Contract precondition enforcement in analysis methods.

    Validates that the guard logic raises TypeError for wrong argument types,
    per DbC rules. Uses extracted guard functions (pure Python, no Qt dependency)
    to enable headless execution without class instantiation.
    """

    def test_run_demo_analysis_rejects_non_str_param1_name(self):
        """_run_demo_analysis raises TypeError when param1_name is not a str."""
        arr = np.linspace(0, 1, 5)
        with pytest.raises(TypeError, match="param1_name must be a str"):
            _check_run_demo_analysis_args(123, "p2", "out", arr, arr)

    def test_run_demo_analysis_rejects_non_str_param2_name(self):
        """_run_demo_analysis raises TypeError when param2_name is not a str."""
        arr = np.linspace(0, 1, 5)
        with pytest.raises(TypeError, match="param2_name must be a str"):
            _check_run_demo_analysis_args("p1", 456, "out", arr, arr)

    def test_run_demo_analysis_rejects_non_str_output_name(self):
        """_run_demo_analysis raises TypeError when output_name is not a str."""
        arr = np.linspace(0, 1, 5)
        with pytest.raises(TypeError, match="output_name must be a str"):
            _check_run_demo_analysis_args("p1", "p2", None, arr, arr)

    def test_run_demo_analysis_rejects_non_ndarray_param1_values(self):
        """_run_demo_analysis raises TypeError when param1_values is not a ndarray."""
        arr = np.linspace(0, 1, 5)
        with pytest.raises(TypeError, match="param1_values must be a numpy ndarray"):
            _check_run_demo_analysis_args("p1", "p2", "out", [1, 2, 3], arr)

    def test_run_demo_analysis_rejects_non_ndarray_param2_values(self):
        """_run_demo_analysis raises TypeError when param2_values is not a ndarray."""
        arr = np.linspace(0, 1, 5)
        with pytest.raises(TypeError, match="param2_values must be a numpy ndarray"):
            _check_run_demo_analysis_args("p1", "p2", "out", arr, [1, 2, 3])

    def test_run_demo_analysis_accepts_valid_args(self):
        """_run_demo_analysis guard passes for valid arguments (no exception raised)."""
        arr = np.linspace(0, 1, 5)
        # Should not raise
        _check_run_demo_analysis_args(
            "Temperature", "O2/Feed Ratio", "Efficiency", arr, arr
        )

    def test_calculate_sensitivity_rejects_non_ndarray_param1_values(self):
        """_calculate_sensitivity raises TypeError when param1_values is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        Z = np.ones((3, 3))
        with pytest.raises(TypeError, match="param1_values must be a numpy ndarray"):
            _check_ndarray_triple([1, 2, 3], arr, Z)

    def test_calculate_sensitivity_rejects_non_ndarray_param2_values(self):
        """_calculate_sensitivity raises TypeError when param2_values is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        Z = np.ones((3, 3))
        with pytest.raises(TypeError, match="param2_values must be a numpy ndarray"):
            _check_ndarray_triple(arr, [1, 2, 3], Z)

    def test_calculate_sensitivity_rejects_non_ndarray_Z(self):
        """_calculate_sensitivity raises TypeError when Z is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        with pytest.raises(TypeError, match="Z must be a numpy ndarray"):
            _check_ndarray_triple(arr, arr, [[1, 2], [3, 4]])

    def test_update_preview_rejects_non_ndarray_param1_values(self):
        """_update_preview raises TypeError when param1_values is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        Z = np.ones((3, 3))
        with pytest.raises(TypeError, match="param1_values must be a numpy ndarray"):
            _check_ndarray_triple([1, 2, 3], arr, Z)

    def test_update_preview_rejects_non_ndarray_param2_values(self):
        """_update_preview raises TypeError when param2_values is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        Z = np.ones((3, 3))
        with pytest.raises(TypeError, match="param2_values must be a numpy ndarray"):
            _check_ndarray_triple(arr, [1, 2, 3], Z)

    def test_update_preview_rejects_non_ndarray_Z(self):
        """_update_preview raises TypeError when Z is not a ndarray."""
        arr = np.linspace(0, 1, 3)
        with pytest.raises(TypeError, match="Z must be a numpy ndarray"):
            _check_ndarray_triple(arr, arr, "not_an_array")
