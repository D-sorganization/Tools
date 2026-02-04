"""
Adam Optimizer GUI Tests
========================

TDD tests for the Adam Optimizer GUI components.
Tests cover PyQt6 main window, Adam algorithm, and convergence.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestOptimizerMainWindow:
    """Tests for the PyQt6 Optimizer main window."""

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
            from optimizer_gui.ui.pyqt6 import main_window

            assert hasattr(main_window, "OptimizerWindow")
        except ImportError:
            pytest.skip("PyQt6 main window not yet implemented")

    def test_main_window_class_exists(self, mock_qt_app):
        """Test that window class is defined and callable."""
        try:
            from optimizer_gui.ui.pyqt6.main_window import OptimizerWindow

            assert callable(OptimizerWindow)
        except ImportError:
            pytest.skip("Main window not yet implemented")


class TestAdamAlgorithm:
    """Tests for the Adam optimization algorithm."""

    def test_adam_momentum_update(self):
        """Test Adam first moment (momentum) update."""
        beta1 = 0.9
        m = 0.0
        gradient = 1.0

        # m_new = beta1 * m + (1 - beta1) * gradient
        m_new = beta1 * m + (1 - beta1) * gradient
        assert m_new == pytest.approx(0.1)

        # Second iteration
        m = m_new
        m_new = beta1 * m + (1 - beta1) * gradient
        assert m_new == pytest.approx(0.19)

    def test_adam_rmsprop_update(self):
        """Test Adam second moment (RMSprop) update."""
        beta2 = 0.999
        v = 0.0
        gradient = 1.0

        # v_new = beta2 * v + (1 - beta2) * gradient^2
        v_new = beta2 * v + (1 - beta2) * (gradient**2)
        assert v_new == pytest.approx(0.001)

    def test_adam_bias_correction(self):
        """Test Adam bias correction."""
        beta1 = 0.9
        beta2 = 0.999
        m = 0.1
        v = 0.001
        iteration = 1

        # Bias-corrected estimates
        m_hat = m / (1 - beta1**iteration)
        v_hat = v / (1 - beta2**iteration)

        assert m_hat == pytest.approx(1.0)  # 0.1 / (1 - 0.9)
        assert v_hat == pytest.approx(1.0)  # 0.001 / (1 - 0.999)

    def test_adam_parameter_update(self):
        """Test Adam parameter update step."""
        learning_rate = 0.01
        epsilon = 1e-8
        m_hat = 1.0
        v_hat = 1.0

        # update = lr * m_hat / (sqrt(v_hat) + epsilon)
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        assert update == pytest.approx(0.01, rel=1e-6)


class TestOptimizationConvergence:
    """Tests for optimization convergence criteria."""

    def test_gradient_norm_convergence(self):
        """Test gradient norm convergence criterion."""
        gradient = np.array([0.001, 0.001, 0.001])
        tolerance = 0.01

        gradient_norm = np.linalg.norm(gradient)
        assert gradient_norm < tolerance

    def test_parameter_change_convergence(self):
        """Test parameter change convergence criterion."""
        current = np.array([1.0, 2.0, 3.0])
        previous = np.array([1.0001, 2.0001, 3.0001])
        tolerance = 0.001

        change = np.linalg.norm(current - previous)
        assert change < tolerance

    def test_bounds_clipping(self):
        """Test parameter bounds clipping."""
        values = np.array([1.5, -0.5, 2.5])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        clipped = np.clip(values, lower, upper)
        np.testing.assert_array_equal(clipped, [1.0, 0.0, 1.0])


class TestDemoFunctions:
    """Tests for demo optimization functions."""

    def test_rosenbrock_minimum(self):
        """Test Rosenbrock function has minimum at (1, 1)."""
        # f(x, y) = (1-x)^2 + 100*(y-x^2)^2
        x, y = 1.0, 1.0
        f = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        assert f == pytest.approx(0.0)

    def test_rosenbrock_gradient_at_minimum(self):
        """Test Rosenbrock gradient is zero at minimum."""
        x, y = 1.0, 1.0
        step = 1e-6

        # Numerical gradient
        f = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        f_x_plus = (1 - (x + step)) ** 2 + 100 * (y - (x + step) ** 2) ** 2
        f_y_plus = (1 - x) ** 2 + 100 * ((y + step) - x**2) ** 2

        grad_x = (f_x_plus - f) / step
        grad_y = (f_y_plus - f) / step

        assert abs(grad_x) < 1e-3  # Numerical approximation tolerance
        assert abs(grad_y) < 1e-3


class TestOptimizerGUIRegistration:
    """Tests for GUI framework registration."""

    def test_gui_registration_exists(self):
        """Test that gui_registration.py exists and has required metadata."""
        try:
            from optimizer_gui import gui_registration

            assert hasattr(gui_registration, "GUI_METADATA")
            metadata = gui_registration.GUI_METADATA

            assert "name" in metadata
            assert "description" in metadata
            assert "category" in metadata
            assert "entry_point" in metadata
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_gui_registration_category(self):
        """Test that optimizer is in optimization category."""
        try:
            from optimizer_gui import gui_registration

            assert gui_registration.GUI_METADATA["category"] == "optimization"
        except ImportError:
            pytest.skip("GUI registration not yet implemented")

    def test_launcher_exists(self):
        """Test that launcher script exists."""
        try:
            from optimizer_gui import launch_pyqt6

            assert hasattr(launch_pyqt6, "main")
        except ImportError:
            pytest.skip("Launcher not yet implemented")
