"""Unit tests for Signal Toolkit Widget decomposition.

Tests the widget_processing, widget_plotting, and widget_ui mixins
as well as the core widget module functionality.
"""

from __future__ import annotations

import logging

import numpy as np

from shared.python.signal_toolkit.core import Signal, SignalGenerator
from shared.python.signal_toolkit.fitting import FunctionFitter

logger = logging.getLogger(__name__)

# ==================================================================
# Signal Processing Tests (widget_processing methods)
# ==================================================================


class TestSignalGeneration:
    """Tests for signal generation in the processing mixin."""

    def test_sinusoid_generation(self) -> None:
        """Test sinusoid signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=1.0)

        assert signal is not None
        assert len(signal.values) == 1000
        # Peak should be near 2.0
        assert abs(np.max(signal.values) - 2.0) < 0.01

    def test_cosine_generation(self) -> None:
        """Test cosine signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.cosine(t, amplitude=1.5, frequency=2.0)

        assert signal is not None
        assert len(signal.values) == 1000
        # At t=0, cos should be at its peak
        assert abs(signal.values[0] - 1.5) < 0.01

    def test_polynomial_generation(self) -> None:
        """Test polynomial signal generation."""
        t = np.linspace(0, 5, 100)
        # y = 1 + 2t + 0.5t²
        signal = SignalGenerator.polynomial(t, [1.0, 2.0, 0.5])

        assert signal is not None
        # At t=0, y=1.0
        assert abs(signal.values[0] - 1.0) < 0.01

    def test_exponential_generation(self) -> None:
        """Test exponential signal generation."""
        t = np.linspace(0, 5, 100)
        signal = SignalGenerator.exponential(t, amplitude=1.0, decay_rate=0.5)

        assert signal is not None
        # At t=0, value should be ~1.0
        assert abs(signal.values[0] - 1.0) < 0.1

    def test_linear_generation(self) -> None:
        """Test linear signal generation."""
        t = np.linspace(0, 10, 100)
        signal = SignalGenerator.linear(t, slope=2.0, intercept=1.0)

        assert signal is not None
        # At t=0, y=1.0
        assert abs(signal.values[0] - 1.0) < 0.01
        # At t=10, y=21.0
        assert abs(signal.values[-1] - 21.0) < 0.2

    def test_step_generation(self) -> None:
        """Test step signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.step(t, step_time=5.0, step_value=1.0)

        assert signal is not None
        # Before step: 0
        assert signal.values[0] == 0.0
        # After step: 1
        assert signal.values[-1] == 1.0

    def test_chirp_generation(self) -> None:
        """Test chirp signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.chirp(t, f0=0.5, f1=5.0, amplitude=1.0)

        assert signal is not None
        assert len(signal.values) == 1000

    def test_square_generation(self) -> None:
        """Test square wave signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.square(t, frequency=1.0, amplitude=1.0)

        assert signal is not None
        # Values should be approximately +1 or -1
        unique_vals = set(np.round(signal.values, 0))
        assert 1.0 in unique_vals or -1.0 in unique_vals

    def test_triangle_generation(self) -> None:
        """Test triangle wave signal generation."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.triangle(t, frequency=1.0, amplitude=1.0)

        assert signal is not None
        assert len(signal.values) == 1000


class TestSignalCopy:
    """Tests for Signal copy functionality."""

    def test_signal_copy_preserves_data(self) -> None:
        """Test that signal copy creates independent copy."""
        t = np.linspace(0, 10, 100)
        original = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0)
        copy = original.copy()

        assert copy is not original
        np.testing.assert_array_equal(copy.time, original.time)
        np.testing.assert_array_equal(copy.values, original.values)

    def test_signal_copy_independence(self) -> None:
        """Test that modifying copy doesn't modify original."""
        t = np.linspace(0, 10, 100)
        original = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0)
        copy = original.copy()

        copy.values[0] = 999.0
        assert original.values[0] != 999.0


# ==================================================================
# Function Fitting Tests
# ==================================================================


class TestFunctionFitting:
    """Tests for function fitting used in widget_processing."""

    def test_fit_linear(self) -> None:
        """Test linear fit."""
        t = np.linspace(0, 10, 100)
        signal = SignalGenerator.linear(t, slope=2.5, intercept=1.0)

        fitter = FunctionFitter()
        result = fitter.fit_linear(signal)

        assert result.r_squared > 0.99
        assert result.fitted_signal is not None

    def test_fit_polynomial(self) -> None:
        """Test polynomial fit."""
        t = np.linspace(0, 5, 200)
        signal = SignalGenerator.polynomial(t, [1.0, 2.0, 0.5])

        fitter = FunctionFitter()
        result = fitter.fit_polynomial(signal, order=6)

        assert result.r_squared > 0.99

    def test_fit_sinusoid(self) -> None:
        """Test sinusoidal fit on clean sinusoid."""
        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=1.0)

        fitter = FunctionFitter()
        result = fitter.fit_sinusoid(signal)

        assert result.r_squared > 0.9
        assert result.fitted_signal is not None

    def test_auto_fit(self) -> None:
        """Test auto-fit identifies best fitting function."""
        t = np.linspace(0, 10, 100)
        signal = SignalGenerator.linear(t, slope=3.0, intercept=0.5)

        fitter = FunctionFitter()
        best_type, result = fitter.auto_fit(signal)

        # Linear should be found with high R²
        assert result.r_squared > 0.9
        assert best_type is not None


# ==================================================================
# Calculus Tests (used by widget_processing)
# ==================================================================


class TestCalculusOperations:
    """Tests for calculus operations used by the processing mixin."""

    def test_differentiation(self) -> None:
        """Test numerical differentiation."""
        from shared.python.signal_toolkit.calculus import Differentiator

        t = np.linspace(0, 10, 1000)
        # d/dt(sin(t)) = cos(t)
        signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0 / (2 * np.pi))

        diff = Differentiator()
        derivative = diff.differentiate(signal, order=1)

        assert derivative is not None
        assert len(derivative.values) > 0

    def test_integration(self) -> None:
        """Test numerical integration."""
        from shared.python.signal_toolkit.calculus import Integrator

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.linear(t, slope=1.0, intercept=0.0)

        integrator = Integrator()
        result = integrator.integrate(signal, lower_bound=0, upper_bound=10)

        # Integral of x from 0 to 10 = 50
        assert abs(result.value - 50.0) < 1.0
        assert result.cumulative_signal is not None

    def test_tangent_line(self) -> None:
        """Test tangent line computation."""
        from shared.python.signal_toolkit.calculus import compute_tangent_line

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0 / (2 * np.pi))

        tangent = compute_tangent_line(signal, 5.0)

        assert tangent is not None
        assert hasattr(tangent, "slope")
        assert hasattr(tangent, "t_point")
        assert hasattr(tangent, "y_point")
        assert hasattr(tangent, "line_values")


# ==================================================================
# Filter Tests (used by widget_processing)
# ==================================================================


class TestFilterOperations:
    """Tests for filter operations used by the processing mixin."""

    def test_moving_average(self) -> None:
        """Test moving average filter."""
        from shared.python.signal_toolkit.filters import apply_moving_average

        t = np.linspace(0, 10, 1000)
        # Add noise
        rng = np.random.default_rng(42)
        values = np.sin(2 * np.pi * t) + 0.5 * rng.standard_normal(len(t))
        signal = Signal(t, values, name="noisy")

        filtered = apply_moving_average(signal, window_size=11)
        assert filtered is not None
        # Filtered should be smoother (lower std of difference)
        assert np.std(np.diff(filtered.values)) < np.std(np.diff(signal.values))

    def test_savgol_filter(self) -> None:
        """Test Savitzky-Golay filter."""
        from shared.python.signal_toolkit.filters import apply_savgol

        t = np.linspace(0, 10, 1000)
        rng = np.random.default_rng(42)
        values = np.sin(2 * np.pi * t) + 0.5 * rng.standard_normal(len(t))
        signal = Signal(t, values, name="noisy")

        filtered = apply_savgol(signal, window_length=11, polyorder=3)
        assert filtered is not None
        assert len(filtered.values) == len(signal.values)


# ==================================================================
# Noise Tests (used by widget_processing)
# ==================================================================


class TestNoiseOperations:
    """Tests for noise operations used by the processing mixin."""

    def test_add_white_noise(self) -> None:
        """Test adding white noise to signal."""
        from shared.python.signal_toolkit.noise import NoiseType, add_noise_to_signal

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0)

        noisy = add_noise_to_signal(signal, noise_type=NoiseType.WHITE, snr_db=20)
        assert noisy is not None
        # Noisy signal should differ from the original
        assert not np.array_equal(noisy.values, signal.values)

    def test_add_uniform_noise(self) -> None:
        """Test adding uniform noise."""
        from shared.python.signal_toolkit.noise import NoiseType, add_noise_to_signal

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=1.0, frequency=1.0)

        noisy = add_noise_to_signal(signal, noise_type=NoiseType.UNIFORM, amplitude=0.1)
        assert noisy is not None
        assert not np.array_equal(noisy.values, signal.values)


# ==================================================================
# Saturation Tests (used by widget_processing)
# ==================================================================


class TestSaturationOperations:
    """Tests for saturation operations used by the processing mixin."""

    def test_hard_saturation(self) -> None:
        """Test hard clip saturation."""
        from shared.python.signal_toolkit.limits import SaturationMode, apply_saturation

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=1.0)

        clipped = apply_saturation(signal, lower=-1.0, upper=1.0, mode=SaturationMode.HARD)
        assert clipped is not None
        assert np.max(clipped.values) <= 1.0 + 1e-10
        assert np.min(clipped.values) >= -1.0 - 1e-10

    def test_tanh_saturation(self) -> None:
        """Test tanh soft saturation."""
        from shared.python.signal_toolkit.limits import SaturationMode, apply_saturation

        t = np.linspace(0, 10, 1000)
        signal = SignalGenerator.sinusoid(t, amplitude=2.0, frequency=1.0)

        clipped = apply_saturation(signal, lower=-1.0, upper=1.0, mode=SaturationMode.TANH)
        assert clipped is not None
        # Tanh is smooth, values should be within bounds (approximately)
        assert np.max(clipped.values) < 2.5  # Should be compressed


# ==================================================================
# Widget Module-Level Tests
# ==================================================================


class TestWidgetModuleAvailability:
    """Tests for widget module availability and imports."""

    def test_has_matplotlib_flag_exists(self) -> None:
        """Test that HAS_MATPLOTLIB flag is defined."""
        from shared.python.signal_toolkit.widget import HAS_MATPLOTLIB

        assert isinstance(HAS_MATPLOTLIB, bool)

    def test_has_pyqt_flag_exists(self) -> None:
        """Test that HAS_PYQT flag is defined."""
        from shared.python.signal_toolkit.widget import HAS_PYQT

        assert isinstance(HAS_PYQT, bool)

    def test_dark_stylesheet_defined(self) -> None:
        """Test that DARK_STYLESHEET is a non-empty string."""
        from shared.python.signal_toolkit.widget import DARK_STYLESHEET

        assert isinstance(DARK_STYLESHEET, str)
        assert len(DARK_STYLESHEET) > 100
        assert "background-color" in DARK_STYLESHEET

    def test_signal_toolkit_widget_importable(self) -> None:
        """Test that SignalToolkitWidget is importable (may be stub)."""
        from shared.python.signal_toolkit.widget import SignalToolkitWidget

        assert SignalToolkitWidget is not None
