"""Tests for the signal_toolkit.fitting module.

Covers:
- FitResult dataclass
- SinusoidFitter: sinusoidal curve fitting
- ExponentialFitter: exponential decay/growth fitting
- R² and RMSE statistics
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np
from numpy.testing import assert_allclose
from signal_toolkit.core import Signal
from signal_toolkit.fitting import (
    CustomFunctionFitter,
    ExponentialFitter,
    FitResult,
    FunctionFitter,
    LinearFitter,
    PolynomialFitter,
    SinusoidFitter,
)

# ── Helper ───────────────────────────────────────────────────────────────


def _make_signal(t: np.ndarray, y: np.ndarray) -> Signal:
    """Create a Signal from time and value arrays."""
    return Signal(time=t, values=y)


# ── FitResult ────────────────────────────────────────────────────────────


class TestFitResult:
    """Test FitResult dataclass."""

    def test_fit_result_r_squared_range(self) -> None:
        """R² should be between 0 and 1 for a good fit."""
        t = np.linspace(0, 1, 10)
        result = FitResult(
            parameters={"a": 1.0},
            covariance=None,
            r_squared=0.95,
            rmse=0.1,
            fitted_signal=Signal(time=t, values=np.ones(10)),
            residuals=np.array([0.1]),
        )
        assert 0.0 <= result.r_squared <= 1.0

    def test_fit_result_success_default(self) -> None:
        t = np.linspace(0, 1, 10)
        result = FitResult(
            parameters={"a": 1.0},
            covariance=None,
            r_squared=0.99,
            rmse=0.01,
            fitted_signal=Signal(time=t, values=np.ones(10)),
            residuals=np.array([0.01]),
        )
        assert result.success is True
        assert result.message == ""


# ── SinusoidFitter ───────────────────────────────────────────────────────


class TestSinusoidFitter:
    """Test sinusoidal function fitting."""

    def test_perfect_sinusoid_recovery(self) -> None:
        """Fitting a perfect sinusoid should recover the parameters."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        amplitude = 2.0
        frequency = 10.0
        phase = np.pi / 4
        offset = 0.5
        y = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

        sig = _make_signal(t, y)
        fitter = SinusoidFitter()
        result = fitter.fit(sig)

        assert result.success
        assert result.r_squared > 0.99
        assert_allclose(abs(result.parameters["amplitude"]), amplitude, rtol=0.1)
        assert_allclose(result.parameters["frequency"], frequency, rtol=0.05)

    def test_sinusoid_rmse_low(self) -> None:
        """RMSE should be very low for a perfect fit."""
        fs = 500.0
        t = np.arange(0, 2.0, 1.0 / fs)
        y = 3.0 * np.sin(2 * np.pi * 5.0 * t)
        sig = _make_signal(t, y)

        fitter = SinusoidFitter()
        result = fitter.fit(sig)
        assert result.rmse < 0.5

    def test_sinusoid_with_initial_guess(self) -> None:
        """Providing an initial guess should still work."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        y = 1.5 * np.sin(2 * np.pi * 20.0 * t + 0.3) + 0.2
        sig = _make_signal(t, y)

        fitter = SinusoidFitter()
        result = fitter.fit(sig, initial_guess=(1.5, 20.0, 0.3, 0.2))
        assert result.success
        assert result.r_squared > 0.95

    def test_estimate_initial_params(self) -> None:
        """FFT-based initial parameter estimation should be reasonable."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0 / fs)
        freq = 25.0
        y = np.sin(2 * np.pi * freq * t)

        amp, est_freq, _phase, _offset = SinusoidFitter.estimate_initial_params(t, y)
        # Frequency estimate should be close
        assert_allclose(est_freq, freq, rtol=0.1)
        assert amp > 0

    def test_function_string(self) -> None:
        fitter = SinusoidFitter()
        s = fitter.get_function_string(
            {"amplitude": 1.0, "frequency": 10.0, "phase": 0.0, "offset": 0.0}
        )
        assert "sin" in s.lower() or "amplitude" in s.lower()


# ── ExponentialFitter ────────────────────────────────────────────────────


class TestExponentialFitter:
    """Test exponential curve fitting."""

    def test_decay_fit(self) -> None:
        """Fitting exponential decay should recover parameters."""
        fs = 100.0
        t = np.arange(0, 5.0, 1.0 / fs)
        amplitude = 3.0
        decay_rate = 1.5
        offset = 0.5
        y = amplitude * np.exp(-decay_rate * t) + offset

        sig = _make_signal(t, y)
        fitter = ExponentialFitter()
        result = fitter.fit_decay(sig)

        assert result.success
        assert result.r_squared > 0.99
        assert_allclose(result.parameters["amplitude"], amplitude, rtol=0.1)
        assert_allclose(result.parameters["decay_rate"], decay_rate, rtol=0.1)

    def test_growth_fit(self) -> None:
        """Fitting exponential growth should recover parameters."""
        fs = 100.0
        t = np.arange(0, 5.0, 1.0 / fs)
        amplitude = 4.0
        growth_rate = 0.8
        offset = 1.0
        y = amplitude * (1.0 - np.exp(-growth_rate * t)) + offset

        sig = _make_signal(t, y)
        fitter = ExponentialFitter()
        result = fitter.fit_growth(sig)

        assert result.success
        assert result.r_squared > 0.95

    def test_decay_fit_residuals(self) -> None:
        """Residuals should be small for a perfect exponential decay."""
        fs = 100.0
        t = np.arange(0, 3.0, 1.0 / fs)
        y = 2.0 * np.exp(-1.0 * t) + 0.1

        sig = _make_signal(t, y)
        fitter = ExponentialFitter()
        result = fitter.fit_decay(sig)

        max_residual = np.max(np.abs(result.residuals))
        assert max_residual < 0.1


# ── LinearFitter ─────────────────────────────────────────────────────────


class TestLinearFitter:
    """Test linear function fitting."""

    def test_linear_fit(self) -> None:
        t = np.linspace(0, 10, 100)
        y = 2.5 * t + 1.0
        sig = _make_signal(t, y)

        fitter = LinearFitter()
        result = fitter.fit(sig)

        assert result.success
        assert result.r_squared == pytest.approx(1.0)
        assert result.parameters["slope"] == pytest.approx(2.5)
        assert result.parameters["intercept"] == pytest.approx(1.0)


# ── PolynomialFitter ─────────────────────────────────────────────────────


class TestPolynomialFitter:
    """Test polynomial function fitting."""

    def test_quadratic_fit(self) -> None:
        t = np.linspace(0, 4, 50)
        y = 3.0 * t**2 - 2.0 * t + 1.5
        sig = _make_signal(t, y)

        fitter = PolynomialFitter(order=2)
        result = fitter.fit(sig)

        assert result.success
        assert result.r_squared == pytest.approx(1.0)
        assert result.parameters["c2"] == pytest.approx(3.0)
        assert result.parameters["c1"] == pytest.approx(-2.0)
        assert result.parameters["c0"] == pytest.approx(1.5)


# ── CustomFunctionFitter ─────────────────────────────────────────────────


class TestCustomFunctionFitter:
    """Test custom function fitting."""

    def test_custom_fit(self) -> None:
        def model(t: np.ndarray, a: float, b: float) -> np.ndarray:
            return a * np.sin(b * t)

        t = np.linspace(0, np.pi, 100)
        y = 2.0 * np.sin(3.0 * t)
        sig = _make_signal(t, y)

        fitter = CustomFunctionFitter(model, ["a", "b"])
        result = fitter.fit(sig, initial_guess=[1.8, 2.8])

        assert result.success
        assert result.r_squared == pytest.approx(1.0)
        assert result.parameters["a"] == pytest.approx(2.0, rel=1e-3)
        assert result.parameters["b"] == pytest.approx(3.0, rel=1e-3)


# ── FunctionFitter ───────────────────────────────────────────────────────


class TestFunctionFitter:
    """Test unified FunctionFitter interface."""

    def test_auto_fit_linear(self) -> None:
        t = np.linspace(0, 5, 50)
        y = 4.0 * t + 2.0
        sig = _make_signal(t, y)

        fitter = FunctionFitter()
        best_type, result = fitter.auto_fit(sig, candidates=["linear", "polynomial"])

        assert best_type in ("linear", "polynomial")
        assert result.r_squared == pytest.approx(1.0)
