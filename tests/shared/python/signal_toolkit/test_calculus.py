"""Tests for signal_toolkit.calculus module.

Covers:
- DifferentiationMethod and IntegrationMethod enums
- Differentiator: differentiate, compute_at_point, various methods
- Integrator: integrate (definite), cumulative_integral
- TangentLine and IntegralResult dataclasses
"""

from __future__ import annotations

import numpy as np
import pytest
from signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
)
from signal_toolkit.core import Signal


@pytest.fixture()
def sine_signal() -> Signal:
    """Create a simple sine wave signal for testing."""
    t = np.linspace(0, 2 * np.pi, 1000)
    values = np.sin(t)
    return Signal(time=t, values=values, name="sine")


@pytest.fixture()
def linear_signal() -> Signal:
    """Create y = 2*t + 1."""
    t = np.linspace(0, 10, 500)
    values = 2.0 * t + 1.0
    return Signal(time=t, values=values, name="linear")


@pytest.fixture()
def constant_signal() -> Signal:
    """Create constant signal y = 5."""
    t = np.linspace(0, 1, 100)
    values = np.full_like(t, 5.0)
    return Signal(time=t, values=values, name="constant")


# ── Enums ───────────────────────────────────────────────────────────────


class TestEnums:
    def test_diff_methods(self) -> None:
        assert DifferentiationMethod.CENTRAL.value == "central"
        assert DifferentiationMethod.SAVGOL.value == "savgol"

    def test_int_methods(self) -> None:
        assert IntegrationMethod.TRAPEZOID.value == "trapezoid"
        assert IntegrationMethod.SIMPSON.value == "simpson"


# ── Differentiator ──────────────────────────────────────────────────────


class TestDifferentiator:
    def test_constant_derivative_near_zero(self, constant_signal: Signal) -> None:
        diff = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = diff.differentiate(constant_signal)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-6)

    def test_linear_derivative_constant(self, linear_signal: Signal) -> None:
        diff = Differentiator(method=DifferentiationMethod.GRADIENT)
        result = diff.differentiate(linear_signal)
        # dy/dt of 2t+1 = 2
        np.testing.assert_allclose(result.values, 2.0, atol=0.1)

    def test_sine_derivative_is_cosine(self, sine_signal: Signal) -> None:
        diff = Differentiator(method=DifferentiationMethod.SAVGOL)
        result = diff.differentiate(sine_signal)
        expected = np.cos(sine_signal.time)
        # Interior points should match (edges may be less accurate)
        np.testing.assert_allclose(result.values[10:-10], expected[10:-10], atol=0.05)

    def test_second_derivative(self, sine_signal: Signal) -> None:
        """d²/dt² sin(t) ≈ -sin(t)."""
        diff = Differentiator(method=DifferentiationMethod.SAVGOL)
        result = diff.differentiate(sine_signal, order=2)
        expected = -np.sin(sine_signal.time)
        np.testing.assert_allclose(result.values[20:-20], expected[20:-20], atol=0.15)

    def test_compute_at_point(self, linear_signal: Signal) -> None:
        diff = Differentiator(method=DifferentiationMethod.GRADIENT)
        value = diff.compute_at_point(linear_signal, t_point=5.0)
        assert value == pytest.approx(2.0, abs=0.2)

    def test_central_method(self, linear_signal: Signal) -> None:
        diff = Differentiator(method=DifferentiationMethod.CENTRAL)
        result = diff.differentiate(linear_signal)
        np.testing.assert_allclose(result.values[1:-1], 2.0, atol=0.1)

    def test_output_is_signal(self, sine_signal: Signal) -> None:
        diff = Differentiator()
        result = diff.differentiate(sine_signal)
        assert isinstance(result, Signal)
        assert len(result.time) == len(sine_signal.time)


# ── Integrator ──────────────────────────────────────────────────────────


class TestIntegrator:
    def test_constant_integral(self, constant_signal: Signal) -> None:
        """∫₀¹ 5 dt = 5."""
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.integrate(constant_signal)
        assert result.value == pytest.approx(5.0, rel=0.02)

    def test_linear_integral(self, linear_signal: Signal) -> None:
        """∫₀¹⁰ (2t+1) dt = t²+t |₀¹⁰ = 110."""
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.integrate(linear_signal)
        assert result.value == pytest.approx(110.0, rel=0.01)

    def test_sine_full_period_zero(self, sine_signal: Signal) -> None:
        """∫₀²π sin(t) dt ≈ 0."""
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.integrate(sine_signal)
        assert result.value == pytest.approx(0.0, abs=0.05)

    def test_simpson_method(self, constant_signal: Signal) -> None:
        integ = Integrator(method=IntegrationMethod.SIMPSON)
        result = integ.integrate(constant_signal)
        assert result.value == pytest.approx(5.0, rel=0.02)

    def test_cumulative_integral_monotonic(self, constant_signal: Signal) -> None:
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.cumulative_integral(constant_signal)
        assert isinstance(result, Signal)
        # Cumulative integral of positive constant should be monotonically increasing
        assert np.all(np.diff(result.values) >= -1e-10)

    def test_integral_bounds(self, linear_signal: Signal) -> None:
        """Test integration with custom bounds."""
        integ = Integrator()
        result = integ.integrate(linear_signal, lower_bound=0, upper_bound=5)
        # ∫₀⁵ (2t+1) dt = t²+t |₀⁵ = 30
        assert result.value == pytest.approx(30.0, rel=0.05)
        assert result.lower_bound == pytest.approx(0.0, abs=0.1)
        assert result.upper_bound == pytest.approx(5.0, abs=0.1)
