"""Tests for signal_toolkit.calculus module.

Covers:
- DifferentiationMethod and IntegrationMethod enums
- Differentiator: differentiate, compute_at_point, various methods
- Integrator: integrate (definite), cumulative_integral
- TangentLine and IntegralResult dataclasses
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np
from signal_toolkit.calculus import (
    DifferentiationMethod,
    Differentiator,
    IntegrationMethod,
    Integrator,
    compute_all_tangent_lines,
    compute_arc_length,
    compute_curvature,
    compute_derivative,
    compute_tangent_line,
    find_extrema,
    find_inflection_points,
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

    @pytest.mark.scientific
    def test_central_derivative_of_sine_matches_cosine_reference(self) -> None:
        """Central-difference derivative stays anchored to d/dt sin(t)=cos(t)."""
        t = np.linspace(0.0, 2 * np.pi, 101)
        signal = Signal(time=t, values=np.sin(t), name="sin", units="")
        deriv = Differentiator(method=DifferentiationMethod.CENTRAL).differentiate(
            signal
        )

        interior_err = np.max(np.abs(deriv.values[2:-2] - np.cos(t)[2:-2]))
        assert interior_err < 1e-2

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

    def test_simpson_x_squared_includes_upper_bound(self) -> None:
        """∫₀³ x² dx = 9 — the upper-bound sample must not be dropped (#3383).

        Previously ``searchsorted(..., side='left')`` excluded the sample at the
        upper bound, leaving Simpson's rule short by the final interval and
        returning 8.13 instead of the exact 9.
        """
        x = np.linspace(0.0, 3.0, 31)  # includes the endpoint x = 3
        signal = Signal(time=x, values=x**2, name="x^2", units="")
        integ = Integrator(method=IntegrationMethod.SIMPSON)
        result = integ.integrate(signal)
        # Simpson on an even-spaced grid through the endpoint is exact for x^2.
        assert result.value == pytest.approx(9.0, abs=1e-6)

    @pytest.mark.scientific
    def test_trapezoid_linear_integral_is_exact_reference(self) -> None:
        """Trapezoid integration is exact on a linear reference signal (#3391)."""
        x = np.linspace(0.0, 10.0, 51)
        signal = Signal(time=x, values=2 * x + 1, name="linear", units="")
        result = Integrator(method=IntegrationMethod.TRAPEZOID).integrate(signal)

        assert result.value == pytest.approx(110.0, rel=1e-9)

    def test_full_integral_not_short_by_one_interval(self) -> None:
        """Default (full-range) integral spans the entire signal (#3383)."""
        x = np.linspace(0.0, 3.0, 31)
        signal = Signal(time=x, values=x**2, name="x^2", units="")
        integ = Integrator(method=IntegrationMethod.TRAPEZOID)
        result = integ.integrate(signal)
        # Trapezoid over the full span is close to 9 (O(h^2) error), and in
        # particular must exceed the old, one-interval-short value of ~8.13.
        assert result.value == pytest.approx(9.0, rel=2e-3)
        assert result.value > 8.5


# ── Explicit guard regressions (#3344) ──────────────────────────────────


class TestDifferentiatorInitPreconditions:
    """Differentiator() raises ValueError when method is None."""

    def test_none_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="method must be provided"):
            Differentiator(method=None)  # type: ignore[arg-type]


class TestDifferentiatePreconditions:
    """Differentiator.differentiate raises ValueError when signal is None."""

    def test_none_signal_raises_value_error(self) -> None:
        differentiator = Differentiator()
        with pytest.raises(ValueError, match="signal must be provided"):
            differentiator.differentiate(None)  # type: ignore[arg-type]


class TestComputeAtPointPreconditions:
    """Differentiator.compute_at_point raises ValueError when signal is None."""

    def test_none_signal_raises_value_error(self) -> None:
        differentiator = Differentiator()
        with pytest.raises(ValueError, match="signal must be provided"):
            differentiator.compute_at_point(None, t_point=1.0)  # type: ignore[arg-type]


class TestIntegratorInitPreconditions:
    """Integrator() raises ValueError when method is None."""

    def test_none_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="method must be provided"):
            Integrator(method=None)  # type: ignore[arg-type]


class TestCumulativeIntegralPreconditions:
    """Integrator.cumulative_integral raises ValueError when signal is None."""

    def test_none_signal_raises_value_error(self) -> None:
        integrator = Integrator()
        with pytest.raises(ValueError, match="signal must be provided"):
            integrator.cumulative_integral(None)  # type: ignore[arg-type]


class TestModuleFunctionPreconditions:
    """Module-level calculus helpers raise ValueError when signal is None."""

    def test_compute_derivative_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_derivative(None)  # type: ignore[arg-type]

    def test_compute_tangent_line_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_tangent_line(None, t_point=1.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("t_point", [-0.1, 10.1])
    def test_compute_tangent_line_rejects_out_of_range_t_point(
        self, linear_signal: Signal, t_point: float
    ) -> None:
        with pytest.raises(ValueError, match="t_point must be within"):
            compute_tangent_line(linear_signal, t_point=t_point)

    def test_compute_all_tangent_lines_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_all_tangent_lines(None)  # type: ignore[arg-type]

    def test_compute_curvature_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_curvature(None)  # type: ignore[arg-type]

    def test_compute_arc_length_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            compute_arc_length(None)  # type: ignore[arg-type]

    def test_find_extrema_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            find_extrema(None)  # type: ignore[arg-type]

    def test_find_inflection_points_none_raises(self) -> None:
        with pytest.raises(ValueError, match="signal must be provided"):
            find_inflection_points(None)  # type: ignore[arg-type]


class TestGuardsSurviveOptimizedMode:
    """Guards are if/raise, not bare asserts, and survive python -O."""

    def test_compute_derivative_raises_under_minus_o(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        env = os.environ.copy()
        src = str(repo_root / "src")
        python_src = str(repo_root / "src" / "python" / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [src, python_src, existing_pythonpath]
            if existing_pythonpath
            else [src, python_src]
        )
        code = (
            "from signal_toolkit.calculus import compute_derivative; "
            "compute_derivative(None)"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            capture_output=True,
            env=env,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "ValueError" in result.stderr
        assert "AttributeError" not in result.stderr
