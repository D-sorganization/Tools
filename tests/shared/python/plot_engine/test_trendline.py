"""Tests for the plot_engine.trendline module.

Covers:
- Linear trendline fitting with R^2 verification
- Polynomial trendline fitting
- Exponential trendline fitting
- Power trendline fitting
- NaN handling
- Edge cases (insufficient data, non-positive values for exp/power)
- TrendlineResult structure
- Equation string format
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from plot_engine.trendline import TrendlineResult, compute_trendline

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_linear_data(
    m: float = 2.0,
    b: float = 1.0,
    n: int = 100,
    noise: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = mx + b with optional noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    y = m * x + b
    if noise > 0:
        y += rng.normal(0, noise, n)
    return x, y


def _make_exponential_data(
    a: float = 2.0,
    b: float = 0.3,
    n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = a * exp(b * x)."""
    x = np.linspace(0, 5, n)
    y = a * np.exp(b * x)
    return x, y


def _make_power_data(
    a: float = 3.0,
    bp: float = 2.0,
    n: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = a * x^b."""
    x = np.linspace(0.1, 5, n)
    y = a * x**bp
    return x, y


# ── Linear Trendline ────────────────────────────────────────────────────


class TestLinearTrendline:
    """Test linear trendline fitting."""

    def test_perfect_linear_r_squared_1(self) -> None:
        x, y = _make_linear_data(m=2.0, b=1.0, noise=0.0)
        result = compute_trendline(x, y, trend_type="linear")
        assert_allclose(result.r_squared, 1.0, atol=1e-10)

    def test_recovers_slope_and_intercept(self) -> None:
        x, y = _make_linear_data(m=3.5, b=-2.0, noise=0.0)
        result = compute_trendline(x, y, trend_type="linear")
        assert_allclose(result.coefficients[0], 3.5, atol=1e-8)
        assert_allclose(result.coefficients[1], -2.0, atol=1e-8)

    def test_noisy_data_r_squared_below_1(self) -> None:
        x, y = _make_linear_data(m=2.0, b=1.0, noise=2.0)
        result = compute_trendline(x, y, trend_type="linear")
        assert 0 < result.r_squared < 1.0

    def test_result_type(self) -> None:
        x, y = _make_linear_data()
        result = compute_trendline(x, y, trend_type="linear")
        assert isinstance(result, TrendlineResult)
        assert result.trend_type == "linear"

    def test_equation_format(self) -> None:
        x, y = _make_linear_data(m=2.0, b=1.0)
        result = compute_trendline(x, y, trend_type="linear")
        assert result.equation.startswith("y = ")
        assert "x" in result.equation

    def test_prediction_arrays(self) -> None:
        x, y = _make_linear_data()
        result = compute_trendline(x, y, trend_type="linear", n_points=50)
        assert len(result.x_pred) == 50
        assert len(result.y_pred) == 50

    def test_negative_intercept_sign(self) -> None:
        x, y = _make_linear_data(m=1.0, b=-5.0)
        result = compute_trendline(x, y, trend_type="linear")
        assert "-" in result.equation


# ── Polynomial Trendline ────────────────────────────────────────────────


class TestPolynomialTrendline:
    """Test polynomial trendline fitting."""

    def test_quadratic_fit(self) -> None:
        x = np.linspace(-5, 5, 100)
        y = 2.0 * x**2 - 3.0 * x + 1.0
        result = compute_trendline(x, y, trend_type="polynomial", degree=2)
        assert_allclose(result.r_squared, 1.0, atol=1e-8)

    def test_recovers_quadratic_coefficients(self) -> None:
        x = np.linspace(0, 10, 200)
        y = 1.5 * x**2 + 2.0 * x + 0.5
        result = compute_trendline(x, y, trend_type="polynomial", degree=2)
        assert_allclose(result.coefficients[0], 1.5, atol=1e-4)
        assert_allclose(result.coefficients[1], 2.0, atol=1e-3)
        assert_allclose(result.coefficients[2], 0.5, atol=1e-2)

    def test_cubic_fit(self) -> None:
        x = np.linspace(-3, 3, 100)
        y = x**3 - 2 * x**2 + x - 1
        result = compute_trendline(x, y, trend_type="polynomial", degree=3)
        assert_allclose(result.r_squared, 1.0, atol=1e-7)

    def test_degree_capped_by_data_points(self) -> None:
        """With only 3 points, a degree-10 request should be capped."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 9.0])
        result = compute_trendline(x, y, trend_type="polynomial", degree=10)
        # Should fit perfectly with the available points
        assert result.r_squared > 0.99

    def test_equation_contains_x_power(self) -> None:
        x = np.linspace(0, 5, 50)
        y = x**2
        result = compute_trendline(x, y, trend_type="polynomial", degree=2)
        assert "x^2" in result.equation


# ── Exponential Trendline ───────────────────────────────────────────────


class TestExponentialTrendline:
    """Test exponential trendline fitting."""

    def test_perfect_exponential(self) -> None:
        x, y = _make_exponential_data(a=2.0, b=0.3)
        result = compute_trendline(x, y, trend_type="exponential")
        assert_allclose(result.r_squared, 1.0, atol=1e-4)

    def test_recovers_parameters(self) -> None:
        x, y = _make_exponential_data(a=5.0, b=0.2)
        result = compute_trendline(x, y, trend_type="exponential")
        assert_allclose(result.coefficients[0], 5.0, rtol=0.1)
        assert_allclose(result.coefficients[1], 0.2, rtol=0.1)

    def test_equation_format(self) -> None:
        x, y = _make_exponential_data()
        result = compute_trendline(x, y, trend_type="exponential")
        assert "exp" in result.equation
        assert result.trend_type == "exponential"

    def test_requires_positive_y(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([-1.0, -2.0, -3.0])
        with pytest.raises(ValueError, match="positive y values"):
            compute_trendline(x, y, trend_type="exponential")


# ── Power Trendline ─────────────────────────────────────────────────────


class TestPowerTrendline:
    """Test power trendline fitting."""

    def test_perfect_power(self) -> None:
        x, y = _make_power_data(a=3.0, bp=2.0)
        result = compute_trendline(x, y, trend_type="power")
        assert_allclose(result.r_squared, 1.0, atol=1e-4)

    def test_recovers_exponent(self) -> None:
        x, y = _make_power_data(a=1.0, bp=0.5)  # square root
        result = compute_trendline(x, y, trend_type="power")
        assert_allclose(result.coefficients[1], 0.5, rtol=0.05)

    def test_equation_format(self) -> None:
        x, y = _make_power_data()
        result = compute_trendline(x, y, trend_type="power")
        assert "x^" in result.equation

    def test_requires_positive_values(self) -> None:
        """If all x values are negative, insufficient positive points remain."""
        x = np.array([-1.0, -2.0, -3.0])
        y = np.array([1.0, 4.0, 9.0])
        with pytest.raises(ValueError):
            compute_trendline(x, y, trend_type="power")


# ── NaN Handling ─────────────────────────────────────────────────────────


class TestNaNHandling:
    """Test that NaN values are properly handled."""

    def test_nan_in_x_removed(self) -> None:
        x = np.array([1, 2, np.nan, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        result = compute_trendline(x, y, trend_type="linear")
        assert result.r_squared > 0.99

    def test_nan_in_y_removed(self) -> None:
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, np.nan, 6, 8, 10], dtype=float)
        result = compute_trendline(x, y, trend_type="linear")
        assert result.r_squared > 0.99

    def test_too_few_after_nan(self) -> None:
        x = np.array([1, np.nan, np.nan])
        y = np.array([2, np.nan, np.nan])
        with pytest.raises(ValueError, match="2 valid"):
            compute_trendline(x, y)


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases."""

    def test_insufficient_data(self) -> None:
        x = np.array([1.0])
        y = np.array([2.0])
        with pytest.raises(ValueError, match="2 valid"):
            compute_trendline(x, y)

    def test_unknown_type_rejected(self) -> None:
        x, y = _make_linear_data()
        with pytest.raises(ValueError, match="Unknown trend type"):
            compute_trendline(x, y, trend_type="logistic")

    def test_two_points_linear(self) -> None:
        """Minimum viable data: 2 points should fit exactly."""
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 20.0])
        result = compute_trendline(x, y, trend_type="linear")
        assert_allclose(result.r_squared, 1.0)
        assert_allclose(result.coefficients[0], 2.0)
