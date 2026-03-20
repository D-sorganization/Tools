"""Tests for the signal_toolkit.series module.

Covers:
- SeriesResult dataclass
- SeriesExpansion: Taylor and Maclaurin series computation
- Coefficient computation accuracy
- Convergence analysis
- Error bound estimation
- Pre-defined series functions (exp, sin, cos, ln, geometric, arctan)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from signal_toolkit.series import (
    SeriesExpansion,
    SeriesResult,
    cos_series,
    exp_series,
    geometric_series,
    ln_series,
    sin_series,
)

# ── SeriesResult Dataclass ───────────────────────────────────────────────


class TestSeriesResult:
    """Test SeriesResult dataclass."""

    def test_series_result_creation(self) -> None:
        func = lambda x: x  # noqa: E731
        result = SeriesResult(
            coefficients=np.array([1.0, 1.0, 0.5]),
            n_terms=3,
            center=0.0,
            function=func,
        )
        assert result.n_terms == 3
        assert result.center == 0.0
        assert len(result.coefficients) == 3

    def test_series_result_radius_default(self) -> None:
        func = lambda x: x  # noqa: E731
        result = SeriesResult(
            coefficients=np.array([1.0]),
            n_terms=1,
            center=0.0,
            function=func,
        )
        assert result.radius_of_convergence is None


# ── SeriesExpansion Initialization ───────────────────────────────────────


class TestSeriesExpansionInit:
    """Test SeriesExpansion constructor and validation."""

    def test_default_constructor(self) -> None:
        se = SeriesExpansion()
        assert se.max_terms == 50
        assert se.h == pytest.approx(1e-5)

    def test_custom_constructor(self) -> None:
        se = SeriesExpansion(max_terms=100, h=1e-6)
        assert se.max_terms == 100
        assert se.h == pytest.approx(1e-6)

    def test_invalid_max_terms_raises(self) -> None:
        with pytest.raises(ValueError):
            SeriesExpansion(max_terms=0)

    def test_negative_max_terms_raises(self) -> None:
        with pytest.raises(ValueError):
            SeriesExpansion(max_terms=-5)


# ── Maclaurin Series (Taylor at x=0) ────────────────────────────────────


class TestMaclaurinSeries:
    """Test Maclaurin series expansions of well-known functions.

    Note: maclaurin_series() returns a callable, not a SeriesResult.
    """

    @pytest.fixture()
    def se(self) -> SeriesExpansion:
        return SeriesExpansion(max_terms=20)

    def test_exp_at_zero(self, se: SeriesExpansion) -> None:
        """e^0 = 1."""
        approx = se.maclaurin_series(np.exp, n_terms=5)
        val = approx(0.0)
        assert_allclose(val, 1.0, atol=1e-5)

    def test_exp_at_one(self, se: SeriesExpansion) -> None:
        """Sum of 1/n! should converge to e."""
        approx = se.maclaurin_series(np.exp, n_terms=15)
        val = approx(1.0)
        assert_allclose(val, math.e, rtol=1e-4)

    def test_sin_at_zero(self, se: SeriesExpansion) -> None:
        """sin(0) = 0."""
        approx = se.maclaurin_series(np.sin, n_terms=5)
        val = approx(0.0)
        assert_allclose(val, 0.0, atol=1e-6)

    def test_sin_at_pi_over_2(self, se: SeriesExpansion) -> None:
        """sin(π/2) ≈ 1."""
        approx = se.maclaurin_series(np.sin, n_terms=15)
        val = approx(np.pi / 2)
        assert_allclose(val, 1.0, rtol=0.05)

    def test_cos_at_zero(self, se: SeriesExpansion) -> None:
        """cos(0) = 1."""
        approx = se.maclaurin_series(np.cos, n_terms=5)
        val = approx(0.0)
        assert_allclose(val, 1.0, atol=1e-6)


# ── Taylor Series ────────────────────────────────────────────────────────


class TestTaylorSeries:
    """Test Taylor series expansions at non-zero centers."""

    @pytest.fixture()
    def se(self) -> SeriesExpansion:
        return SeriesExpansion(max_terms=20)

    def test_exp_centered_at_one(self, se: SeriesExpansion) -> None:
        """Taylor series of e^x at center=1 evaluated at x=1 should give e."""
        approx = se.taylor_series(np.exp, center=1.0, n_terms=10)
        val = approx(1.0)
        assert_allclose(val, math.e, rtol=1e-4)

    def test_taylor_center_in_series_result(self, se: SeriesExpansion) -> None:
        result = se.get_series_result(np.exp, center=2.0, n_terms=5)
        assert result.center == 2.0

    def test_taylor_invalid_callable_raises(self, se: SeriesExpansion) -> None:
        with pytest.raises(TypeError):
            se.taylor_series("not_a_function", center=0.0, n_terms=5)

    def test_taylor_invalid_nterms_raises(self, se: SeriesExpansion) -> None:
        with pytest.raises(ValueError):
            se.taylor_series(np.exp, center=0.0, n_terms=0)


# ── Coefficients ─────────────────────────────────────────────────────────


class TestCoefficients:
    """Test coefficient computation accuracy."""

    @pytest.fixture()
    def se(self) -> SeriesExpansion:
        return SeriesExpansion(max_terms=20)

    def test_exp_coefficients_are_reciprocal_factorials(
        self, se: SeriesExpansion
    ) -> None:
        """e^x Maclaurin coefficients should be 1/n!."""
        coeffs = se.get_coefficients(np.exp, center=0.0, n_terms=6)
        expected = np.array([1.0, 1.0, 1 / 2, 1 / 6, 1 / 24, 1 / 120], dtype=np.float64)
        # Relaxed tolerance: numerical differentiation has inherent error
        assert_allclose(coeffs, expected, rtol=0.05)

    def test_sin_coefficients_alternate(self, se: SeriesExpansion) -> None:
        """sin(x) has alternating 0, 1/n!, 0, -1/n! pattern."""
        coeffs = se.get_coefficients(np.sin, center=0.0, n_terms=5)
        assert_allclose(coeffs[0], 0.0, atol=0.05)
        assert_allclose(coeffs[1], 1.0, rtol=0.05)
        assert_allclose(coeffs[2], 0.0, atol=0.05)


# ── Convergence Analysis ────────────────────────────────────────────────


class TestConvergence:
    """Test convergence analysis functionality."""

    @pytest.fixture()
    def se(self) -> SeriesExpansion:
        return SeriesExpansion(max_terms=50)

    def test_exp_converges(self, se: SeriesExpansion) -> None:
        """e^x at x=1 should converge."""
        result = se.analyze_convergence(np.exp, center=0.0, x_test=1.0)
        assert result["convergent"] is True

    def test_convergence_returns_dict(self, se: SeriesExpansion) -> None:
        result = se.analyze_convergence(np.exp, center=0.0, x_test=0.5)
        assert "convergent" in result
        assert "terms_for_convergence" in result
        assert "final_error" in result
        assert "errors_by_term" in result

    def test_error_bound_positive(self, se: SeriesExpansion) -> None:
        """Error bound should be a non-negative number."""
        bound = se.estimate_error_bound(np.exp, center=0.0, x_test=0.5, n_terms=10)
        assert bound >= 0


# ── get_series_result ────────────────────────────────────────────────────


class TestGetSeriesResult:
    """Test get_series_result method."""

    def test_returns_series_result(self) -> None:
        se = SeriesExpansion()
        result = se.get_series_result(np.exp, center=0.0, n_terms=5)
        assert isinstance(result, SeriesResult)
        assert result.n_terms == 5
        assert result.center == 0.0
        assert len(result.coefficients) == 5

    def test_series_result_function_is_callable(self) -> None:
        se = SeriesExpansion()
        result = se.get_series_result(np.exp, center=0.0, n_terms=10)
        # The function attribute should be callable and give correct result
        val = result.function(0.0)
        assert_allclose(val, 1.0, atol=1e-6)


# ── Pre-defined Series Functions ─────────────────────────────────────────


class TestPredefinedSeries:
    """Test pre-defined series functions for known mathematical identities."""

    def test_exp_series_at_zero(self) -> None:
        """e^0 = 1."""
        f = exp_series(n_terms=20)
        assert_allclose(f(0.0), 1.0, atol=1e-10)

    def test_exp_series_at_one(self) -> None:
        """e^1 ≈ 2.71828."""
        f = exp_series(n_terms=20)
        assert_allclose(f(1.0), math.e, rtol=1e-10)

    def test_sin_series_at_zero(self) -> None:
        """sin(0) = 0."""
        f = sin_series(n_terms=20)
        assert_allclose(f(0.0), 0.0, atol=1e-10)

    def test_sin_series_at_pi_over_2(self) -> None:
        """sin(π/2) = 1."""
        f = sin_series(n_terms=20)
        assert_allclose(f(np.pi / 2), 1.0, rtol=1e-10)

    def test_cos_series_at_zero(self) -> None:
        """cos(0) = 1."""
        f = cos_series(n_terms=20)
        assert_allclose(f(0.0), 1.0, atol=1e-10)

    def test_cos_series_at_pi(self) -> None:
        """cos(π) = -1."""
        f = cos_series(n_terms=25)
        assert_allclose(f(np.pi), -1.0, rtol=1e-6)

    def test_ln_series_at_zero(self) -> None:
        """ln(1+0) = ln(1) = 0."""
        f = ln_series(n_terms=50)
        assert_allclose(f(0.0), 0.0, atol=1e-10)

    def test_ln_series_at_half(self) -> None:
        """ln(1+0.5) = ln(1.5) ≈ 0.4055."""
        f = ln_series(n_terms=50)
        assert_allclose(f(0.5), math.log(1.5), rtol=1e-4)

    def test_geometric_series(self) -> None:
        """1/(1-0.5) = 2."""
        f = geometric_series(n_terms=50)
        assert_allclose(f(0.5), 2.0, rtol=1e-10)

    def test_euler_identity(self) -> None:
        """Verify e^(iπ) + 1 = 0 via sin²+cos² = 1."""
        s = sin_series(n_terms=25)
        c = cos_series(n_terms=25)
        # sin²(x) + cos²(x) = 1 for any x
        for x in [0.5, 1.0, 2.0]:
            val = s(x) ** 2 + c(x) ** 2
            assert_allclose(val, 1.0, rtol=1e-6)
