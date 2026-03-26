"""Tests for CrossCorrelationAnalyzer — core cross-correlation analysis.

Covers cross_correlate, lagged_correlation, find_optimal_lag,
rolling_cross_correlation, and multi_series_correlation_matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
from data_processor.core.cross_correlation import (
    CrossCorrelationAnalyzer,
)


@pytest.fixture()
def analyzer() -> CrossCorrelationAnalyzer:
    """Create with default config."""
    return CrossCorrelationAnalyzer()


@pytest.fixture()
def sine_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two sine waves: y is x shifted by 5 samples."""
    n = 200
    t = np.linspace(0, 4 * np.pi, n)
    x = np.sin(t)
    y = np.sin(t - 5 * (4 * np.pi / n))
    return x, y


class TestCrossCorrelate:
    """Tests for the core cross_correlate method."""

    def test_identical_signals_max_at_zero(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Identical signals should have max correlation at lag 0."""
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = analyzer.cross_correlate(x, x)
        assert result.optimal_lag == 0
        assert result.max_correlation == pytest.approx(1.0, abs=0.05)

    def test_result_has_correct_fields(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Result should contain all expected fields."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        result = analyzer.cross_correlate(x, y, max_lag=10)
        assert hasattr(result, "lags")
        assert hasattr(result, "ccf_values")
        assert hasattr(result, "optimal_lag")
        assert hasattr(result, "max_correlation")
        assert hasattr(result, "confidence_interval")
        assert len(result.lags) == 21  # -10 to +10

    def test_mismatched_lengths_raises(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Unequal series should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            analyzer.cross_correlate(np.array([1.0, 2.0]), np.array([1.0]))


class TestLaggedCorrelation:
    """Tests for lagged_correlation method."""

    def test_zero_lag_matches_corrcoef(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Lag-0 correlation should match numpy corrcoef."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
        corr, _p = analyzer.lagged_correlation(x, y, lag=0)
        expected = float(np.corrcoef(x, y)[0, 1])
        assert corr == pytest.approx(expected, abs=0.01)

    def test_large_lag_returns_zero(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Lag >= n should return (0.0, 1.0)."""
        x = np.arange(5.0)
        corr, p = analyzer.lagged_correlation(x, x, lag=100)
        assert corr == 0.0
        assert p == 1.0


class TestFindOptimalLag:
    """Tests for find_optimal_lag method."""

    def test_finds_shifted_signal_lag(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Should detect the lag between shifted copies."""
        n = 200
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        shift = 5
        y = np.roll(x, shift)
        lag, _corr = analyzer.find_optimal_lag(x, y, max_lag=20)
        assert abs(lag) <= shift + 2


class TestRollingCrossCorrelation:
    """Tests for rolling_cross_correlation method."""

    def test_returns_correct_window_size(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Result should report the window size used."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        result = analyzer.rolling_cross_correlation(x, y, window=20)
        assert result.window_size == 20
        assert len(result.correlations) == len(x)


class TestMultiSeries:
    """Tests for multi_series_correlation_matrix."""

    def test_identity_diagonal(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Diagonal of correlation matrix should be 1.0."""
        rng = np.random.default_rng(42)
        series = {
            "a": rng.standard_normal(50),
            "b": rng.standard_normal(50),
            "c": rng.standard_normal(50),
        }
        mat, names = analyzer.multi_series_correlation_matrix(series)
        assert mat.shape == (3, 3)
        np.testing.assert_array_almost_equal(np.diag(mat), [1.0, 1.0, 1.0])
        assert names == ["a", "b", "c"]

    def test_symmetric(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Correlation matrix should be symmetric."""
        series = {
            "x": np.sin(np.arange(50.0)),
            "y": np.cos(np.arange(50.0)),
        }
        mat, _ = analyzer.multi_series_correlation_matrix(series)
        np.testing.assert_array_almost_equal(mat, mat.T)
