"""Tests for CrossCorrelationAnalyzer — core cross-correlation analysis.

Covers cross_correlate, lagged_correlation, find_optimal_lag,
rolling_cross_correlation, and multi_series_correlation_matrix.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from data_processor.core import cross_correlation as cross_correlation_module
from data_processor.core.cross_correlation import (
    CrossCorrelationAnalyzer,
    CrossCorrelationConfig,
    CrossCorrelationResult,
    cross_correlate,
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


def test_cross_correlation_module_imports() -> None:
    """The module should import with real Numba installed."""
    assert cross_correlation_module.CrossCorrelationAnalyzer is CrossCorrelationAnalyzer


def test_instance_methods_remain_plain_python_functions() -> None:
    """Self methods must not be wrapped in Numba dispatchers."""
    method_names = (
        "rolling_cross_correlation",
        "_compute_pvalues",
        "_select_lag_order",
        "_create_lag_matrix",
        "_conditional_entropy",
    )

    for method_name in method_names:
        method = CrossCorrelationAnalyzer.__dict__[method_name]
        assert inspect.isfunction(method), method_name
        assert not hasattr(method, "py_func"), method_name


class TestCrossCorrelate:
    """Tests for the core cross_correlate method."""

    def test_identical_signals_max_at_zero(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Identical signals should have max correlation at lag 0."""
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        result = analyzer.cross_correlate(x, x)
        assert result.optimal_lag == 0
        assert result.max_correlation == pytest.approx(1.0, abs=0.05)

    def test_result_has_correct_fields(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
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

    def test_convenience_function_returns_pvalues(self) -> None:
        """Module-level cross_correlate should run the p-value path."""
        rng = np.random.default_rng(3667)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)

        result = cross_correlate(x, y, max_lag=5)

        assert result.p_values is not None
        assert len(result.p_values) == len(result.ccf_values)
        assert np.all(np.isfinite(result.p_values))

    def test_mismatched_lengths_raises(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Unequal series should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            analyzer.cross_correlate(np.array([1.0, 2.0]), np.array([1.0]))

    def test_compute_pvalues_handles_perfect_correlation_extremes(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Exact +/-1 correlations avoid division-by-zero p-value math."""
        p_values = analyzer._compute_pvalues(np.array([-1.0, 0.0, 1.0]), n=50)

        assert p_values[0] == 0.0
        assert p_values[1] == pytest.approx(1.0)
        assert p_values[2] == 0.0

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, float("nan")])
    def test_normal_ppf_rejects_out_of_range_probabilities(
        self, analyzer: CrossCorrelationAnalyzer, p: float
    ) -> None:
        """Out-of-range probabilities must not silently return median z=0."""
        with pytest.raises(ValueError, match="p must"):
            analyzer._normal_ppf(p)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, float("nan")])
    def test_confidence_interval_rejects_invalid_significance_level(
        self, analyzer: CrossCorrelationAnalyzer, alpha: float
    ) -> None:
        """Invalid alpha must not produce a zero-width confidence interval."""
        with pytest.raises(ValueError, match="significance_level"):
            analyzer._compute_confidence_interval(20, alpha)

    def test_invalid_significance_level_rejects_significance_path(self) -> None:
        """Degenerate alpha=0 should fail before significant-lag selection."""
        analyzer = CrossCorrelationAnalyzer(
            CrossCorrelationConfig(significance_level=0)
        )
        x = np.linspace(-1.0, 1.0, 20)

        with pytest.raises(ValueError, match="significance_level"):
            analyzer.cross_correlate(x, x, max_lag=3)


class TestLaggedCorrelation:
    """Tests for lagged_correlation method."""

    def test_zero_lag_matches_corrcoef(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
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

    def test_mismatched_lengths_raise_clear_contract(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Length mismatches should fail before NumPy shape errors."""
        with pytest.raises(ValueError, match="same length"):
            analyzer.lagged_correlation(np.arange(5.0), np.arange(4.0), lag=0)

    def test_lag_alignment_helper_is_shared_by_lagged_paths(self) -> None:
        """The duplicated lag-slicing block should stay single-sourced."""
        assert hasattr(CrossCorrelationAnalyzer, "_align_lagged_series")
        for method_name in (
            "lagged_correlation",
            "rolling_cross_correlation",
            "_compute_ccf_at_lag",
        ):
            source = inspect.getsource(getattr(CrossCorrelationAnalyzer, method_name))
            assert "_align_lagged_series" in source, method_name


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

    def test_returns_correct_window_size(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Result should report the window size used."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        result = analyzer.rolling_cross_correlation(x, y, window=20)
        assert result.window_size == 20
        assert len(result.correlations) == len(x)

    def test_correlation_stability_is_not_negative(self) -> None:
        """The documented 1-CV stability score should stay in [0, 1]."""
        x = np.linspace(-1.0, 1.0, 80)
        y = np.concatenate([x[:40], -x[40:]])
        analyzer = CrossCorrelationAnalyzer(
            CrossCorrelationConfig(rolling_min_periods=5)
        )

        result = analyzer.rolling_cross_correlation(x, y, window=10)

        assert result.correlation_stability >= 0.0

    def test_mismatched_lengths_raise_clear_contract(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Rolling correlation should share the public length precondition."""
        with pytest.raises(ValueError, match="same length"):
            analyzer.rolling_cross_correlation(np.arange(10.0), np.arange(9.0))


class TestCausalityRuntime:
    """Regression tests for causal-analysis runtime paths."""

    def test_granger_causality_test_runs_end_to_end(self) -> None:
        rng = np.random.default_rng(3667)
        x = rng.standard_normal(80)
        y = 0.35 * np.roll(x, 1) + rng.standard_normal(80) * 0.1
        y[0] = rng.standard_normal()
        analyzer = CrossCorrelationAnalyzer(CrossCorrelationConfig(granger_max_lag=2))

        result = analyzer.granger_causality_test(x, y, max_lag=2)

        assert np.isfinite(result.x_causes_y_fstat)
        assert np.isfinite(result.y_causes_x_fstat)
        assert np.isfinite(result.x_causes_y_pvalue)
        assert np.isfinite(result.y_causes_x_pvalue)

    def test_granger_causality_rejects_mismatched_lengths(self) -> None:
        analyzer = CrossCorrelationAnalyzer(CrossCorrelationConfig(granger_max_lag=2))

        with pytest.raises(ValueError, match="same length"):
            analyzer.granger_causality_test(np.arange(12.0), np.arange(11.0))

    def test_transfer_entropy_runs_end_to_end(self) -> None:
        rng = np.random.default_rng(3666)
        x = rng.standard_normal(60)
        y = 0.25 * np.roll(x, 1) + rng.standard_normal(60) * 0.2
        y[0] = rng.standard_normal()
        analyzer = CrossCorrelationAnalyzer(
            CrossCorrelationConfig(num_permutations=5, te_bins=4)
        )

        result = analyzer.transfer_entropy(x, y, history_length=1)

        assert np.isfinite(result.te_x_to_y)
        assert np.isfinite(result.te_y_to_x)
        assert 0.0 <= result.te_x_to_y_pvalue <= 1.0
        assert 0.0 <= result.te_y_to_x_pvalue <= 1.0

    @pytest.mark.parametrize("history_length", [0, 5])
    def test_transfer_entropy_rejects_invalid_history_length(
        self, history_length: int
    ) -> None:
        analyzer = CrossCorrelationAnalyzer(CrossCorrelationConfig(te_bins=3))
        x = np.arange(5.0)

        with pytest.raises(ValueError, match="history_length"):
            analyzer.transfer_entropy(x, x, history_length=history_length)

    def test_transfer_entropy_rejects_mismatched_lengths(self) -> None:
        analyzer = CrossCorrelationAnalyzer(CrossCorrelationConfig(te_bins=3))

        with pytest.raises(ValueError, match="same length"):
            analyzer.transfer_entropy(np.arange(8.0), np.arange(7.0))


class TestPartialCrossCorrelation:
    """Tests for partial_cross_correlation method."""

    def test_partial_cross_correlation_returns_result(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        rng = np.random.default_rng(3724)
        z = rng.standard_normal(80)
        x = 0.4 * z + rng.standard_normal(80) * 0.2
        y = 0.3 * z + 0.5 * x + rng.standard_normal(80) * 0.2

        result = analyzer.partial_cross_correlation(x, y, z, max_lag=4)

        assert isinstance(result, CrossCorrelationResult)
        assert len(result.lags) == 9
        assert np.all(np.isfinite(result.ccf_values))

    def test_partial_cross_correlation_rejects_mismatched_control_length(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="same length"):
            analyzer.partial_cross_correlation(
                np.arange(8.0), np.arange(8.0), np.arange(7.0)
            )


class TestLeadLagRelationship:
    """Tests for find_lead_lag_relationship method."""

    def test_finds_pairwise_relationship(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        n = 200
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        y = np.roll(x, 5)

        result = analyzer.find_lead_lag_relationship({"x": x, "y": y}, max_lag=20)

        relationship = result["x_vs_y"]
        assert relationship["leader"] in {"x", "y"}
        assert relationship["follower"] in {"x", "y"}
        assert relationship["leader"] != relationship["follower"]
        assert relationship["lag_magnitude"] <= 7

    def test_rejects_mismatched_series_lengths(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        with pytest.raises(ValueError, match="same length"):
            analyzer.find_lead_lag_relationship(
                {"x": np.arange(8.0), "y": np.arange(7.0)}
            )


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

    def test_mismatched_lengths_raise_clear_contract(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Multi-series correlations should validate all series lengths."""
        with pytest.raises(ValueError, match="same length"):
            analyzer.multi_series_correlation_matrix(
                {"x": np.arange(8.0), "y": np.arange(7.0)}
            )


class TestTransferEntropy:
    """Tests for transfer_entropy method."""

    def test_seeded_permutation_pvalues_repeat_on_same_analyzer(self) -> None:
        """Seeded transfer-entropy permutation tests should be reproducible."""
        rng = np.random.default_rng(123)
        x = rng.standard_normal(80)
        y = np.roll(x, 1) + rng.standard_normal(80) * 0.05
        analyzer = CrossCorrelationAnalyzer(
            CrossCorrelationConfig(
                num_permutations=25,
                permutation_random_seed=42,
                te_bins=4,
            )
        )

        first = analyzer.transfer_entropy(x, y)
        second = analyzer.transfer_entropy(x, y)

        assert second.te_x_to_y_pvalue == first.te_x_to_y_pvalue
        assert second.te_y_to_x_pvalue == first.te_y_to_x_pvalue
        assert second.dominant_direction == first.dominant_direction

    def test_permutation_test_does_not_use_global_numpy_permutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Transfer entropy should use its local generator for permutations."""
        rng = np.random.default_rng(456)
        x = rng.standard_normal(60)
        y = np.roll(x, 1) + rng.standard_normal(60) * 0.1
        analyzer = CrossCorrelationAnalyzer(
            CrossCorrelationConfig(
                num_permutations=5,
                permutation_random_seed=7,
                te_bins=4,
            )
        )

        def fail_global_permutation(_source: np.ndarray) -> np.ndarray:
            raise AssertionError("global np.random.permutation was called")

        monkeypatch.setattr(np.random, "permutation", fail_global_permutation)

        result = analyzer.transfer_entropy(x, y)

        assert 0.0 < result.te_x_to_y_pvalue <= 1.0
        assert 0.0 < result.te_y_to_x_pvalue <= 1.0
