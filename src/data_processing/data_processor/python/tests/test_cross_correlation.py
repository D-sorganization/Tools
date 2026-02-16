"""Tests for data_processor.core.cross_correlation module."""

from __future__ import annotations

import numpy as np
import pytest

from data_processor.core.cross_correlation import (
    CausalityMethod,
    CrossCorrelationAnalyzer,
    CrossCorrelationConfig,
    CrossCorrelationResult,
    GrangerCausalityResult,
    NormalizationMethod,
    RollingCorrelationResult,
    TransferEntropyResult,
)


class TestNormalizationMethod:
    """Tests for NormalizationMethod enum."""

    def test_values(self) -> None:
        assert NormalizationMethod.NONE.value == "none"
        assert NormalizationMethod.BIASED.value == "biased"
        assert NormalizationMethod.UNBIASED.value == "unbiased"
        assert NormalizationMethod.COEFF.value == "coeff"


class TestCausalityMethod:
    """Tests for CausalityMethod enum."""

    def test_values(self) -> None:
        assert CausalityMethod.GRANGER.value == "granger"
        assert CausalityMethod.TRANSFER_ENTROPY.value == "transfer_entropy"
        assert CausalityMethod.CONVERGENT_CROSS_MAPPING.value == "ccm"


class TestCrossCorrelationConfig:
    """Tests for CrossCorrelationConfig dataclass."""

    def test_defaults(self) -> None:
        config = CrossCorrelationConfig()
        assert config.max_lag is None
        assert config.normalization == NormalizationMethod.COEFF
        assert config.detrend is True
        assert config.remove_mean is True
        assert config.significance_level == 0.05
        assert config.num_permutations == 1000
        assert config.granger_max_lag == 10
        assert config.granger_criterion == "aic"
        assert config.rolling_window is None
        assert config.rolling_min_periods is None
        assert config.te_history_length == 1
        assert config.te_bins == 10

    def test_custom_values(self) -> None:
        config = CrossCorrelationConfig(
            max_lag=50,
            normalization=NormalizationMethod.UNBIASED,
            detrend=False,
            significance_level=0.01,
        )
        assert config.max_lag == 50
        assert config.normalization == NormalizationMethod.UNBIASED
        assert config.detrend is False
        assert config.significance_level == 0.01


class TestCrossCorrelationResult:
    """Tests for CrossCorrelationResult dataclass."""

    def test_construction(self) -> None:
        result = CrossCorrelationResult(
            lags=np.arange(-10, 11),
            ccf_values=np.zeros(21),
            optimal_lag=0,
            max_correlation=1.0,
            correlation_at_zero=1.0,
            confidence_interval=(-0.2, 0.2),
            significant_lags=[0],
        )
        assert len(result.lags) == 21
        assert result.optimal_lag == 0
        assert result.max_correlation == 1.0
        assert result.series_x_name == "X"
        assert result.series_y_name == "Y"

    def test_is_significant_at_lag(self) -> None:
        result = CrossCorrelationResult(
            lags=np.arange(-5, 6),
            ccf_values=np.zeros(11),
            optimal_lag=0,
            max_correlation=0.9,
            correlation_at_zero=0.9,
            confidence_interval=(-0.3, 0.3),
            significant_lags=[0, 1, -1],
        )
        assert result.is_significant_at_lag(0) is True
        assert result.is_significant_at_lag(1) is True
        assert result.is_significant_at_lag(3) is False


class TestGrangerCausalityResult:
    """Tests for GrangerCausalityResult dataclass."""

    def test_construction(self) -> None:
        result = GrangerCausalityResult(
            x_causes_y=True,
            x_causes_y_pvalue=0.01,
            x_causes_y_fstat=5.5,
            y_causes_x=False,
            y_causes_x_pvalue=0.9,
            y_causes_x_fstat=0.1,
            optimal_lag_xy=2,
            optimal_lag_yx=1,
            causal_direction="x->y",
        )
        assert result.x_causes_y is True
        assert result.y_causes_x is False
        assert result.causal_direction == "x->y"


class TestTransferEntropyResult:
    """Tests for TransferEntropyResult dataclass."""

    def test_construction(self) -> None:
        result = TransferEntropyResult(
            te_x_to_y=0.5,
            te_y_to_x=0.1,
            net_te=0.4,
            te_x_to_y_pvalue=0.01,
            te_y_to_x_pvalue=0.8,
            dominant_direction="x->y",
        )
        assert result.net_te == pytest.approx(0.4)
        assert result.dominant_direction == "x->y"


class TestRollingCorrelationResult:
    """Tests for RollingCorrelationResult dataclass."""

    def test_construction(self) -> None:
        result = RollingCorrelationResult(
            timestamps=np.arange(100),
            correlations=np.random.default_rng(42).random(100),
            window_size=20,
            mean_correlation=0.5,
            std_correlation=0.1,
            correlation_stability=0.9,
        )
        assert result.window_size == 20
        assert result.correlation_stability == 0.9


class TestCrossCorrelationAnalyzer:
    """Tests for CrossCorrelationAnalyzer class."""

    @pytest.fixture()
    def analyzer(self) -> CrossCorrelationAnalyzer:
        return CrossCorrelationAnalyzer()

    def test_construction(self, analyzer: CrossCorrelationAnalyzer) -> None:
        assert analyzer is not None

    def test_identical_signals(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Identical signals should have max correlation at lag=0."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        result = analyzer.cross_correlate(x, x)
        assert isinstance(result, CrossCorrelationResult)
        assert result.optimal_lag == 0
        assert result.max_correlation == pytest.approx(1.0, abs=0.05)

    def test_lagged_signal(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """A shifted signal should show optimal lag matching the shift."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        lag = 5
        y = np.zeros_like(x)
        y[lag:] = x[:-lag]
        result = analyzer.cross_correlate(x, y, max_lag=20)
        # Optimal lag should be close to the actual shift
        assert abs(result.optimal_lag - lag) <= 2

    def test_uncorrelated_signals(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Independent random signals should show low correlation."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        y = rng.random(200)
        result = analyzer.cross_correlate(x, y)
        assert abs(result.max_correlation) < 0.3

    def test_lagged_correlation_at_zero(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Lagged correlation at lag=0 for identical signals should be ~1."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        corr, _ = analyzer.lagged_correlation(x, x, lag=0)
        assert corr == pytest.approx(1.0, abs=0.05)

    def test_find_optimal_lag(self, analyzer: CrossCorrelationAnalyzer) -> None:
        """Find optimal lag for a shifted signal."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        lag = 3
        y = np.zeros_like(x)
        y[lag:] = x[:-lag]
        opt_lag, corr = analyzer.find_optimal_lag(x, y, max_lag=10)
        assert abs(opt_lag - lag) <= 2
        assert abs(corr) > 0.5

    def test_rolling_cross_correlation(
        self, analyzer: CrossCorrelationAnalyzer
    ) -> None:
        """Rolling correlation should produce a result with correlations array."""
        rng = np.random.default_rng(42)
        x = rng.random(200)
        y = rng.random(200)
        result = analyzer.rolling_cross_correlation(x, y, window=50)
        assert isinstance(result, RollingCorrelationResult)
        assert result.window_size == 50
        assert len(result.correlations) > 0

    def test_custom_config(self) -> None:
        """Test analyzer with custom configuration."""
        config = CrossCorrelationConfig(
            max_lag=20,
            detrend=False,
            significance_level=0.01,
        )
        analyzer = CrossCorrelationAnalyzer(config=config)
        rng = np.random.default_rng(42)
        x = rng.random(100)
        result = analyzer.cross_correlate(x, x)
        assert result.optimal_lag == 0
