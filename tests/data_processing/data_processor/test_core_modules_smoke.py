"""Smoke tests for 5 largest untested data_processor core modules.

Covers: anova, cross_correlation, time_series_decomposition,
uncertainty_quantification, state_space.

Each test imports the module, instantiates the main class with sample
data, and calls the primary public method(s).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------


class TestANOVASmoke:
    """Smoke tests for data_processor.core.anova."""

    def test_import(self):
        from data_processor.core.anova import ANOVAAnalyzer, OneWayANOVAResult

        assert ANOVAAnalyzer is not None
        assert OneWayANOVAResult is not None

    def test_one_way_anova(self):
        from data_processor.core.anova import ANOVAAnalyzer

        rng = np.random.default_rng(42)
        groups = {
            "A": rng.normal(10, 2, 30),
            "B": rng.normal(12, 2, 30),
            "C": rng.normal(10.5, 2, 30),
        }
        analyzer = ANOVAAnalyzer(alpha=0.05)
        result = analyzer.one_way_anova(groups)
        assert hasattr(result, "f_statistic")
        assert hasattr(result, "p_value")
        assert result.f_statistic >= 0

    def test_two_way_anova(self):
        import pandas as pd
        from data_processor.core.anova import ANOVAAnalyzer

        rng = np.random.default_rng(42)
        n = 60
        df = pd.DataFrame(
            {
                "value": rng.normal(10, 2, n),
                "factor_a": np.tile(["low", "high"], n // 2),
                "factor_b": np.repeat(["x", "y", "z"], n // 3),
            }
        )
        analyzer = ANOVAAnalyzer()
        result = analyzer.two_way_anova(df, "value", "factor_a", "factor_b")
        assert hasattr(result, "anova_table")


# ---------------------------------------------------------------------------
# Cross-Correlation
# ---------------------------------------------------------------------------


class TestCrossCorrelationSmoke:
    """Smoke tests for data_processor.core.cross_correlation."""

    def test_import(self):
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        assert CrossCorrelationAnalyzer is not None

    def test_cross_correlate(self):
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = np.roll(x, 5) + rng.normal(0, 0.1, 200)
        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.cross_correlate(x, y)
        assert hasattr(result, "lags")
        assert hasattr(result, "ccf")
        assert len(result.ccf) > 0

    def test_find_optimal_lag(self):
        from data_processor.core.cross_correlation import CrossCorrelationAnalyzer

        rng = np.random.default_rng(42)
        x = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.normal(0, 0.1, 200)
        y = np.roll(x, 3) + rng.normal(0, 0.1, 200)
        analyzer = CrossCorrelationAnalyzer()
        result = analyzer.find_optimal_lag(x, y)
        assert hasattr(result, "optimal_lag")


# ---------------------------------------------------------------------------
# Time-Series Decomposition
# ---------------------------------------------------------------------------


class TestTimeSeriesDecompositionSmoke:
    """Smoke tests for data_processor.core.time_series_decomposition."""

    def test_import(self):
        from data_processor.core.time_series_decomposition import TimeSeriesDecomposer

        assert TimeSeriesDecomposer is not None

    def test_decompose(self):
        from data_processor.core.time_series_decomposition import (
            DecompositionConfig,
            DecompositionMethod,
            TimeSeriesDecomposer,
        )

        t = np.linspace(0, 4 * np.pi, 200)
        data = (
            np.sin(t)
            + 0.5 * t / (4 * np.pi)
            + np.random.default_rng(42).normal(0, 0.1, 200)
        )
        config = DecompositionConfig(
            method=DecompositionMethod.MOVING_AVERAGE,
            period=50,
        )
        decomposer = TimeSeriesDecomposer(config)
        result = decomposer.decompose(data)
        assert hasattr(result, "trend")
        assert hasattr(result, "seasonal")
        assert hasattr(result, "residual")
        assert len(result.trend) == len(data)

    def test_detect_seasonality(self):
        from data_processor.core.time_series_decomposition import TimeSeriesDecomposer

        t = np.linspace(0, 8 * np.pi, 400)
        data = np.sin(t) + 0.2 * np.random.default_rng(42).normal(0, 1, 400)
        decomposer = TimeSeriesDecomposer()
        result = decomposer.detect_seasonality(data)
        assert hasattr(result, "period")


# ---------------------------------------------------------------------------
# Uncertainty Quantification
# ---------------------------------------------------------------------------


class TestUncertaintyQuantificationSmoke:
    """Smoke tests for data_processor.core.uncertainty_quantification."""

    def test_import(self):
        from data_processor.core.uncertainty_quantification import UncertaintyQuantifier

        assert UncertaintyQuantifier is not None

    def test_bootstrap_ci(self):
        from data_processor.core.uncertainty_quantification import (
            UncertaintyConfig,
            UncertaintyQuantifier,
        )

        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, 100)
        config = UncertaintyConfig(n_bootstrap=200, random_seed=42)
        uq = UncertaintyQuantifier(config)
        result = uq.bootstrap_ci(data, statistic=np.mean)
        assert hasattr(result, "confidence_interval")
        assert result.confidence_interval.lower < result.confidence_interval.upper

    def test_monte_carlo_propagation(self):
        from data_processor.core.uncertainty_quantification import (
            UncertaintyConfig,
            UncertaintyQuantifier,
        )

        config = UncertaintyConfig(n_monte_carlo=500, random_seed=42)
        uq = UncertaintyQuantifier(config)

        def model(x, y):
            return x * y

        means = [5.0, 3.0]
        stds = [0.5, 0.3]
        result = uq.monte_carlo_propagation(model, means, stds)
        assert hasattr(result, "mean")
        assert hasattr(result, "std")
        assert result.std > 0


# ---------------------------------------------------------------------------
# State Space
# ---------------------------------------------------------------------------


class TestStateSpaceSmoke:
    """Smoke tests for data_processor.core.state_space."""

    def test_import(self):
        from data_processor.core.state_space import LocalLevelModel, StateSpaceConfig

        assert LocalLevelModel is not None
        assert StateSpaceConfig is not None

    def test_local_level_fit(self):
        from data_processor.core.state_space import LocalLevelModel, StateSpaceConfig

        rng = np.random.default_rng(42)
        # Random walk + noise
        n = 100
        level = np.cumsum(rng.normal(0, 0.5, n))
        data = level + rng.normal(0, 1, n)

        config = StateSpaceConfig(max_iterations=50)
        model = LocalLevelModel(config)
        result = model.fit(data)
        assert hasattr(result, "filtered_states")
        assert hasattr(result, "log_likelihood")
        assert len(result.filtered_states) == n

    def test_local_level_forecast(self):
        from data_processor.core.state_space import LocalLevelModel, StateSpaceConfig

        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 0.5, 80)) + rng.normal(0, 1, 80)

        config = StateSpaceConfig(max_iterations=50, forecast_horizon=5)
        model = LocalLevelModel(config)
        model.fit(data)
        forecast = model.forecast(steps=5)
        assert hasattr(forecast, "mean")
        assert len(forecast.mean) == 5
