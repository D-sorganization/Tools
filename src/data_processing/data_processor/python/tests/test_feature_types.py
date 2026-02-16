"""Tests for data_processor.core.feature_types module."""

from __future__ import annotations

import numpy as np
import pytest

from data_processor.core.feature_types import (
    FeatureCategory,
    FeatureConfig,
    FeatureResult,
    SelectionMethod,
    SelectionResult,
    TransformationType,
)


class TestFeatureCategory:
    """Tests for FeatureCategory enum."""

    def test_all_values(self) -> None:
        assert FeatureCategory.STATISTICAL.value == "statistical"
        assert FeatureCategory.TIME_DOMAIN.value == "time_domain"
        assert FeatureCategory.FREQUENCY_DOMAIN.value == "frequency_domain"
        assert FeatureCategory.ROLLING.value == "rolling"
        assert FeatureCategory.LAG.value == "lag"
        assert FeatureCategory.POLYNOMIAL.value == "polynomial"
        assert FeatureCategory.INTERACTION.value == "interaction"
        assert FeatureCategory.CUSTOM.value == "custom"

    def test_count(self) -> None:
        assert len(FeatureCategory) == 8


class TestSelectionMethod:
    """Tests for SelectionMethod enum."""

    def test_all_values(self) -> None:
        assert SelectionMethod.CORRELATION.value == "correlation"
        assert SelectionMethod.MUTUAL_INFO.value == "mutual_info"
        assert SelectionMethod.VARIANCE.value == "variance"
        assert SelectionMethod.IMPORTANCE.value == "importance"
        assert SelectionMethod.RECURSIVE.value == "recursive"


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_values(self) -> None:
        assert TransformationType.LOG.value == "log"
        assert TransformationType.SQRT.value == "sqrt"
        assert TransformationType.SQUARE.value == "square"
        assert TransformationType.RECIPROCAL.value == "reciprocal"
        assert TransformationType.STANDARDIZE.value == "standardize"
        assert TransformationType.NORMALIZE.value == "normalize"
        assert TransformationType.QUANTILE.value == "quantile"
        assert TransformationType.BINNING.value == "binning"

    def test_count(self) -> None:
        assert len(TransformationType) == 8


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_defaults(self) -> None:
        config = FeatureConfig()
        assert config.compute_mean is True
        assert config.compute_std is True
        assert config.compute_min is True
        assert config.compute_max is True
        assert config.compute_skewness is True
        assert config.compute_kurtosis is True
        assert config.polynomial_degree == 2
        assert config.include_interactions is True
        assert config.correlation_threshold == 0.95
        assert config.variance_threshold == 0.01
        assert config.n_bins == 10

    def test_default_rolling_windows(self) -> None:
        config = FeatureConfig()
        assert config.rolling_windows == [5, 10, 20]

    def test_default_lag_values(self) -> None:
        config = FeatureConfig()
        assert config.lag_values == [1, 2, 3, 5, 10]

    def test_default_percentiles(self) -> None:
        config = FeatureConfig()
        assert config.compute_percentiles == [25, 75]

    def test_default_spectral_percentiles(self) -> None:
        config = FeatureConfig()
        assert config.spectral_percentiles == [25, 50, 75, 95]


class TestFeatureResult:
    """Tests for FeatureResult dataclass."""

    def test_construction(self) -> None:
        result = FeatureResult(
            features=np.array([[1.0, 2.0], [3.0, 4.0]]),
            feature_names=["f1", "f2"],
            n_samples=2,
            n_features=2,
            categories={"statistical": ["f1", "f2"]},
            feature_stats={"f1": {"mean": 2.0}},
        )
        assert result.n_samples == 2
        assert result.n_features == 2
        assert len(result.feature_names) == 2
        assert result.features.shape == (2, 2)


class TestSelectionResult:
    """Tests for SelectionResult dataclass."""

    def test_construction(self) -> None:
        result = SelectionResult(
            selected_indices=np.array([0, 2]),
            selected_names=["f1", "f3"],
            removed_names=["f2"],
            scores={"f1": 0.9, "f2": 0.1, "f3": 0.8},
            threshold_used=0.5,
            n_original=3,
            n_selected=2,
        )
        assert result.n_original == 3
        assert result.n_selected == 2
        assert len(result.removed_names) == 1
        assert result.threshold_used == 0.5
