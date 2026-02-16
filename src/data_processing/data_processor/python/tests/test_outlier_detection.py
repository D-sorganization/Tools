"""Tests for data_processor.core.outlier_detection module."""

from __future__ import annotations

import numpy as np
import pytest

from data_processor.core.outlier_detection import (
    OutlierConfig,
    OutlierDetector,
    OutlierMethod,
    OutlierResult,
)


class TestOutlierMethod:
    """Tests for OutlierMethod enum."""

    def test_all_methods(self) -> None:
        assert OutlierMethod.ZSCORE.value == "zscore"
        assert OutlierMethod.MODIFIED_ZSCORE.value == "modified_zscore"
        assert OutlierMethod.IQR.value == "iqr"
        assert OutlierMethod.GRUBBS.value == "grubbs"
        assert OutlierMethod.ISOLATION_FOREST.value == "isolation_forest"
        assert OutlierMethod.LOF.value == "lof"
        assert OutlierMethod.DBSCAN.value == "dbscan"
        assert OutlierMethod.MAHALANOBIS.value == "mahalanobis"
        assert OutlierMethod.ENSEMBLE.value == "ensemble"


class TestOutlierConfig:
    """Tests for OutlierConfig."""

    def test_default_config(self) -> None:
        config = OutlierConfig()
        assert config.method == OutlierMethod.ENSEMBLE
        assert config.zscore_threshold == 3.0

    def test_custom_threshold(self) -> None:
        config = OutlierConfig(zscore_threshold=2.5)
        assert config.zscore_threshold == 2.5

    def test_custom_method(self) -> None:
        config = OutlierConfig(method=OutlierMethod.IQR)
        assert config.method == OutlierMethod.IQR


class TestOutlierDetector:
    """Tests for OutlierDetector."""

    @pytest.fixture()
    def clean_data(self) -> np.ndarray:
        """Generate clean normal data with obvious outliers."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        # Inject extreme outliers
        data = np.append(data, [50.0, -50.0, 100.0])
        return data

    def test_zscore_detects_outliers(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.ZSCORE, zscore_threshold=3.0)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert isinstance(result, OutlierResult)
        assert result.n_outliers >= 3  # at least our injected ones
        assert result.outlier_fraction > 0

    def test_modified_zscore(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.MODIFIED_ZSCORE)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert result.n_outliers >= 3

    def test_iqr_method(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.IQR)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert result.n_outliers >= 3

    def test_grubbs_method(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.GRUBBS)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert result.n_outliers >= 1

    def test_isolation_forest(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.ISOLATION_FOREST)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert isinstance(result, OutlierResult)

    def test_result_has_mask(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.ZSCORE)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert result.outlier_mask.shape == clean_data.shape
        assert result.outlier_mask.dtype == bool

    def test_result_has_scores(self, clean_data: np.ndarray) -> None:
        config = OutlierConfig(method=OutlierMethod.ZSCORE)
        detector = OutlierDetector(config)
        result = detector.detect(clean_data)
        assert len(result.scores) == len(clean_data)

    def test_no_outliers_in_uniform_data(self) -> None:
        data = np.ones(50)
        config = OutlierConfig(method=OutlierMethod.ZSCORE)
        detector = OutlierDetector(config)
        result = detector.detect(data)
        assert result.n_outliers == 0

    def test_default_config_works(self, clean_data: np.ndarray) -> None:
        detector = OutlierDetector()
        result = detector.detect(clean_data)
        assert isinstance(result, OutlierResult)

    def test_2d_data(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 3))
        data = np.vstack([data, [[50, 50, 50]]])
        config = OutlierConfig(method=OutlierMethod.ZSCORE)
        detector = OutlierDetector(config)
        result = detector.detect(data)
        assert isinstance(result, OutlierResult)
