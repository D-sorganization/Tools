"""Feature transformation using various mathematical methods.

Provides methods to transform feature matrices including log, sqrt, square,
standardization, normalization, quantile, and binning transformations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .feature_types import FeatureConfig, TransformationType

logger = logging.getLogger(__name__)


class FeatureTransformer:
    """Transforms features using various methods."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the transformer.

        Args:
            config: Configuration options
        """
        self.config = config or FeatureConfig()
        self._fit_params: dict[str, Any] = {}

    def fit_transform(
        self,
        features: np.ndarray,
        method: TransformationType,
    ) -> np.ndarray:
        """Fit and transform features.

        Args:
            features: Feature matrix
            method: Transformation method

        Returns:
            Transformed features
        """
        if not (features is not None):
            raise ValueError("features must be provided")
        features = np.atleast_2d(features)

        if method == TransformationType.LOG:
            # Log transform (handle zeros and negatives)
            min_val = np.min(features)
            shift = abs(min_val) + 1 if min_val <= 0 else 0
            self._fit_params["log_shift"] = shift
            return np.log(features + shift)

        elif method == TransformationType.SQRT:
            min_val = np.min(features)
            shift = abs(min_val) if min_val < 0 else 0
            self._fit_params["sqrt_shift"] = shift
            return np.sqrt(features + shift)

        elif method == TransformationType.SQUARE:
            return features**2

        elif method == TransformationType.RECIPROCAL:
            # Avoid division by zero
            return 1 / (features + 1e-10)

        elif method == TransformationType.STANDARDIZE:
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            self._fit_params["standardize_mean"] = mean
            self._fit_params["standardize_std"] = std
            return (features - mean) / std

        elif method == TransformationType.NORMALIZE:
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            self._fit_params["normalize_min"] = min_val
            self._fit_params["normalize_range"] = range_val
            return (features - min_val) / range_val

        elif method == TransformationType.QUANTILE:
            # Quantile transform to uniform distribution
            result = np.zeros_like(features)
            for i in range(features.shape[1]):
                sorted_idx = np.argsort(features[:, i])
                ranks = np.zeros(len(sorted_idx))
                ranks[sorted_idx] = np.arange(len(sorted_idx))
                result[:, i] = ranks / (len(ranks) - 1)
            return result

        elif method == TransformationType.BINNING:
            # Equal-width binning
            n_bins = self.config.n_bins
            result = np.zeros_like(features)
            for i in range(features.shape[1]):
                min_val = np.min(features[:, i])
                max_val = np.max(features[:, i])
                bins = np.linspace(min_val, max_val, n_bins + 1)
                result[:, i] = np.digitize(features[:, i], bins[1:-1])
            return result

        else:
            return features

    def transform(
        self,
        features: np.ndarray,
        method: TransformationType,
    ) -> np.ndarray:
        """Transform features using fitted parameters.

        Args:
            features: Feature matrix
            method: Transformation method

        Returns:
            Transformed features
        """
        if not (features is not None):
            raise ValueError("features must be provided")
        features = np.atleast_2d(features)

        if method == TransformationType.STANDARDIZE:
            mean = self._fit_params.get("standardize_mean", np.mean(features, axis=0))
            std = self._fit_params.get("standardize_std", np.std(features, axis=0))
            std[std == 0] = 1
            return (features - mean) / std

        elif method == TransformationType.NORMALIZE:
            min_val = self._fit_params.get("normalize_min", np.min(features, axis=0))
            range_val = self._fit_params.get("normalize_range", np.max(features, axis=0) - min_val)
            range_val[range_val == 0] = 1
            return (features - min_val) / range_val

        else:
            return self.fit_transform(features, method)


__all__ = [
    "FeatureTransformer",
]
