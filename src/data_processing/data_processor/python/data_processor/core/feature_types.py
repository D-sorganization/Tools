"""Feature engineering types and configuration.

Shared enums, dataclasses, and configuration for feature engineering
submodules (extraction, selection, transformation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FeatureCategory(Enum):
    """Categories of features."""

    STATISTICAL = "statistical"
    TIME_DOMAIN = "time_domain"
    FREQUENCY_DOMAIN = "frequency_domain"
    ROLLING = "rolling"
    LAG = "lag"
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"
    CUSTOM = "custom"


class SelectionMethod(Enum):
    """Feature selection methods."""

    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    VARIANCE = "variance"
    IMPORTANCE = "importance"
    RECURSIVE = "recursive"


class TransformationType(Enum):
    """Feature transformation types."""

    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    RECIPROCAL = "reciprocal"
    STANDARDIZE = "standardize"
    NORMALIZE = "normalize"
    QUANTILE = "quantile"
    BINNING = "binning"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Statistical features
    compute_mean: bool = True
    compute_std: bool = True
    compute_min: bool = True
    compute_max: bool = True
    compute_median: bool = True
    compute_skewness: bool = True
    compute_kurtosis: bool = True
    compute_percentiles: list[int] = field(default_factory=lambda: [25, 75])

    # Time domain features
    compute_trend: bool = True
    compute_peaks: bool = True
    compute_crossings: bool = True
    compute_autocorr: bool = True

    # Frequency domain features
    compute_spectral: bool = True
    spectral_percentiles: list[int] = field(default_factory=lambda: [25, 50, 75, 95])

    # Rolling features
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])

    # Lag features
    lag_values: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])

    # Polynomial features
    polynomial_degree: int = 2
    include_interactions: bool = True

    # Selection
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    importance_threshold: float = 0.01

    # Binning
    n_bins: int = 10


@dataclass
class FeatureResult:
    """Result of feature extraction."""

    # Features
    features: np.ndarray
    feature_names: list[str]

    # Metadata
    n_samples: int
    n_features: int
    categories: dict[str, list[str]]

    # Statistics
    feature_stats: dict[str, dict[str, float]]


@dataclass
class SelectionResult:
    """Result of feature selection."""

    # Selected features
    selected_indices: np.ndarray
    selected_names: list[str]
    removed_names: list[str]

    # Selection scores
    scores: dict[str, float]
    threshold_used: float

    # Original feature count
    n_original: int
    n_selected: int


__all__ = [
    "FeatureCategory",
    "SelectionMethod",
    "TransformationType",
    "FeatureConfig",
    "FeatureResult",
    "SelectionResult",
]
