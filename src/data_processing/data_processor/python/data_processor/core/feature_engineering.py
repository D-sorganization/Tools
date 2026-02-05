"""Feature Engineering Automation Module.

Provides automated feature engineering capabilities for time series
and tabular data, including feature generation, selection, and transformation.

Features:
- Automatic feature generation from time series
- Statistical features (mean, std, skewness, kurtosis, etc.)
- Time domain features (trends, peaks, crossings)
- Frequency domain features (spectral features)
- Rolling window features
- Lag features
- Feature selection (correlation, mutual information, importance)
- Feature transformation (polynomial, log, binning)
- Feature pipeline management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


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


class FeatureExtractor:
    """Extracts features from time series and tabular data.

    Provides comprehensive feature extraction including statistical,
    time domain, frequency domain, and derived features.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the extractor.

        Args:
            config: Configuration options
        """
        self.config = config or FeatureConfig()

    def extract_all(
        self,
        data: np.ndarray,
        column_names: list[str] | None = None,
    ) -> FeatureResult:
        """Extract all configured features.

        Args:
            data: Input data (n_samples, n_features) or (n_samples, seq_len, n_features)
            column_names: Optional names for input columns

        Returns:
            FeatureResult with extracted features
        """
        data = np.asarray(data, dtype=np.float64)

        # Ensure 2D or 3D
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]

        # Generate column names if not provided
        if column_names is None:
            if data.ndim == 2:
                column_names = [f"col_{i}" for i in range(data.shape[1])]
            else:
                column_names = [f"col_{i}" for i in range(data.shape[2])]

        all_features = []
        all_names = []
        categories: dict[str, list[str]] = {}

        # Check if data is time series (3D) or tabular (2D)
        is_time_series = data.ndim == 3

        if is_time_series:
            # Extract time series features for each sample
            for sample_idx in range(n_samples):
                sample_features = []
                sample_names = []

                for col_idx, col_name in enumerate(column_names):
                    series = data[sample_idx, :, col_idx]

                    # Statistical features
                    stat_feats, stat_names = self._extract_statistical(series, col_name)
                    sample_features.extend(stat_feats)
                    sample_names.extend(stat_names)

                    # Time domain features
                    if self.config.compute_trend or self.config.compute_peaks:
                        time_feats, time_names = self._extract_time_domain(
                            series, col_name
                        )
                        sample_features.extend(time_feats)
                        sample_names.extend(time_names)

                    # Frequency domain features
                    if self.config.compute_spectral:
                        freq_feats, freq_names = self._extract_frequency_domain(
                            series, col_name
                        )
                        sample_features.extend(freq_feats)
                        sample_names.extend(freq_names)

                all_features.append(sample_features)

                if sample_idx == 0:
                    all_names = sample_names

            features_array = np.array(all_features)

        else:
            # Tabular data - extract features per column
            features_list = []

            for col_idx, col_name in enumerate(column_names):
                column = data[:, col_idx]

                # Basic statistical features for tabular
                stat_feats, stat_names = self._extract_column_features(column, col_name)
                features_list.append(stat_feats)
                all_names.extend(stat_names)

            features_array = (
                np.column_stack(features_list) if features_list else np.array([])
            )

        # Categorize features
        for name in all_names:
            category = self._categorize_feature(name)
            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        # Compute feature statistics
        feature_stats = self._compute_feature_stats(features_array, all_names)

        return FeatureResult(
            features=features_array,
            feature_names=all_names,
            n_samples=n_samples,
            n_features=len(all_names),
            categories=categories,
            feature_stats=feature_stats,
        )

    def extract_statistical(
        self, data: np.ndarray, prefix: str = ""
    ) -> tuple[np.ndarray, list[str]]:
        """Extract statistical features from data.

        Args:
            data: 1D array
            prefix: Prefix for feature names

        Returns:
            Tuple of (features, names)
        """
        features, names = self._extract_statistical(data, prefix)
        return np.array(features), names

    def extract_rolling(
        self,
        data: np.ndarray,
        windows: list[int] | None = None,
        prefix: str = "",
    ) -> tuple[np.ndarray, list[str]]:
        """Extract rolling window features.

        Args:
            data: 1D time series
            windows: Window sizes
            prefix: Prefix for feature names

        Returns:
            Tuple of (features, names)
        """
        windows = windows or self.config.rolling_windows
        features = []
        names = []

        for window in windows:
            # Rolling mean
            rolling_mean = self._rolling_stat(data, window, np.mean)
            features.append(rolling_mean[-1] if len(rolling_mean) > 0 else np.nan)
            names.append(f"{prefix}_rolling_mean_{window}")

            # Rolling std
            rolling_std = self._rolling_stat(data, window, np.std)
            features.append(rolling_std[-1] if len(rolling_std) > 0 else np.nan)
            names.append(f"{prefix}_rolling_std_{window}")

            # Rolling min
            rolling_min = self._rolling_stat(data, window, np.min)
            features.append(rolling_min[-1] if len(rolling_min) > 0 else np.nan)
            names.append(f"{prefix}_rolling_min_{window}")

            # Rolling max
            rolling_max = self._rolling_stat(data, window, np.max)
            features.append(rolling_max[-1] if len(rolling_max) > 0 else np.nan)
            names.append(f"{prefix}_rolling_max_{window}")

        return np.array(features), names

    def extract_lag(
        self,
        data: np.ndarray,
        lags: list[int] | None = None,
        prefix: str = "",
    ) -> tuple[np.ndarray, list[str]]:
        """Extract lag features.

        Args:
            data: 1D time series
            lags: Lag values
            prefix: Prefix for feature names

        Returns:
            Tuple of (features, names)
        """
        lags = lags or self.config.lag_values
        features = []
        names = []

        n = len(data)

        for lag in lags:
            if lag < n:
                # Lag value
                features.append(data[-lag - 1] if lag + 1 <= n else np.nan)
                names.append(f"{prefix}_lag_{lag}")

                # Difference from lag
                features.append(data[-1] - data[-lag - 1] if lag + 1 <= n else np.nan)
                names.append(f"{prefix}_diff_{lag}")

                # Ratio to lag
                if data[-lag - 1] != 0:
                    features.append(data[-1] / data[-lag - 1])
                else:
                    features.append(np.nan)
                names.append(f"{prefix}_ratio_{lag}")

        return np.array(features), names

    def create_polynomial_features(
        self,
        data: np.ndarray,
        degree: int | None = None,
        include_interactions: bool | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Create polynomial features.

        Args:
            data: 2D array (n_samples, n_features)
            degree: Polynomial degree
            include_interactions: Include interaction terms

        Returns:
            Tuple of (features, names)
        """
        degree = degree or self.config.polynomial_degree
        include_interactions = (
            include_interactions
            if include_interactions is not None
            else self.config.include_interactions
        )

        data = np.atleast_2d(data)
        n_samples, n_features = data.shape

        features_list = [data]
        names = [f"x{i}" for i in range(n_features)]

        # Add polynomial terms
        for d in range(2, degree + 1):
            for i in range(n_features):
                power_feature = data[:, i : i + 1] ** d
                features_list.append(power_feature)
                names.append(f"x{i}^{d}")

        # Add interactions if requested
        if include_interactions and n_features > 1:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = (data[:, i] * data[:, j]).reshape(-1, 1)
                    features_list.append(interaction)
                    names.append(f"x{i}*x{j}")

        return np.hstack(features_list), names

    # Private extraction methods

    def _extract_statistical(
        self, data: np.ndarray, prefix: str
    ) -> tuple[list[float], list[str]]:
        """Extract statistical features."""
        features = []
        names = []

        if self.config.compute_mean:
            features.append(float(np.nanmean(data)))
            names.append(f"{prefix}_mean")

        if self.config.compute_std:
            features.append(float(np.nanstd(data)))
            names.append(f"{prefix}_std")

        if self.config.compute_min:
            features.append(float(np.nanmin(data)))
            names.append(f"{prefix}_min")

        if self.config.compute_max:
            features.append(float(np.nanmax(data)))
            names.append(f"{prefix}_max")

        if self.config.compute_median:
            features.append(float(np.nanmedian(data)))
            names.append(f"{prefix}_median")

        if self.config.compute_skewness:
            features.append(float(self._skewness(data)))
            names.append(f"{prefix}_skewness")

        if self.config.compute_kurtosis:
            features.append(float(self._kurtosis(data)))
            names.append(f"{prefix}_kurtosis")

        for p in self.config.compute_percentiles:
            features.append(float(np.nanpercentile(data, p)))
            names.append(f"{prefix}_p{p}")

        # Additional statistical features
        features.append(float(np.nanmax(data) - np.nanmin(data)))
        names.append(f"{prefix}_range")

        iqr = np.nanpercentile(data, 75) - np.nanpercentile(data, 25)
        features.append(float(iqr))
        names.append(f"{prefix}_iqr")

        # Coefficient of variation
        mean = np.nanmean(data)
        std = np.nanstd(data)
        cv = std / abs(mean) if mean != 0 else 0
        features.append(float(cv))
        names.append(f"{prefix}_cv")

        # Count of values above/below mean
        above_mean = np.sum(data > mean)
        features.append(float(above_mean / len(data)))
        names.append(f"{prefix}_above_mean_ratio")

        return features, names

    def _extract_time_domain(
        self, data: np.ndarray, prefix: str
    ) -> tuple[list[float], list[str]]:
        """Extract time domain features."""
        features = []
        names = []

        n = len(data)

        if self.config.compute_trend:
            # Linear trend slope
            x = np.arange(n)
            if n > 1:
                slope, _ = np.polyfit(x, data, 1)
            else:
                slope = 0
            features.append(float(slope))
            names.append(f"{prefix}_trend_slope")

            # Trend strength (R-squared of linear fit)
            if n > 1:
                trend = slope * x + np.mean(data)
                ss_res = np.sum((data - trend) ** 2)
                ss_tot = np.sum((data - np.mean(data)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            else:
                r_squared = 0
            features.append(float(r_squared))
            names.append(f"{prefix}_trend_strength")

        if self.config.compute_peaks:
            # Number of peaks
            peaks = self._count_peaks(data)
            features.append(float(peaks / n))
            names.append(f"{prefix}_peak_rate")

            # Peak prominence (average)
            prominences = self._peak_prominences(data)
            features.append(float(np.mean(prominences)) if prominences else 0.0)
            names.append(f"{prefix}_peak_prominence")

        if self.config.compute_crossings:
            # Zero crossings
            zero_crossings = self._count_zero_crossings(data - np.mean(data))
            features.append(float(zero_crossings / n))
            names.append(f"{prefix}_zero_crossing_rate")

            # Mean crossings
            mean_crossings = self._count_zero_crossings(data - np.mean(data))
            features.append(float(mean_crossings / n))
            names.append(f"{prefix}_mean_crossing_rate")

        if self.config.compute_autocorr:
            # Autocorrelation at lag 1
            if n > 1:
                acf1 = self._autocorrelation(data, 1)
            else:
                acf1 = 0
            features.append(float(acf1))
            names.append(f"{prefix}_autocorr_lag1")

            # Partial autocorrelation at lag 1
            features.append(float(acf1))  # Simplified: same as ACF for lag 1
            names.append(f"{prefix}_pacf_lag1")

        # First and last values
        features.append(float(data[0]))
        names.append(f"{prefix}_first")

        features.append(float(data[-1]))
        names.append(f"{prefix}_last")

        # Change from first to last
        features.append(float(data[-1] - data[0]))
        names.append(f"{prefix}_total_change")

        return features, names

    def _extract_frequency_domain(
        self, data: np.ndarray, prefix: str
    ) -> tuple[list[float], list[str]]:
        """Extract frequency domain features."""
        features = []
        names = []

        n = len(data)

        # FFT
        fft = np.fft.rfft(data)
        power_spectrum = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # Spectral centroid
        if np.sum(power_spectrum) > 0:
            centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        else:
            centroid = 0
        features.append(float(centroid))
        names.append(f"{prefix}_spectral_centroid")

        # Spectral spread
        if np.sum(power_spectrum) > 0:
            spread = np.sqrt(
                np.sum((freqs - centroid) ** 2 * power_spectrum)
                / np.sum(power_spectrum)
            )
        else:
            spread = 0
        features.append(float(spread))
        names.append(f"{prefix}_spectral_spread")

        # Spectral rolloff
        cumsum = np.cumsum(power_spectrum)
        total = cumsum[-1]
        if total > 0:
            rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
            rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        else:
            rolloff = 0
        features.append(float(rolloff))
        names.append(f"{prefix}_spectral_rolloff")

        # Spectral entropy
        if np.sum(power_spectrum) > 0:
            ps_norm = power_spectrum / np.sum(power_spectrum)
            ps_norm = ps_norm[ps_norm > 0]
            entropy = -np.sum(ps_norm * np.log2(ps_norm))
        else:
            entropy = 0
        features.append(float(entropy))
        names.append(f"{prefix}_spectral_entropy")

        # Dominant frequency
        if len(power_spectrum) > 0:
            dominant_idx = np.argmax(power_spectrum)
            dominant_freq = freqs[dominant_idx]
        else:
            dominant_freq = 0
        features.append(float(dominant_freq))
        names.append(f"{prefix}_dominant_freq")

        # Power at spectral percentiles
        for p in self.config.spectral_percentiles:
            power_at_p = np.percentile(power_spectrum, p)
            features.append(float(power_at_p))
            names.append(f"{prefix}_power_p{p}")

        return features, names

    def _extract_column_features(
        self, column: np.ndarray, prefix: str
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from a single column (tabular data)."""
        features = []
        names = []

        # Basic stats
        stat_feats, stat_names = self._extract_statistical(column, prefix)
        features.extend(stat_feats)
        names.extend(stat_names)

        # Histogram-based features
        hist, _ = np.histogram(column, bins=self.config.n_bins, density=True)
        for i, h in enumerate(hist):
            features.append(float(h))
            names.append(f"{prefix}_hist_bin{i}")

        return np.array(features), names

    # Helper methods

    def _rolling_stat(
        self, data: np.ndarray, window: int, func: Callable
    ) -> np.ndarray:
        """Compute rolling statistic."""
        n = len(data)
        if window > n:
            window = n

        result = np.zeros(n - window + 1)
        for i in range(len(result)):
            result[i] = func(data[i : i + window])

        return result

    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        if n < 3:
            return 0.0

        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std == 0:
            return 0.0

        return float(np.nanmean(((data - mean) / std) ** 3))

    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0

        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std == 0:
            return 0.0

        return float(np.nanmean(((data - mean) / std) ** 4) - 3)

    def _count_peaks(self, data: np.ndarray) -> int:
        """Count number of peaks."""
        n = len(data)
        if n < 3:
            return 0

        peaks = 0
        for i in range(1, n - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks += 1

        return peaks

    def _peak_prominences(self, data: np.ndarray) -> list[float]:
        """Get peak prominences."""
        n = len(data)
        if n < 3:
            return []

        prominences = []
        for i in range(1, n - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                # Simple prominence: height above neighbors
                prominence = data[i] - max(data[i - 1], data[i + 1])
                prominences.append(prominence)

        return prominences

    def _count_zero_crossings(self, data: np.ndarray) -> int:
        """Count zero crossings."""
        return int(np.sum(np.abs(np.diff(np.sign(data))) > 0))

    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        n = len(data)
        if lag >= n:
            return 0.0

        data_centered = data - np.mean(data)
        var = np.var(data_centered)

        if var == 0:
            return 0.0

        return float(
            np.sum(data_centered[lag:] * data_centered[: n - lag]) / ((n - lag) * var)
        )

    def _categorize_feature(self, name: str) -> str:
        """Categorize a feature by its name."""
        if any(
            x in name
            for x in [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "skewness",
                "kurtosis",
                "range",
                "iqr",
                "cv",
                "p25",
                "p50",
                "p75",
            ]
        ):
            return "statistical"
        elif any(
            x in name
            for x in [
                "trend",
                "peak",
                "crossing",
                "autocorr",
                "first",
                "last",
                "change",
            ]
        ):
            return "time_domain"
        elif any(x in name for x in ["spectral", "freq", "power"]):
            return "frequency_domain"
        elif "rolling" in name:
            return "rolling"
        elif any(x in name for x in ["lag", "diff", "ratio"]):
            return "lag"
        elif any(x in name for x in ["^", "*"]):
            return "polynomial"
        elif "hist" in name:
            return "histogram"
        else:
            return "other"

    def _compute_feature_stats(
        self, features: np.ndarray, names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Compute statistics for each feature."""
        stats = {}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        for i, name in enumerate(names):
            col = features[:, i] if i < features.shape[1] else np.array([])
            stats[name] = {
                "mean": float(np.nanmean(col)) if len(col) > 0 else 0.0,
                "std": float(np.nanstd(col)) if len(col) > 0 else 0.0,
                "min": float(np.nanmin(col)) if len(col) > 0 else 0.0,
                "max": float(np.nanmax(col)) if len(col) > 0 else 0.0,
                "missing_ratio": float(np.mean(np.isnan(col))) if len(col) > 0 else 1.0,
            }

        return stats


class FeatureSelector:
    """Selects relevant features based on various criteria."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        """Initialize the selector.

        Args:
            config: Configuration options
        """
        self.config = config or FeatureConfig()

    def select_by_correlation(
        self,
        features: np.ndarray,
        feature_names: list[str],
        target: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> SelectionResult:
        """Select features by removing highly correlated ones.

        Args:
            features: Feature matrix (n_samples, n_features)
            feature_names: Feature names
            target: Optional target variable
            threshold: Correlation threshold

        Returns:
            SelectionResult with selected features
        """
        threshold = threshold or self.config.correlation_threshold
        features = np.atleast_2d(features)
        n_features = features.shape[1]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(features, rowvar=False)

        # Find highly correlated pairs
        to_remove = set()
        scores = {}

        for i in range(n_features):
            if i in to_remove:
                continue

            for j in range(i + 1, n_features):
                if j in to_remove:
                    continue

                if abs(corr_matrix[i, j]) > threshold:
                    # If target is provided, keep the one with higher correlation to target
                    if target is not None:
                        corr_i = abs(np.corrcoef(features[:, i], target)[0, 1])
                        corr_j = abs(np.corrcoef(features[:, j], target)[0, 1])
                        to_remove.add(j if corr_i >= corr_j else i)
                    else:
                        # Remove the one with higher average correlation
                        avg_corr_i = np.mean(np.abs(corr_matrix[i, :]))
                        avg_corr_j = np.mean(np.abs(corr_matrix[j, :]))
                        to_remove.add(j if avg_corr_i <= avg_corr_j else i)

        selected_indices = np.array(
            [i for i in range(n_features) if i not in to_remove]
        )
        selected_names = [feature_names[i] for i in selected_indices]
        removed_names = [feature_names[i] for i in to_remove]

        # Compute scores (1 - max correlation with other features)
        for i, name in enumerate(feature_names):
            other_corrs = [abs(corr_matrix[i, j]) for j in range(n_features) if j != i]
            scores[name] = 1 - max(other_corrs) if other_corrs else 1.0

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=scores,
            threshold_used=threshold,
            n_original=n_features,
            n_selected=len(selected_indices),
        )

    def select_by_variance(
        self,
        features: np.ndarray,
        feature_names: list[str],
        threshold: float | None = None,
    ) -> SelectionResult:
        """Select features by variance threshold.

        Args:
            features: Feature matrix
            feature_names: Feature names
            threshold: Minimum variance

        Returns:
            SelectionResult with selected features
        """
        threshold = threshold or self.config.variance_threshold
        features = np.atleast_2d(features)

        # Compute variances
        variances = np.var(features, axis=0)

        # Select features above threshold
        mask = variances >= threshold
        selected_indices = np.where(mask)[0]
        selected_names = [feature_names[i] for i in selected_indices]
        removed_names = [feature_names[i] for i, m in enumerate(mask) if not m]

        scores = {
            feature_names[i]: float(variances[i]) for i in range(len(feature_names))
        }

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=scores,
            threshold_used=threshold,
            n_original=len(feature_names),
            n_selected=len(selected_indices),
        )

    def select_by_mutual_info(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: list[str],
        k: int | None = None,
    ) -> SelectionResult:
        """Select top-k features by mutual information.

        Args:
            features: Feature matrix
            target: Target variable
            feature_names: Feature names
            k: Number of features to select

        Returns:
            SelectionResult with selected features
        """
        features = np.atleast_2d(features)
        n_features = features.shape[1]
        k = k or n_features // 2

        # Compute mutual information (simplified)
        mi_scores = {}
        for i in range(n_features):
            mi = self._mutual_information(features[:, i], target)
            mi_scores[feature_names[i]] = float(mi)

        # Sort and select top-k
        sorted_features = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        selected_names = [name for name, _ in sorted_features[:k]]
        selected_indices = np.array(
            [feature_names.index(name) for name in selected_names]
        )
        removed_names = [name for name, _ in sorted_features[k:]]

        return SelectionResult(
            selected_indices=selected_indices,
            selected_names=selected_names,
            removed_names=removed_names,
            scores=mi_scores,
            threshold_used=float(k),
            n_original=n_features,
            n_selected=k,
        )

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information (simplified binning approach)."""
        n_bins = self.config.n_bins

        # Discretize
        x_bins = np.digitize(x, np.linspace(np.min(x), np.max(x), n_bins + 1)[1:-1])
        y_bins = np.digitize(y, np.linspace(np.min(y), np.max(y), n_bins + 1)[1:-1])

        # Joint and marginal distributions
        joint_hist = np.histogram2d(x_bins, y_bins, bins=n_bins)[0]
        joint_prob = joint_hist / np.sum(joint_hist)

        x_prob = np.sum(joint_prob, axis=1)
        y_prob = np.sum(joint_prob, axis=0)

        # Mutual information
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (x_prob[i] * y_prob[j])
                    )

        return max(0, mi)


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
        features = np.atleast_2d(features)

        if method == TransformationType.STANDARDIZE:
            mean = self._fit_params.get("standardize_mean", np.mean(features, axis=0))
            std = self._fit_params.get("standardize_std", np.std(features, axis=0))
            std[std == 0] = 1
            return (features - mean) / std

        elif method == TransformationType.NORMALIZE:
            min_val = self._fit_params.get("normalize_min", np.min(features, axis=0))
            range_val = self._fit_params.get(
                "normalize_range", np.max(features, axis=0) - min_val
            )
            range_val[range_val == 0] = 1
            return (features - min_val) / range_val

        else:
            return self.fit_transform(features, method)


def extract_features(
    data: np.ndarray,
    column_names: list[str] | None = None,
) -> FeatureResult:
    """Convenience function for feature extraction.

    Args:
        data: Input data
        column_names: Optional column names

    Returns:
        FeatureResult with extracted features

    Example:
        >>> data = np.random.randn(100, 50, 3)  # 100 samples, 50 timesteps, 3 channels
        >>> result = extract_features(data, ['x', 'y', 'z'])
        >>> print(f"Extracted {result.n_features} features")
    """
    extractor = FeatureExtractor()
    return extractor.extract_all(data, column_names)


def select_features(
    features: np.ndarray,
    feature_names: list[str],
    target: np.ndarray | None = None,
    method: str = "correlation",
) -> SelectionResult:
    """Convenience function for feature selection.

    Args:
        features: Feature matrix
        feature_names: Feature names
        target: Optional target variable
        method: Selection method

    Returns:
        SelectionResult with selected features
    """
    selector = FeatureSelector()

    if method == "correlation":
        return selector.select_by_correlation(features, feature_names, target)
    elif method == "variance":
        return selector.select_by_variance(features, feature_names)
    elif method == "mutual_info" and target is not None:
        return selector.select_by_mutual_info(features, target, feature_names)
    else:
        return selector.select_by_correlation(features, feature_names, target)


__all__ = [
    "FeatureCategory",
    "SelectionMethod",
    "TransformationType",
    "FeatureConfig",
    "FeatureResult",
    "SelectionResult",
    "FeatureExtractor",
    "FeatureSelector",
    "FeatureTransformer",
    "extract_features",
    "select_features",
]
