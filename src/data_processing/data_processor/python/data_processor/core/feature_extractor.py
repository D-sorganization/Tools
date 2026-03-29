# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Feature extraction from time series and tabular data.

Provides comprehensive feature extraction including statistical,
time domain, frequency domain, rolling window, lag, and polynomial features.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from numba import jit

from .feature_types import FeatureConfig, FeatureResult

logger = logging.getLogger(__name__)


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

    def _extract_time_series_features(
        self,
        data: np.ndarray,
        column_names: list[str],
        n_samples: int,
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from 3D time series data.

        Args:
            data: 3D array (n_samples, seq_len, n_features)
            column_names: Column names
            n_samples: Number of samples

        Returns:
            Tuple of (features_array, feature_names)
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        all_features = []
        all_names: list[str] = []

        for sample_idx in range(n_samples):
            sample_features: list[float] = []
            sample_names: list[str] = []

            for col_idx, col_name in enumerate(column_names):
                series = data[sample_idx, :, col_idx]

                stat_feats, stat_names = self._extract_statistical(series, col_name)
                sample_features.extend(stat_feats)
                sample_names.extend(stat_names)

                if self.config.compute_trend or self.config.compute_peaks:
                    time_feats, time_names = self._extract_time_domain(series, col_name)
                    sample_features.extend(time_feats)
                    sample_names.extend(time_names)

                if self.config.compute_spectral:
                    freq_feats, freq_names = self._extract_frequency_domain(
                        series, col_name
                    )
                    sample_features.extend(freq_feats)
                    sample_names.extend(freq_names)

            all_features.append(sample_features)
            if sample_idx == 0:
                all_names = sample_names

        return np.array(all_features), all_names

    def _extract_tabular_features(
        self,
        data: np.ndarray,
        column_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Extract features from 2D tabular data.

        Args:
            data: 2D array (n_samples, n_features)
            column_names: Column names

        Returns:
            Tuple of (features_array, feature_names)
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        features_list = []
        all_names: list[str] = []

        for col_idx, col_name in enumerate(column_names):
            column = data[:, col_idx]
            stat_feats, stat_names = self._extract_column_features(column, col_name)
            features_list.append(stat_feats)
            all_names.extend(stat_names)

        features_array = (
            np.column_stack(features_list) if features_list else np.array([])
        )
        return features_array, all_names

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
        if not (data is not None):
            raise ValueError("data must be provided")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_samples = data.shape[0]

        if column_names is None:
            dim = data.shape[2] if data.ndim == 3 else data.shape[1]
            column_names = [f"col_{i}" for i in range(dim)]

        if data.ndim == 3:
            features_array, all_names = self._extract_time_series_features(
                data,
                column_names,
                n_samples,
            )
        else:
            features_array, all_names = self._extract_tabular_features(
                data,
                column_names,
            )

        categories: dict[str, list[str]] = {}
        for name in all_names:
            category = self._categorize_feature(name)
            if category not in categories:
                categories[category] = []
            categories[category].append(name)

        return FeatureResult(
            features=features_array,
            feature_names=all_names,
            n_samples=n_samples,
            n_features=len(all_names),
            categories=categories,
            feature_stats=self._compute_feature_stats(features_array, all_names),
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
        if not (data is not None):
            raise ValueError("data must be provided")
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
        if not (data is not None):
            raise ValueError("data must be provided")
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

    @jit(nopython=True, fastmath=True)
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
        if not (data is not None):
            raise ValueError("data must be provided")
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

    @jit(nopython=True, fastmath=True)
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
        if not (data is not None):
            raise ValueError("data must be provided")
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
        if not (data is not None):
            raise ValueError("data must be provided")
        features: list[float] = []
        names: list[str] = []

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
        if not (data is not None):
            raise ValueError("data must be provided")
        features: list[float] = []
        names: list[str] = []

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
            # Zero crossings (relative to absolute zero)
            zero_crossings = self._count_zero_crossings(data)
            features.append(float(zero_crossings / n))
            names.append(f"{prefix}_zero_crossing_rate")

            # Mean crossings (relative to the mean)
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
        if not (data is not None):
            raise ValueError("data must be provided")
        features: list[float] = []
        names: list[str] = []

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
        if not (column is not None):
            raise ValueError("column must be provided")
        features: list[float] = []
        names: list[str] = []

        # Basic stats
        stat_feats, stat_names = self._extract_statistical(column, prefix)
        features.extend(stat_feats)
        names.extend(stat_names)

        # Histogram-based features
        hist, _ = np.histogram(column, bins=self.config.n_bins, density=True)
        features.extend(float(h) for h in hist)
        names.extend(f"{prefix}_hist_bin{i}" for i in range(len(hist)))

        return np.array(features), names

    # Helper methods

    def _rolling_stat(
        self, data: np.ndarray, window: int, func: Callable
    ) -> np.ndarray:
        """Compute rolling statistic."""
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)
        if window > n:
            window = n

        result = np.zeros(n - window + 1)
        for i in range(len(result)):
            result[i] = func(data[i : i + window])

        return result

    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        if not (data is not None):
            raise ValueError("data must be provided")
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
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)
        if n < 4:
            return 0.0

        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std == 0:
            return 0.0

        return float(np.nanmean(((data - mean) / std) ** 4) - 3)

    @jit(nopython=True, fastmath=True)
    def _count_peaks(self, data: np.ndarray) -> int:
        """Count number of peaks."""
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)
        if n < 3:
            return 0

        peaks = 0
        for i in range(1, n - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks += 1

        return peaks

    @jit(nopython=True, fastmath=True)
    def _peak_prominences(self, data: np.ndarray) -> list[float]:
        """Get peak prominences."""
        if not (data is not None):
            raise ValueError("data must be provided")
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
        if not (data is not None):
            raise ValueError("data must be provided")
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
        if not (features is not None):
            raise ValueError("features must be provided")
        stats: dict[str, dict[str, float]] = {}

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
    if not (data is not None):
        raise ValueError("data must be provided")
    extractor = FeatureExtractor()
    return extractor.extract_all(data, column_names)


__all__ = [
    "FeatureExtractor",
    "extract_features",
]
