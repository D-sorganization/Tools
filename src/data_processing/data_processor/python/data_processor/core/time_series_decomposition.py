"""Time-Series Decomposition Module.

Provides comprehensive time-series decomposition capabilities including
STL (Seasonal and Trend decomposition using Loess), classical decomposition,
and advanced methods for extracting trend, seasonal, and residual components.

Features:
- STL decomposition with configurable parameters
- Classical additive and multiplicative decomposition
- Multiple seasonality detection
- Trend extraction using various methods
- Anomaly detection in residuals
- Forecasting support using decomposed components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DecompositionMethod(Enum):
    """Available decomposition methods."""

    STL = "stl"
    CLASSICAL_ADDITIVE = "classical_additive"
    CLASSICAL_MULTIPLICATIVE = "classical_multiplicative"
    MOVING_AVERAGE = "moving_average"
    LOWESS = "lowess"
    HP_FILTER = "hp_filter"  # Hodrick-Prescott filter


class SeasonalModel(Enum):
    """Seasonal model types."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class TrendModel(Enum):
    """Trend estimation methods."""

    MOVING_AVERAGE = "moving_average"
    LOWESS = "lowess"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    HP_FILTER = "hp_filter"


@dataclass
class DecompositionConfig:
    """Configuration for time-series decomposition."""

    # General settings
    method: DecompositionMethod = DecompositionMethod.STL
    seasonal_model: SeasonalModel = SeasonalModel.ADDITIVE

    # Seasonality settings
    period: int | None = None  # Auto-detect if None
    seasonal_deg: int = 1  # Degree of LOESS for seasonal extraction
    seasonal_jump: int = 1  # LOESS jump for seasonal

    # Trend settings
    trend_deg: int = 1  # Degree of LOESS for trend extraction
    trend_jump: int = 1  # LOESS jump for trend
    low_pass_deg: int = 1  # Degree of LOESS for low-pass filter
    low_pass_jump: int = 1  # LOESS jump for low-pass

    # Robustness
    robust: bool = False  # Use robust fitting (iterative re-weighting)
    robust_iterations: int = 2  # Number of robustness iterations

    # HP filter
    hp_lambda: float = 1600.0  # Smoothing parameter for HP filter

    # Moving average
    ma_window: int | None = None  # Window size for moving average

    # Polynomial trend
    polynomial_degree: int = 2


@dataclass
class DecompositionResult:
    """Results from time-series decomposition."""

    # Original data
    observed: np.ndarray

    # Decomposed components
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray

    # Method information
    method: DecompositionMethod
    seasonal_model: SeasonalModel
    period: int

    # Quality metrics
    trend_strength: float = 0.0
    seasonal_strength: float = 0.0
    residual_variance: float = 0.0

    # Optional: multiple seasonalities
    seasonal_components: dict[int, np.ndarray] = field(default_factory=dict)

    # Statistics
    residual_mean: float = 0.0
    residual_std: float = 0.0
    residual_autocorrelation: float = 0.0

    def get_deseasonalized(self) -> np.ndarray:
        """Get data with seasonal component removed."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.observed - self.seasonal
        else:
            return self.observed / np.where(self.seasonal != 0, self.seasonal, 1.0)

    def get_detrended(self) -> np.ndarray:
        """Get data with trend removed."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.observed - self.trend
        else:
            return self.observed / np.where(self.trend != 0, self.trend, 1.0)

    def get_seasonally_adjusted(self) -> np.ndarray:
        """Get seasonally adjusted series (trend + residual)."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.trend + self.residual
        else:
            return self.trend * self.residual

    def reconstruct(self) -> np.ndarray:
        """Reconstruct original series from components."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.trend + self.seasonal + self.residual
        else:
            return self.trend * self.seasonal * self.residual


@dataclass
class SeasonalityDetectionResult:
    """Results from seasonality detection."""

    detected_periods: list[int]
    period_strengths: dict[int, float]
    dominant_period: int | None
    is_seasonal: bool
    autocorrelation_peaks: list[int]


class TimeSeriesDecomposer:
    """Comprehensive time-series decomposition engine.

    Supports multiple decomposition methods including STL,
    classical decomposition, and various trend extraction techniques.
    """

    def __init__(self, config: DecompositionConfig | None = None) -> None:
        """Initialize the decomposer.

        Args:
            config: Configuration options
        """
        self.config = config or DecompositionConfig()

    def decompose(
        self,
        data: np.ndarray,
        period: int | None = None,
        method: DecompositionMethod | None = None,
    ) -> DecompositionResult:
        """Decompose time series into trend, seasonal, and residual components.

        Args:
            data: 1D time series data
            period: Seasonal period (auto-detected if None)
            method: Decomposition method (uses config default if None)

        Returns:
            DecompositionResult with all components
        """
        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 1:
            raise ValueError("Data must be 1-dimensional")

        if len(data) < 4:
            raise ValueError("Data must have at least 4 observations")

        # Detect period if not provided
        if period is None:
            period = self.config.period
        if period is None:
            detection = self.detect_seasonality(data)
            period = detection.dominant_period or 1

        method = method or self.config.method

        # Route to appropriate method
        if method == DecompositionMethod.STL:
            result = self._stl_decompose(data, period)
        elif method == DecompositionMethod.CLASSICAL_ADDITIVE:
            result = self._classical_decompose(data, period, SeasonalModel.ADDITIVE)
        elif method == DecompositionMethod.CLASSICAL_MULTIPLICATIVE:
            result = self._classical_decompose(
                data, period, SeasonalModel.MULTIPLICATIVE
            )
        elif method == DecompositionMethod.MOVING_AVERAGE:
            result = self._ma_decompose(data, period)
        elif method == DecompositionMethod.LOWESS:
            result = self._lowess_decompose(data, period)
        elif method == DecompositionMethod.HP_FILTER:
            result = self._hp_filter_decompose(data, period)
        else:
            raise ValueError(f"Unknown decomposition method: {method}")

        # Calculate quality metrics
        result = self._calculate_metrics(result)

        return result

    def detect_seasonality(
        self,
        data: np.ndarray,
        max_period: int | None = None,
    ) -> SeasonalityDetectionResult:
        """Detect seasonal periods in the data.

        Args:
            data: 1D time series data
            max_period: Maximum period to check

        Returns:
            SeasonalityDetectionResult with detected periods
        """
        assert data is not None, "data must be provided"
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        if max_period is None:
            max_period = min(n // 2, 365)  # Reasonable default

        # Calculate autocorrelation
        acf = self._autocorrelation(data, max_period)

        # Find peaks in ACF
        peaks = self._find_acf_peaks(acf)

        # Calculate strength for each peak
        period_strengths = {}
        for peak in peaks:
            if peak > 0:
                period_strengths[peak] = float(acf[peak])

        # Determine dominant period
        # Prefer smaller periods if strengths are similar (e.g. 10 vs 20)
        dominant_period = None
        if period_strengths:
            max_strength = max(period_strengths.values())
            # Get all periods with strength at least 95% of max
            candidates = [
                p for p, s in period_strengths.items() if s >= 0.95 * max_strength
            ]
            dominant_period = min(candidates)

        # Determine if data is seasonal
        is_seasonal = bool(period_strengths) and max(period_strengths.values()) > 0.3

        return SeasonalityDetectionResult(
            detected_periods=list(period_strengths.keys()),
            period_strengths=period_strengths,
            dominant_period=dominant_period,
            is_seasonal=is_seasonal,
            autocorrelation_peaks=peaks,
        )

    def extract_trend(
        self,
        data: np.ndarray,
        method: TrendModel = TrendModel.MOVING_AVERAGE,
        **kwargs: Any,
    ) -> np.ndarray:
        """Extract trend component using specified method.

        Args:
            data: 1D time series data
            method: Trend extraction method
            **kwargs: Additional method-specific parameters

        Returns:
            Trend component array
        """
        assert data is not None, "data must be provided"
        data = np.asarray(data, dtype=np.float64)

        if method == TrendModel.MOVING_AVERAGE:
            window = kwargs.get("window", self.config.ma_window or len(data) // 10)
            return self._moving_average(data, window)

        elif method == TrendModel.LOWESS:
            frac = kwargs.get("frac", 0.3)
            return self._lowess_smooth(data, frac)

        elif method == TrendModel.POLYNOMIAL:
            degree = kwargs.get("degree", self.config.polynomial_degree)
            return self._polynomial_trend(data, degree)

        elif method == TrendModel.EXPONENTIAL:
            alpha = kwargs.get("alpha", 0.3)
            return self._exponential_smooth(data, alpha)

        elif method == TrendModel.HP_FILTER:
            lambd = kwargs.get("lambda", self.config.hp_lambda)
            return self._hp_filter(data, lambd)

        else:
            raise ValueError(f"Unknown trend method: {method}")

    def extract_seasonal(
        self,
        data: np.ndarray,
        period: int,
        method: str = "average",
    ) -> np.ndarray:
        """Extract seasonal component.

        Args:
            data: 1D time series data
            period: Seasonal period
            method: Extraction method ('average', 'median', 'robust')

        Returns:
            Seasonal component array
        """
        assert data is not None, "data must be provided"
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        # Calculate seasonal indices
        seasonal_indices = np.zeros(period)

        for i in range(period):
            indices = np.arange(i, n, period)
            values = data[indices]

            if method == "average":
                seasonal_indices[i] = np.nanmean(values)
            elif method == "median":
                seasonal_indices[i] = np.nanmedian(values)
            elif method == "robust":
                # Trimmed mean
                sorted_vals = np.sort(values)
                trim = len(sorted_vals) // 10
                if trim > 0:
                    seasonal_indices[i] = np.mean(sorted_vals[trim:-trim])
                else:
                    seasonal_indices[i] = np.mean(sorted_vals)

        # Normalize to zero mean for additive model
        seasonal_indices -= np.mean(seasonal_indices)

        # Tile to match data length
        seasonal = np.tile(seasonal_indices, n // period + 1)[:n]

        return seasonal

    def multi_seasonal_decompose(
        self,
        data: np.ndarray,
        periods: list[int],
    ) -> DecompositionResult:
        """Decompose with multiple seasonal components.

        Args:
            data: 1D time series data
            periods: List of seasonal periods

        Returns:
            DecompositionResult with multiple seasonal components
        """
        assert data is not None, "data must be provided"
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        # Sort periods from longest to shortest
        periods = sorted(periods, reverse=True)

        # Initialize
        residual = data.copy()
        seasonal_components: dict[int, np.ndarray] = {}
        combined_seasonal = np.zeros(n)

        # Extract each seasonal component
        for period in periods:
            # Extract trend from current residual
            trend = self._moving_average(residual, period)

            # Detrend
            detrended = residual - trend

            # Extract seasonal
            seasonal = self.extract_seasonal(detrended, period)
            seasonal_components[period] = seasonal
            combined_seasonal += seasonal

            # Remove this seasonal component
            residual = residual - seasonal

        # Final trend extraction
        trend = self._moving_average(residual, periods[-1])
        final_residual = residual - trend

        result = DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=combined_seasonal,
            residual=final_residual,
            method=DecompositionMethod.STL,
            seasonal_model=SeasonalModel.ADDITIVE,
            period=periods[0],
            seasonal_components=seasonal_components,
        )

        return self._calculate_metrics(result)

    def forecast_components(
        self,
        result: DecompositionResult,
        horizon: int,
        trend_method: str = "linear",
    ) -> dict[str, np.ndarray]:
        """Forecast future values using decomposed components.

        Args:
            result: Decomposition result
            horizon: Number of periods to forecast
            trend_method: Method for trend extrapolation

        Returns:
            Dictionary with forecasted components and combined forecast
        """
        assert result is not None, "result must be provided"
        n = len(result.observed)

        # Forecast trend
        if trend_method == "linear":
            # Fit linear trend to last portion
            x = np.arange(n)
            slope, intercept = np.polyfit(x, result.trend, 1)
            future_x = np.arange(n, n + horizon)
            trend_forecast = slope * future_x + intercept
        elif trend_method == "last":
            trend_forecast = np.full(horizon, result.trend[-1])
        else:
            # Exponential extrapolation
            trend_forecast = self._extrapolate_exponential(result.trend, horizon)

        # Forecast seasonal (just repeat the pattern)
        period = result.period
        seasonal_pattern = result.seasonal[-period:]
        seasonal_forecast = np.tile(seasonal_pattern, horizon // period + 1)[:horizon]

        # Combined forecast (additive model)
        if result.seasonal_model == SeasonalModel.ADDITIVE:
            combined = trend_forecast + seasonal_forecast
        else:
            combined = trend_forecast * seasonal_forecast

        return {
            "trend": trend_forecast,
            "seasonal": seasonal_forecast,
            "combined": combined,
            "lower_ci": combined - 1.96 * result.residual_std,
            "upper_ci": combined + 1.96 * result.residual_std,
        }

    def detect_anomalies_in_residuals(
        self,
        result: DecompositionResult,
        threshold: float = 3.0,
    ) -> dict[str, Any]:
        """Detect anomalies in the residual component.

        Args:
            result: Decomposition result
            threshold: Number of standard deviations for anomaly detection

        Returns:
            Dictionary with anomaly information
        """
        assert result is not None, "result must be provided"
        residuals = result.residual
        mean = np.nanmean(residuals)
        std = np.nanstd(residuals)

        if std == 0:
            return {
                "anomaly_indices": np.array([]),
                "anomaly_values": np.array([]),
                "z_scores": np.zeros_like(residuals),
                "threshold": threshold,
            }

        z_scores = (residuals - mean) / std
        anomaly_mask = np.abs(z_scores) > threshold
        anomaly_indices = np.where(anomaly_mask)[0]

        return {
            "anomaly_indices": anomaly_indices,
            "anomaly_values": result.observed[anomaly_indices],
            "z_scores": z_scores,
            "threshold": threshold,
            "num_anomalies": len(anomaly_indices),
            "anomaly_rate": len(anomaly_indices) / len(residuals),
        }

    # Private methods for decomposition

    def _stl_decompose(self, data: np.ndarray, period: int) -> DecompositionResult:
        """STL decomposition implementation."""
        # Initial trend using moving average
        assert data is not None, "data must be provided"
        trend = self._moving_average(data, period)

        # Iterate for robustness
        iterations = self.config.robust_iterations if self.config.robust else 1

        for _ in range(iterations):
            # Detrend
            detrended = data - trend

            # Extract seasonal
            seasonal = self._extract_stl_seasonal(detrended, period)

            # Calculate new trend from deseasonalized data
            deseasonalized = data - seasonal
            trend = self._lowess_smooth(deseasonalized, frac=0.3)

        # Calculate residual
        residual = data - trend - seasonal

        return DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method=DecompositionMethod.STL,
            seasonal_model=SeasonalModel.ADDITIVE,
            period=period,
        )

    def _classical_decompose(
        self,
        data: np.ndarray,
        period: int,
        model: SeasonalModel,
    ) -> DecompositionResult:
        """Classical decomposition."""
        assert data is not None, "data must be provided"
        n = len(data)

        # Step 1: Calculate trend using centered moving average
        trend = self._centered_moving_average(data, period)

        # Step 2: Detrend
        if model == SeasonalModel.ADDITIVE:
            detrended = data - trend
        else:
            detrended = data / np.where(trend != 0, trend, 1.0)

        # Step 3: Calculate seasonal indices
        seasonal_indices = np.zeros(period)
        for i in range(period):
            indices = np.arange(i, n, period)
            valid_mask = ~np.isnan(detrended[indices])
            if np.any(valid_mask):
                seasonal_indices[i] = np.nanmean(detrended[indices])

        # Normalize seasonal indices
        if model == SeasonalModel.ADDITIVE:
            seasonal_indices -= np.mean(seasonal_indices)
        else:
            seasonal_indices /= np.mean(seasonal_indices)

        # Tile seasonal component
        seasonal = np.tile(seasonal_indices, n // period + 1)[:n]

        # Step 4: Calculate residual
        if model == SeasonalModel.ADDITIVE:
            residual = data - trend - seasonal
        else:
            residual = data / (trend * seasonal)
            residual = np.where(np.isfinite(residual), residual, 1.0)

        return DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method=(
                DecompositionMethod.CLASSICAL_ADDITIVE
                if model == SeasonalModel.ADDITIVE
                else DecompositionMethod.CLASSICAL_MULTIPLICATIVE
            ),
            seasonal_model=model,
            period=period,
        )

    def _ma_decompose(self, data: np.ndarray, period: int) -> DecompositionResult:
        """Moving average based decomposition."""
        # Trend
        assert data is not None, "data must be provided"
        trend = self._moving_average(data, period)

        # Detrend
        detrended = data - trend

        # Seasonal
        seasonal = self.extract_seasonal(detrended, period)

        # Residual
        residual = data - trend - seasonal

        return DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method=DecompositionMethod.MOVING_AVERAGE,
            seasonal_model=SeasonalModel.ADDITIVE,
            period=period,
        )

    def _lowess_decompose(self, data: np.ndarray, period: int) -> DecompositionResult:
        """LOWESS-based decomposition."""
        # Trend using LOWESS
        assert data is not None, "data must be provided"
        trend = self._lowess_smooth(data, frac=0.3)

        # Detrend
        detrended = data - trend

        # Seasonal
        seasonal = self.extract_seasonal(detrended, period)

        # Residual
        residual = data - trend - seasonal

        return DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method=DecompositionMethod.LOWESS,
            seasonal_model=SeasonalModel.ADDITIVE,
            period=period,
        )

    def _hp_filter_decompose(
        self, data: np.ndarray, period: int
    ) -> DecompositionResult:
        """Hodrick-Prescott filter decomposition."""
        # Trend using HP filter
        assert data is not None, "data must be provided"
        trend = self._hp_filter(data, self.config.hp_lambda)

        # Cycle (detrended)
        cycle = data - trend

        # Extract seasonal from cycle
        seasonal = self.extract_seasonal(cycle, period)

        # Residual
        residual = cycle - seasonal

        return DecompositionResult(
            observed=data,
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            method=DecompositionMethod.HP_FILTER,
            seasonal_model=SeasonalModel.ADDITIVE,
            period=period,
        )

    # Helper methods

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        assert data is not None, "data must be provided"
        if window < 1:
            window = 1
        if window > len(data):
            window = len(data)

        # Use convolution for efficiency
        kernel = np.ones(window) / window
        ma = np.convolve(data, kernel, mode="same")

        # Handle edges
        half = window // 2
        for i in range(half):
            ma[i] = np.mean(data[: i + half + 1])
            ma[-(i + 1)] = np.mean(data[-(i + half + 1) :])

        return ma

    def _centered_moving_average(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate centered moving average for classical decomposition."""
        assert data is not None, "data must be provided"
        n = len(data)
        result = np.full(n, np.nan)

        half = period // 2

        for i in range(half, n - half):
            if period % 2 == 0:
                # For even periods, use weighted average
                result[i] = (
                    0.5 * data[i - half]
                    + np.sum(data[i - half + 1 : i + half])
                    + 0.5 * data[i + half]
                ) / period
            else:
                result[i] = np.mean(data[i - half : i + half + 1])

        # Fill edges with nearest valid values
        first_valid = half
        last_valid = n - half - 1

        result[:first_valid] = result[first_valid]
        result[last_valid + 1 :] = result[last_valid]

        return result

    def _lowess_smooth(self, data: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """LOWESS (Locally Weighted Scatterplot Smoothing)."""
        assert data is not None, "data must be provided"
        n = len(data)
        x = np.arange(n)
        result = np.zeros(n)

        # Number of points to use for local regression
        k = max(int(frac * n), 2)

        for i in range(n):
            # Calculate distances
            distances = np.abs(x - x[i])

            # Get k nearest neighbors
            nearest_idx = np.argsort(distances)[:k]

            # Calculate weights using tricube kernel
            max_dist = distances[nearest_idx[-1]]
            if max_dist == 0:
                max_dist = 1.0
            u = distances[nearest_idx] / max_dist
            weights = (1 - u**3) ** 3
            weights = np.clip(weights, 0, None)

            # Weighted linear regression
            x_local = x[nearest_idx]
            y_local = data[nearest_idx]

            # Weighted least squares
            sum_w = np.sum(weights)
            if sum_w == 0:
                result[i] = data[i]
                continue

            sum_wx = np.sum(weights * x_local)
            sum_wy = np.sum(weights * y_local)
            sum_wxx = np.sum(weights * x_local * x_local)
            sum_wxy = np.sum(weights * x_local * y_local)

            denom = sum_w * sum_wxx - sum_wx * sum_wx
            if abs(denom) < 1e-10:
                result[i] = sum_wy / sum_w
            else:
                b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
                a = (sum_wy - b * sum_wx) / sum_w
                result[i] = a + b * x[i]

        return result

    def _polynomial_trend(self, data: np.ndarray, degree: int) -> np.ndarray:
        """Fit polynomial trend."""
        assert data is not None, "data must be provided"
        n = len(data)
        x = np.arange(n)
        coeffs = np.polyfit(x, data, degree)
        return np.polyval(coeffs, x)

    def _exponential_smooth(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Exponential smoothing."""
        assert data is not None, "data must be provided"
        n = len(data)
        result = np.zeros(n)
        result[0] = data[0]

        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

        return result

    def _hp_filter(self, data: np.ndarray, lambd: float) -> np.ndarray:
        """Hodrick-Prescott filter implementation."""
        assert data is not None, "data must be provided"
        n = len(data)

        # Construct the penalty matrix
        # Second difference matrix
        d2 = np.zeros((n - 2, n))
        for i in range(n - 2):
            d2[i, i] = 1
            d2[i, i + 1] = -2
            d2[i, i + 2] = 1

        # Solve (I + lambda * D2' * D2) * trend = data
        penalty = lambd * d2.T @ d2
        identity = np.eye(n)

        try:
            trend = np.linalg.solve(identity + penalty, data)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            trend = np.linalg.lstsq(identity + penalty, data, rcond=None)[0]

        return trend

    def _extract_stl_seasonal(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal component for STL."""
        assert detrended is not None, "detrended must be provided"
        n = len(detrended)

        # Calculate cycle-subseries
        seasonal_indices = np.zeros(period)

        for i in range(period):
            subseries = detrended[i::period]
            # Apply LOWESS to each subseries
            smoothed = self._lowess_smooth(subseries, frac=0.5)
            seasonal_indices[i] = np.mean(smoothed)

        # Normalize
        seasonal_indices -= np.mean(seasonal_indices)

        # Tile to match data length
        seasonal = np.tile(seasonal_indices, n // period + 1)[:n]

        return seasonal

    def _autocorrelation(self, data: np.ndarray, max_lag: int) -> np.ndarray:
        """Calculate autocorrelation function."""
        assert data is not None, "data must be provided"
        n = len(data)
        data_centered = data - np.mean(data)
        var = np.var(data_centered)

        if var == 0:
            return np.zeros(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, max_lag + 1):
            acf[lag] = np.sum(data_centered[lag:] * data_centered[:-lag]) / (
                (n - lag) * var
            )

        return acf

    def _find_acf_peaks(self, acf: np.ndarray) -> list[int]:
        """Find peaks in autocorrelation function."""
        assert acf is not None, "acf must be provided"
        peaks = []
        n = len(acf)

        for i in range(2, n - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
                peaks.append(i)

        return peaks

    def _extrapolate_exponential(self, trend: np.ndarray, horizon: int) -> np.ndarray:
        """Extrapolate trend exponentially."""
        # Fit exponential to last portion of trend
        assert trend is not None, "trend must be provided"
        n = len(trend)
        fit_length = min(n, 50)

        x = np.arange(fit_length)
        y = trend[-fit_length:]

        # Log transform for linear fit
        y_pos = y - np.min(y) + 1
        log_y = np.log(y_pos)

        slope, intercept = np.polyfit(x, log_y, 1)

        # Extrapolate
        future_x = np.arange(fit_length, fit_length + horizon)
        forecast = np.exp(intercept + slope * future_x) + np.min(y) - 1

        return forecast

    def _calculate_metrics(self, result: DecompositionResult) -> DecompositionResult:
        """Calculate quality metrics for decomposition."""
        # Variance of detrended data
        assert result is not None, "result must be provided"
        detrended_var = np.var(result.observed - result.trend)
        residual_var = np.var(result.residual)

        # Trend strength
        if detrended_var > 0:
            result.trend_strength = max(0, 1 - residual_var / detrended_var)

        # Seasonal strength: 1 - var(res) / var(sea + res)
        # sea + res = observed - trend
        detrended_var = np.var(result.observed - result.trend)
        if detrended_var > 0:
            result.seasonal_strength = max(0, 1 - residual_var / detrended_var)

        # Residual statistics
        result.residual_variance = residual_var
        result.residual_mean = float(np.mean(result.residual))
        result.residual_std = float(np.std(result.residual))

        # Residual autocorrelation (lag 1)
        if len(result.residual) > 1:
            acf = self._autocorrelation(result.residual, 1)
            result.residual_autocorrelation = float(acf[1]) if len(acf) > 1 else 0.0

        return result


def decompose_time_series(
    data: np.ndarray,
    period: int | None = None,
    method: str = "stl",
) -> DecompositionResult:
    """Convenience function for time series decomposition.

    Args:
        data: 1D time series data
        period: Seasonal period (auto-detected if None)
        method: Decomposition method ('stl', 'classical', 'multiplicative')

    Returns:
        DecompositionResult with trend, seasonal, and residual components

    Example:
        >>> data = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1
        >>> result = decompose_time_series(data, period=25)
        >>> print(f"Trend strength: {result.trend_strength:.2f}")
    """
    assert data is not None, "data must be provided"
    method_map = {
        "stl": DecompositionMethod.STL,
        "classical": DecompositionMethod.CLASSICAL_ADDITIVE,
        "multiplicative": DecompositionMethod.CLASSICAL_MULTIPLICATIVE,
        "ma": DecompositionMethod.MOVING_AVERAGE,
        "lowess": DecompositionMethod.LOWESS,
        "hp": DecompositionMethod.HP_FILTER,
    }

    decomp_method = method_map.get(method.lower(), DecompositionMethod.STL)

    config = DecompositionConfig(method=decomp_method, period=period)
    decomposer = TimeSeriesDecomposer(config)

    return decomposer.decompose(data, period)


__all__ = [
    "DecompositionMethod",
    "SeasonalModel",
    "TrendModel",
    "DecompositionConfig",
    "DecompositionResult",
    "SeasonalityDetectionResult",
    "TimeSeriesDecomposer",
    "decompose_time_series",
]
