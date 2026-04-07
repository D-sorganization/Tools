# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

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
from typing import Any

import numpy as np
from numba import jit

from data_processor.core.time_series_decomposition_contracts import (
    DecompositionConfig,
    DecompositionMethod,
    DecompositionResult,
    SeasonalityDetectionResult,
    SeasonalModel,
    TrendModel,
)
from data_processor.core.time_series_decomposition_helpers import (
    autocorrelation,
    centered_moving_average,
    exponential_smooth,
    extract_stl_seasonal,
    extrapolate_exponential,
    find_acf_peaks,
    hp_filter,
    lowess_smooth,
    moving_average,
    polynomial_trend,
)

logger = logging.getLogger(__name__)


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
        if not (data is not None):
            raise ValueError("data must be provided")
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        if max_period is None:
            max_period = min(n // 2, 365)  # Reasonable default

        # Calculate autocorrelation
        acf = autocorrelation(data, max_period)

        # Find peaks in ACF
        peaks = find_acf_peaks(acf)

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
        if not (data is not None):
            raise ValueError("data must be provided")
        data = np.asarray(data, dtype=np.float64)

        if method == TrendModel.MOVING_AVERAGE:
            window = kwargs.get("window", self.config.ma_window or len(data) // 10)
            return moving_average(data, window)

        elif method == TrendModel.LOWESS:
            frac = kwargs.get("frac", 0.3)
            return lowess_smooth(data, frac)

        elif method == TrendModel.POLYNOMIAL:
            degree = kwargs.get("degree", self.config.polynomial_degree)
            return polynomial_trend(data, degree)

        elif method == TrendModel.EXPONENTIAL:
            alpha = kwargs.get("alpha", 0.3)
            return exponential_smooth(data, alpha)

        elif method == TrendModel.HP_FILTER:
            lambd = kwargs.get("lambda", self.config.hp_lambda)
            return hp_filter(data, lambd)

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
        if not (data is not None):
            raise ValueError("data must be provided")
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

    @jit(nopython=True, fastmath=True)
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
        if not (data is not None):
            raise ValueError("data must be provided")
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
            trend = moving_average(residual, period)

            # Detrend
            detrended = residual - trend

            # Extract seasonal
            seasonal = self.extract_seasonal(detrended, period)
            seasonal_components[period] = seasonal
            combined_seasonal += seasonal

            # Remove this seasonal component
            residual = residual - seasonal

        # Final trend extraction
        trend = moving_average(residual, periods[-1])
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
        if not (result is not None):
            raise ValueError("result must be provided")
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
            trend_forecast = extrapolate_exponential(result.trend, horizon)

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
        if not (result is not None):
            raise ValueError("result must be provided")
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
        if not (data is not None):
            raise ValueError("data must be provided")
        trend = moving_average(data, period)

        # Iterate for robustness
        iterations = self.config.robust_iterations if self.config.robust else 1

        for _ in range(iterations):
            # Detrend
            detrended = data - trend

            # Extract seasonal
            seasonal = extract_stl_seasonal(detrended, period)

            # Calculate new trend from deseasonalized data
            deseasonalized = data - seasonal
            trend = lowess_smooth(deseasonalized, frac=0.3)

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
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)

        # Step 1: Calculate trend using centered moving average
        trend = centered_moving_average(data, period)

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
        if not (data is not None):
            raise ValueError("data must be provided")
        trend = moving_average(data, period)

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
        if not (data is not None):
            raise ValueError("data must be provided")
        trend = lowess_smooth(data, frac=0.3)

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
        if not (data is not None):
            raise ValueError("data must be provided")
        trend = hp_filter(data, self.config.hp_lambda)

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

    def _calculate_metrics(self, result: DecompositionResult) -> DecompositionResult:
        """Calculate quality metrics for decomposition."""
        # Variance of detrended data
        if not (result is not None):
            raise ValueError("result must be provided")
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
            acf = autocorrelation(result.residual, 1)
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
    if not (data is not None):
        raise ValueError("data must be provided")
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
