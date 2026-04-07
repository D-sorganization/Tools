"""Public contracts for time-series decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class DecompositionMethod(Enum):
    """Available decomposition methods."""

    STL = "stl"
    CLASSICAL_ADDITIVE = "classical_additive"
    CLASSICAL_MULTIPLICATIVE = "classical_multiplicative"
    MOVING_AVERAGE = "moving_average"
    LOWESS = "lowess"
    HP_FILTER = "hp_filter"


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

    method: DecompositionMethod = DecompositionMethod.STL
    seasonal_model: SeasonalModel = SeasonalModel.ADDITIVE
    period: int | None = None
    seasonal_deg: int = 1
    seasonal_jump: int = 1
    trend_deg: int = 1
    trend_jump: int = 1
    low_pass_deg: int = 1
    low_pass_jump: int = 1
    robust: bool = False
    robust_iterations: int = 2
    hp_lambda: float = 1600.0
    ma_window: int | None = None
    polynomial_degree: int = 2


@dataclass
class DecompositionResult:
    """Results from time-series decomposition."""

    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    method: DecompositionMethod
    seasonal_model: SeasonalModel
    period: int
    trend_strength: float = 0.0
    seasonal_strength: float = 0.0
    residual_variance: float = 0.0
    seasonal_components: dict[int, np.ndarray] = field(default_factory=dict)
    residual_mean: float = 0.0
    residual_std: float = 0.0
    residual_autocorrelation: float = 0.0

    def get_deseasonalized(self) -> np.ndarray:
        """Get data with seasonal component removed."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.observed - self.seasonal
        return self.observed / np.where(self.seasonal != 0, self.seasonal, 1.0)

    def get_detrended(self) -> np.ndarray:
        """Get data with trend removed."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.observed - self.trend
        return self.observed / np.where(self.trend != 0, self.trend, 1.0)

    def get_seasonally_adjusted(self) -> np.ndarray:
        """Get seasonally adjusted series."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.trend + self.residual
        return self.trend * self.residual

    def reconstruct(self) -> np.ndarray:
        """Reconstruct original series from components."""
        if self.seasonal_model == SeasonalModel.ADDITIVE:
            return self.trend + self.seasonal + self.residual
        return self.trend * self.seasonal * self.residual


@dataclass
class SeasonalityDetectionResult:
    """Results from seasonality detection."""

    detected_periods: list[int]
    period_strengths: dict[int, float]
    dominant_period: int | None
    is_seasonal: bool
    autocorrelation_peaks: list[int]


__all__ = [
    "DecompositionConfig",
    "DecompositionMethod",
    "DecompositionResult",
    "SeasonalModel",
    "SeasonalityDetectionResult",
    "TrendModel",
]
