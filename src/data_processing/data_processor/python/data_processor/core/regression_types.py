# mypy: disable-error-code="arg-type"
"""Regression type definitions.

Shared enums, dataclasses, and configuration for the regression module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RegularizationType(Enum):
    """Types of regularization."""

    NONE = "none"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"


class SelectionMethod(Enum):
    """Feature selection methods."""

    NONE = "none"
    FORWARD = "forward"
    BACKWARD = "backward"
    STEPWISE = "stepwise"


@dataclass
class CoefficientInfo:
    """Information about a regression coefficient."""

    name: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    vif: float = 1.0  # Variance Inflation Factor

    @property
    def significant(self) -> bool:
        """Check if coefficient is significant at 0.05 level."""
        return self.p_value < 0.05


@dataclass
class RegressionDiagnostics:
    """Diagnostic statistics for regression."""

    # Residual analysis
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    studentized_residuals: np.ndarray

    # Influence measures
    leverage: np.ndarray
    cooks_distance: np.ndarray

    # Tests
    durbin_watson: float
    breusch_pagan_stat: float
    breusch_pagan_p: float
    shapiro_stat: float
    shapiro_p: float

    # Outlier/influence points
    high_leverage_points: list[int]
    influential_points: list[int]
    outlier_points: list[int]


@dataclass
class RegressionResult:
    """Complete results from regression analysis."""

    # Model information
    model_type: str
    n_observations: int
    n_predictors: int

    # Coefficients
    coefficients: list[CoefficientInfo]
    intercept: float
    intercept_se: float

    # Goodness of fit
    r_squared: float
    adj_r_squared: float
    rmse: float
    mae: float
    aic: float
    bic: float

    # Overall model test
    f_statistic: float
    f_p_value: float
    df_model: int
    df_residual: int

    # Predictions
    fitted_values: np.ndarray
    residuals: np.ndarray

    # Feature names
    feature_names: list[str]

    # For surface plot integration
    predict_func: Callable[[np.ndarray], np.ndarray] | None = None

    # Diagnostics
    diagnostics: RegressionDiagnostics | None = None

    # Variable importance
    variable_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class RegressionConfig:
    """Configuration for regression analysis."""

    # Regularization
    regularization: RegularizationType = RegularizationType.NONE
    alpha: float = 1.0  # Regularization strength
    l1_ratio: float = 0.5  # For elastic net

    # Feature selection
    selection_method: SelectionMethod = SelectionMethod.NONE
    selection_criterion: str = "aic"  # "aic", "bic", "p_value"
    p_entry: float = 0.05
    p_removal: float = 0.10

    # Polynomial features
    polynomial_degree: int = 1
    include_interactions: bool = False

    # Other options
    standardize: bool = False
    confidence_level: float = 0.95
    compute_diagnostics: bool = True
