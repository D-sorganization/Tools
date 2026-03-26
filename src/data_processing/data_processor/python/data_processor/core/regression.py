# mypy: disable-error-code="arg-type"
"""Multivariable Regression Module.

Provides comprehensive regression analysis including:
- Multiple linear regression
- Polynomial regression
- Ridge and Lasso regularization
- Stepwise regression (forward/backward selection)
- Interaction terms
- Diagnostic plots and statistics
- Integration with surface plots for visualization

Designed for modeling complex relationships in multivariate data.

This module serves as a facade, composing the following submodules:
- regression_types: Enums and dataclasses
- regression_fitting: OLS, Ridge, LASSO, Elastic Net, feature selection
- regression_diagnostics: Statistics, diagnostics, VIF, importance
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Re-export all public types for backward compatibility
from data_processor.core.regression_diagnostics import DiagnosticsMixin
from data_processor.core.regression_fitting import FittingMixin
from data_processor.core.regression_types import (
    CoefficientInfo,
    RegressionConfig,
    RegressionDiagnostics,
    RegressionResult,
    RegularizationType,
    SelectionMethod,
)

logger = logging.getLogger(__name__)


class MultivariateRegressor(FittingMixin, DiagnosticsMixin):
    """Comprehensive multivariable regression analyzer.

    Provides linear and polynomial regression with various
    regularization and feature selection options.
    """

    def __init__(self, config: RegressionConfig | None = None) -> None:
        """Initialize the regressor.

        Args:
            config: Regression configuration
        """
        self.config = config or RegressionConfig()

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        predictors: list[str] | None = None,
    ) -> RegressionResult:
        """Fit regression model.

        Args:
            df: DataFrame with data
            target: Name of target variable
            predictors: List of predictor names (None = all numeric except target)

        Returns:
            Complete regression results
        """
        # Select predictors
        if predictors is None:
            predictors = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != target
            ]

        if not predictors:
            raise ValueError("No predictors available")

        # Prepare data
        data = df[[target] + predictors].dropna()
        y = data[target].values
        X_raw = data[predictors].values
        n, p = X_raw.shape

        if n < p + 1:
            raise ValueError(f"Not enough observations ({n}) for {p} predictors")

        # Build feature matrix (polynomial, interactions)
        X, feature_names = self._build_features(X_raw, predictors)

        # Apply feature selection if requested
        if self.config.selection_method != SelectionMethod.NONE:
            X, feature_names = self._select_features(X, y, feature_names)

        # Standardize if requested
        X_scaled, x_mean, x_std, y_mean, y_std = self._standardize_data(X, y)

        # Fit model
        if self.config.regularization == RegularizationType.NONE:
            coeffs, intercept = self._fit_ols(X_scaled, y)
        elif self.config.regularization == RegularizationType.RIDGE:
            coeffs, intercept = self._fit_ridge(X_scaled, y)
        elif self.config.regularization == RegularizationType.LASSO:
            coeffs, intercept = self._fit_lasso(X_scaled, y)
        else:
            coeffs, intercept = self._fit_elastic_net(X_scaled, y)

        # Calculate predictions and residuals
        y_pred = X @ coeffs + intercept
        residuals = y - y_pred

        # Calculate statistics
        result = self._calculate_statistics(
            X, y, y_pred, residuals, coeffs, intercept, feature_names
        )

        # Create prediction function for surface plots
        result.predict_func = self._create_predict_func(
            coeffs, intercept, feature_names, predictors
        )

        # Calculate diagnostics
        if self.config.compute_diagnostics:
            result.diagnostics = self._calculate_diagnostics(X, y, y_pred, residuals)

        # Calculate variable importance
        result.variable_importance = self._calculate_importance(
            coeffs, feature_names, X
        )

        return result

    def predict(
        self,
        result: RegressionResult,
        new_data: pd.DataFrame,
    ) -> np.ndarray:
        """Make predictions using fitted model.

        Args:
            result: Fitted regression result
            new_data: DataFrame with predictor values

        Returns:
            Predicted values
        """
        if result.predict_func is None:
            raise ValueError("Model does not have a prediction function")

        X = new_data[result.feature_names].values
        return result.predict_func(X)

    def predict_surface(
        self,
        result: RegressionResult,
        x_var: str,
        y_var: str,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        grid_size: int = 50,
        fixed_values: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate surface prediction for 3D plotting.

        Args:
            result: Fitted regression result
            x_var: Variable for x-axis
            y_var: Variable for y-axis
            x_range: (min, max) for x
            y_range: (min, max) for y
            grid_size: Number of grid points per dimension
            fixed_values: Fixed values for other predictors

        Returns:
            X grid, Y grid, Z predictions
        """
        # Create grid
        if not (result is not None):
            raise ValueError("result must be provided")
        x_lin = np.linspace(x_range[0], x_range[1], grid_size)
        y_lin = np.linspace(y_range[0], y_range[1], grid_size)
        x_grid, y_grid = np.meshgrid(x_lin, y_lin)

        # Create prediction data
        n_points = grid_size * grid_size
        pred_data = {}

        # Find original predictor names (without polynomial/interaction suffixes)
        original_predictors = list(
            set(
                name.split("^")[0].split("×")[0]
                for name in result.feature_names
                if name != "intercept"
            )
        )

        for var in original_predictors:
            if var == x_var:
                pred_data[var] = x_grid.ravel()
            elif var == y_var:
                pred_data[var] = y_grid.ravel()
            elif fixed_values and var in fixed_values:
                pred_data[var] = np.full(n_points, fixed_values[var])
            else:
                # Use mean value
                pred_data[var] = np.zeros(n_points)

        pred_df = pd.DataFrame(pred_data)

        # Build features
        X_raw = pred_df.values
        X_features, _ = self._build_features(X_raw, original_predictors)

        # Predict
        z_pred = (
            X_features @ np.array([c.estimate for c in result.coefficients])
            + result.intercept
        )
        z_grid = z_pred.reshape(x_grid.shape)

        return x_grid, y_grid, z_grid


def format_regression_report(result: RegressionResult) -> str:
    """Format regression results as a text report."""
    lines = ["=" * 70, "REGRESSION RESULTS", "=" * 70, ""]

    # Model summary
    lines.append(f"Model: {result.model_type}")
    lines.append(f"Observations: {result.n_observations}")
    lines.append(f"Predictors: {result.n_predictors}")
    lines.append("")

    # Goodness of fit
    lines.append("Goodness of Fit:")
    lines.append(f"  R-squared:          {result.r_squared:.4f}")
    lines.append(f"  Adjusted R-squared: {result.adj_r_squared:.4f}")
    lines.append(f"  RMSE:               {result.rmse:.4f}")
    lines.append(f"  MAE:                {result.mae:.4f}")
    lines.append(f"  AIC:                {result.aic:.2f}")
    lines.append(f"  BIC:                {result.bic:.2f}")
    lines.append("")

    # Overall model test
    lines.append(
        f"F-statistic: {result.f_statistic:.4f} "
        f"(df={result.df_model}, {result.df_residual})"
    )
    lines.append(f"p-value: {result.f_p_value:.4e}")
    lines.append("")

    # Coefficients table
    lines.append("Coefficients:")
    lines.append("-" * 70)
    lines.append(
        f"{'Variable':<20} {'Estimate':>12} "
        f"{'Std.Err':>10} {'t-stat':>10} {'p-value':>10}"
    )
    lines.append("-" * 70)

    # Intercept
    lines.append(
        f"{'(Intercept)':<20} {result.intercept:>12.4f} {result.intercept_se:>10.4f}"
    )

    # Coefficients
    for coef in result.coefficients:
        sig = "*" if coef.significant else ""
        lines.append(
            f"{coef.name:<20} {coef.estimate:>12.4f} {coef.std_error:>10.4f} "
            f"{coef.t_statistic:>10.4f} {coef.p_value:>10.4e} {sig}"
        )

    lines.append("-" * 70)
    lines.append("Signif. codes: * p < 0.05")

    # Variable importance
    if result.variable_importance:
        lines.append("")
        lines.append("Variable Importance (standardized):")
        sorted_imp = sorted(
            result.variable_importance.items(), key=lambda x: x[1], reverse=True
        )
        for name, imp in sorted_imp[:10]:
            lines.append(f"  {name}: {imp:.4f}")

    # Diagnostics summary
    if result.diagnostics:
        diag = result.diagnostics
        lines.append("")
        lines.append("Diagnostics:")
        lines.append(f"  Durbin-Watson:     {diag.durbin_watson:.4f}")
        lines.append(
            f"  Breusch-Pagan:     \u03c7\u00b2 = {diag.breusch_pagan_stat:.4f}, "
            f"p = {diag.breusch_pagan_p:.4f}"
        )
        lines.append(
            f"  Shapiro-Wilk:      W = {diag.shapiro_stat:.4f}, "
            f"p = {diag.shapiro_p:.4f}"
        )

        if diag.high_leverage_points:
            lines.append(f"  High leverage points: {len(diag.high_leverage_points)}")
        if diag.influential_points:
            lines.append(f"  Influential points: {len(diag.influential_points)}")
        if diag.outlier_points:
            lines.append(f"  Outliers: {len(diag.outlier_points)}")

    lines.append("=" * 70)
    return "\n".join(lines)


__all__ = [
    "RegularizationType",
    "SelectionMethod",
    "CoefficientInfo",
    "RegressionDiagnostics",
    "RegressionResult",
    "RegressionConfig",
    "MultivariateRegressor",
    "format_regression_report",
]
