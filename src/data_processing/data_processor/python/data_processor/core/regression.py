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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


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


class MultivariateRegressor:
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

    def _build_features(
        self,
        X: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Build feature matrix with polynomial and interaction terms."""
        features = [X]
        feature_names = list(names)

        degree = self.config.polynomial_degree

        # Polynomial terms
        if degree > 1:
            for d in range(2, degree + 1):
                for i, name in enumerate(names):
                    features.append(X[:, i : i + 1] ** d)
                    feature_names.append(f"{name}^{d}")

        # Interaction terms
        if self.config.include_interactions:
            for i, j in combinations(range(len(names)), 2):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
                feature_names.append(f"{names[i]}×{names[j]}")

        X_full = np.hstack(features)
        return X_full, feature_names

    def _standardize_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Standardize features and target."""
        if self.config.standardize:
            x_mean = np.mean(X, axis=0)
            x_std = np.std(X, axis=0)
            x_std[x_std == 0] = 1
            X_scaled = (X - x_mean) / x_std

            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std == 0:
                y_std = 1
        else:
            X_scaled = X
            x_mean = np.zeros(X.shape[1])
            x_std = np.ones(X.shape[1])
            y_mean = 0
            y_std = 1

        return X_scaled, x_mean, x_std, y_mean, y_std

    def _fit_ols(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit ordinary least squares regression."""
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(y)), X])

        # Solve normal equations
        try:
            coeffs = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            coeffs = np.linalg.pinv(X_with_intercept) @ y

        intercept = coeffs[0]
        beta = coeffs[1:]

        return beta, intercept

    def _fit_ridge(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit ridge regression (L2 regularization)."""
        n, p = X.shape
        alpha = self.config.alpha

        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n), X])

        # Ridge normal equations: (X'X + αI)β = X'y
        # Don't regularize intercept
        I = np.eye(p + 1)
        I[0, 0] = 0

        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y

        coeffs = np.linalg.solve(XtX + alpha * I, Xty)

        return coeffs[1:], coeffs[0]

    def _fit_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit LASSO regression (L1 regularization) using coordinate descent."""
        n, p = X.shape
        alpha = self.config.alpha

        # Initialize with OLS
        beta, intercept = self._fit_ols(X, y)

        # Coordinate descent
        max_iter = 1000
        tol = 1e-6

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                # Residual without j-th feature
                r_j = y - intercept - X @ beta + X[:, j] * beta[j]

                # Soft thresholding
                rho = X[:, j] @ r_j
                z = X[:, j] @ X[:, j]

                if z > 0:
                    beta[j] = self._soft_threshold(rho / z, alpha * n / z)

            # Update intercept
            intercept = np.mean(y - X @ beta)

            # Check convergence
            if np.max(np.abs(beta - beta_old)) < tol:
                break

        return beta, intercept

    def _fit_elastic_net(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fit elastic net regression (L1 + L2 regularization)."""
        n, p = X.shape
        alpha = self.config.alpha
        l1_ratio = self.config.l1_ratio

        # Initialize with OLS
        beta, intercept = self._fit_ols(X, y)

        # Coordinate descent with elastic net penalty
        max_iter = 1000
        tol = 1e-6

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                r_j = y - intercept - X @ beta + X[:, j] * beta[j]
                rho = X[:, j] @ r_j
                z = X[:, j] @ X[:, j] + alpha * (1 - l1_ratio) * n

                if z > 0:
                    beta[j] = self._soft_threshold(rho / z, alpha * l1_ratio * n / z)

            intercept = np.mean(y - X @ beta)

            if np.max(np.abs(beta - beta_old)) < tol:
                break

        return beta, intercept

    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator for LASSO."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        return 0.0

    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """Perform feature selection."""
        method = self.config.selection_method
        criterion = self.config.selection_criterion

        if method == SelectionMethod.FORWARD:
            return self._forward_selection(X, y, names, criterion)
        elif method == SelectionMethod.BACKWARD:
            return self._backward_selection(X, y, names, criterion)
        elif method == SelectionMethod.STEPWISE:
            return self._stepwise_selection(X, y, names, criterion)

        return X, names

    def _forward_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        criterion: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Forward stepwise selection."""
        n, p = X.shape
        selected = []
        remaining = list(range(p))
        best_score = np.inf

        while remaining:
            scores = []
            for idx in remaining:
                test_features = selected + [idx]
                X_test = X[:, test_features]
                score = self._calculate_criterion(X_test, y, criterion)
                scores.append((score, idx))

            best_new_score, best_idx = min(scores)

            if best_new_score < best_score:
                best_score = best_new_score
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        if not selected:
            selected = [0]  # Keep at least one feature

        X_selected = X[:, selected]
        names_selected = [names[i] for i in selected]

        return X_selected, names_selected

    def _backward_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        criterion: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Backward stepwise selection."""
        n, p = X.shape
        selected = list(range(p))
        best_score = self._calculate_criterion(X, y, criterion)

        while len(selected) > 1:
            scores = []
            for idx in selected:
                test_features = [i for i in selected if i != idx]
                X_test = X[:, test_features]
                score = self._calculate_criterion(X_test, y, criterion)
                scores.append((score, idx))

            best_new_score, remove_idx = min(scores)

            if best_new_score < best_score:
                best_score = best_new_score
                selected.remove(remove_idx)
            else:
                break

        X_selected = X[:, selected]
        names_selected = [names[i] for i in selected]

        return X_selected, names_selected

    def _stepwise_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        criterion: str,
    ) -> tuple[np.ndarray, list[str]]:
        """Bidirectional stepwise selection."""
        # Start with forward selection, then try backward at each step
        n, p = X.shape
        selected = []
        remaining = list(range(p))
        best_score = np.inf

        while True:
            improved = False

            # Forward step
            if remaining:
                scores = []
                for idx in remaining:
                    test_features = selected + [idx]
                    X_test = X[:, test_features]
                    score = self._calculate_criterion(X_test, y, criterion)
                    scores.append((score, idx))

                best_new_score, best_idx = min(scores)

                if best_new_score < best_score:
                    best_score = best_new_score
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    improved = True

            # Backward step
            if len(selected) > 1:
                scores = []
                for idx in selected:
                    test_features = [i for i in selected if i != idx]
                    X_test = X[:, test_features]
                    score = self._calculate_criterion(X_test, y, criterion)
                    scores.append((score, idx))

                best_new_score, remove_idx = min(scores)

                if best_new_score < best_score:
                    best_score = best_new_score
                    selected.remove(remove_idx)
                    remaining.append(remove_idx)
                    improved = True

            if not improved:
                break

        if not selected:
            selected = [0]

        X_selected = X[:, selected]
        names_selected = [names[i] for i in selected]

        return X_selected, names_selected

    def _calculate_criterion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: str,
    ) -> float:
        """Calculate model selection criterion."""
        n = len(y)
        k = X.shape[1] + 1  # +1 for intercept

        # Fit model
        beta, intercept = self._fit_ols(X, y)
        y_pred = X @ beta + intercept
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        mse = ss_res / n

        if criterion == "aic":
            return n * np.log(mse) + 2 * k
        elif criterion == "bic":
            return n * np.log(mse) + k * np.log(n)
        else:  # r_squared (negative for minimization)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return -(1 - ss_res / ss_tot)

    def _calculate_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        coeffs: np.ndarray,
        intercept: float,
        feature_names: list[str],
    ) -> RegressionResult:
        """Calculate comprehensive regression statistics."""
        n, p = X.shape

        # Sum of squares
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_reg = ss_tot - ss_res

        # R-squared and adjusted R-squared
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = (
            1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        )

        # RMSE and MAE
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(residuals))

        # MSE for standard errors
        mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / n

        # Standard errors of coefficients
        X_with_intercept = np.column_stack([np.ones(n), X])
        try:
            var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            se = np.zeros(p + 1)

        intercept_se = se[0] if len(se) > 0 else 0
        coef_se = se[1:] if len(se) > 1 else np.zeros(p)

        # Confidence intervals
        alpha = 1 - self.config.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - p - 1) if n > p + 1 else 1.96

        # Calculate VIF
        vifs = self._calculate_vif(X)

        # Build coefficient info
        coef_info = []
        for i, name in enumerate(feature_names):
            t_stat = coeffs[i] / coef_se[i] if coef_se[i] > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - p - 1)) if n > p + 1 else 1

            coef_info.append(
                CoefficientInfo(
                    name=name,
                    estimate=float(coeffs[i]),
                    std_error=float(coef_se[i]),
                    t_statistic=float(t_stat),
                    p_value=float(p_val),
                    ci_lower=float(coeffs[i] - t_crit * coef_se[i]),
                    ci_upper=float(coeffs[i] + t_crit * coef_se[i]),
                    vif=float(vifs[i]) if i < len(vifs) else 1.0,
                )
            )

        # F-statistic
        df_model = p
        df_residual = n - p - 1
        if df_residual > 0 and ss_res > 0:
            f_stat = (ss_reg / df_model) / (ss_res / df_residual)
            f_p_value = 1 - stats.f.cdf(f_stat, df_model, df_residual)
        else:
            f_stat = 0
            f_p_value = 1

        # AIC and BIC
        k = p + 2  # coefficients + intercept + variance
        aic = n * np.log(ss_res / n) + 2 * k
        bic = n * np.log(ss_res / n) + k * np.log(n)

        return RegressionResult(
            model_type="Multiple Linear Regression",
            n_observations=n,
            n_predictors=p,
            coefficients=coef_info,
            intercept=float(intercept),
            intercept_se=float(intercept_se),
            r_squared=float(r_squared),
            adj_r_squared=float(adj_r_squared),
            rmse=float(rmse),
            mae=float(mae),
            aic=float(aic),
            bic=float(bic),
            f_statistic=float(f_stat),
            f_p_value=float(f_p_value),
            df_model=df_model,
            df_residual=df_residual,
            fitted_values=y_pred,
            residuals=residuals,
            feature_names=feature_names,
        )

    def _calculate_vif(self, X: np.ndarray) -> np.ndarray:
        """Calculate Variance Inflation Factors."""
        n, p = X.shape
        vifs = np.ones(p)

        for i in range(p):
            other_cols = [j for j in range(p) if j != i]
            if not other_cols:
                continue

            X_other = X[:, other_cols]
            y_i = X[:, i]

            # Fit regression of x_i on other x's
            beta, intercept = self._fit_ols(X_other, y_i)
            y_pred = X_other @ beta + intercept
            ss_res = np.sum((y_i - y_pred) ** 2)
            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)

            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r_squared < 1:
                vifs[i] = 1 / (1 - r_squared)
            else:
                vifs[i] = np.inf

        return vifs

    def _calculate_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        residuals: np.ndarray,
    ) -> RegressionDiagnostics:
        """Calculate regression diagnostics."""
        n, p = X.shape

        # Hat matrix for leverage
        X_with_intercept = np.column_stack([np.ones(n), X])
        try:
            H = (
                X_with_intercept
                @ np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                @ X_with_intercept.T
            )
            leverage = np.diag(H)
        except np.linalg.LinAlgError:
            leverage = np.zeros(n)

        # MSE
        mse = np.sum(residuals**2) / (n - p - 1) if n > p + 1 else np.var(residuals)

        # Standardized residuals
        std_residuals = residuals / np.sqrt(mse) if mse > 0 else residuals

        # Studentized residuals
        with np.errstate(divide="ignore", invalid="ignore"):
            student_residuals = residuals / np.sqrt(mse * (1 - leverage))
            student_residuals = np.nan_to_num(student_residuals)

        # Cook's distance
        with np.errstate(divide="ignore", invalid="ignore"):
            cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage))
            cooks_d = np.nan_to_num(cooks_d)

        # Durbin-Watson statistic
        diff_residuals = np.diff(residuals)
        durbin_watson = (
            np.sum(diff_residuals**2) / np.sum(residuals**2)
            if np.sum(residuals**2) > 0
            else 2
        )

        # Breusch-Pagan test for heteroscedasticity
        residuals_sq = residuals**2
        beta_bp, intercept_bp = self._fit_ols(X, residuals_sq)
        fitted_bp = X @ beta_bp + intercept_bp
        ss_res_bp = np.sum((residuals_sq - fitted_bp) ** 2)
        ss_tot_bp = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
        r_sq_bp = 1 - ss_res_bp / ss_tot_bp if ss_tot_bp > 0 else 0
        bp_stat = n * r_sq_bp
        bp_p = 1 - stats.chi2.cdf(bp_stat, p)

        # Shapiro-Wilk test for normality
        if n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            # Use subset for large samples
            shapiro_stat, shapiro_p = stats.shapiro(
                np.random.choice(residuals, 5000, replace=False)
            )

        # Identify problematic points
        leverage_threshold = 2 * (p + 1) / n
        high_leverage = [i for i, h in enumerate(leverage) if h > leverage_threshold]

        cooks_threshold = 4 / n
        influential = [i for i, c in enumerate(cooks_d) if c > cooks_threshold]

        outlier_threshold = 3
        outliers = [
            i for i, r in enumerate(student_residuals) if abs(r) > outlier_threshold
        ]

        return RegressionDiagnostics(
            residuals=residuals,
            standardized_residuals=std_residuals,
            studentized_residuals=student_residuals,
            leverage=leverage,
            cooks_distance=cooks_d,
            durbin_watson=float(durbin_watson),
            breusch_pagan_stat=float(bp_stat),
            breusch_pagan_p=float(bp_p),
            shapiro_stat=float(shapiro_stat),
            shapiro_p=float(shapiro_p),
            high_leverage_points=high_leverage,
            influential_points=influential,
            outlier_points=outliers,
        )

    def _calculate_importance(
        self,
        coeffs: np.ndarray,
        names: list[str],
        X: np.ndarray,
    ) -> dict[str, float]:
        """Calculate variable importance (standardized coefficients)."""
        # Standardize coefficients by feature standard deviation
        x_std = np.std(X, axis=0)
        x_std[x_std == 0] = 1

        std_coeffs = np.abs(coeffs * x_std)
        total = np.sum(std_coeffs)

        if total > 0:
            importance = std_coeffs / total
        else:
            importance = np.ones(len(coeffs)) / len(coeffs)

        return {name: float(imp) for name, imp in zip(names, importance)}

    def _create_predict_func(
        self,
        coeffs: np.ndarray,
        intercept: float,
        feature_names: list[str],
        original_predictors: list[str],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create prediction function for the model."""

        def predict(X: np.ndarray) -> np.ndarray:
            # Build features if needed
            X_features, _ = self._build_features(X, original_predictors)
            return X_features @ coeffs + intercept

        return predict


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
        f"F-statistic: {result.f_statistic:.4f} (df={result.df_model}, {result.df_residual})"
    )
    lines.append(f"p-value: {result.f_p_value:.4e}")
    lines.append("")

    # Coefficients table
    lines.append("Coefficients:")
    lines.append("-" * 70)
    lines.append(
        f"{'Variable':<20} {'Estimate':>12} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}"
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
            f"  Breusch-Pagan:     χ² = {diag.breusch_pagan_stat:.4f}, p = {diag.breusch_pagan_p:.4f}"
        )
        lines.append(
            f"  Shapiro-Wilk:      W = {diag.shapiro_stat:.4f}, p = {diag.shapiro_p:.4f}"
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
