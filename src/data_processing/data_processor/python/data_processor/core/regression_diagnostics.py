from numba import jit

# mypy: disable-error-code="arg-type"
"""Regression diagnostics and statistics.

Provides statistical calculations, diagnostics, VIF computation,
variable importance, and reporting as a mixin for MultivariateRegressor.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy import stats

from data_processor.core.regression_types import (
    CoefficientInfo,
    RegressionConfig,
    RegressionDiagnostics,
    RegressionResult,
)

logger = logging.getLogger(__name__)


class DiagnosticsMixin:
    """Diagnostics and statistics methods for MultivariateRegressor.

    Expects the host class to provide:
    - self.config: RegressionConfig
    - self._fit_ols(X, y): OLS fitting method
    - self._build_features(X, names): Feature building method
    """

    config: RegressionConfig

    def _fit_ols(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]: ...

    def _build_features(
        self,
        X: np.ndarray,
        names: list[str],
    ) -> tuple[np.ndarray, list[str]]: ...

    def _compute_standard_errors(
        self,
        X: np.ndarray,
        n: int,
        p: int,
        mse: float,
    ) -> tuple[float, np.ndarray]:
        """Compute standard errors of intercept and coefficients.

        Args:
            X: Feature matrix (n x p)
            n: Number of observations
            p: Number of predictors
            mse: Mean squared error

        Returns:
            Tuple of (intercept_se, coefficient_se_array)
        """
        if not (X is not None):
            raise ValueError("X must be provided")
        X_with_intercept = np.column_stack([np.ones(n), X])
        try:
            var_covar = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            se = np.sqrt(np.diag(var_covar))
        except np.linalg.LinAlgError:
            se = np.zeros(p + 1)

        intercept_se = se[0] if len(se) > 0 else 0
        coef_se = se[1:] if len(se) > 1 else np.zeros(p)
        return intercept_se, coef_se

    @jit(nopython=True, fastmath=True)
    def _build_coefficient_info(
        self,
        coeffs: np.ndarray,
        coef_se: np.ndarray,
        feature_names: list[str],
        vifs: np.ndarray,
        t_crit: float,
        n: int,
        p: int,
    ) -> list[CoefficientInfo]:
        """Build CoefficientInfo list with t-tests and confidence intervals.

        Args:
            coeffs: Coefficient estimates
            coef_se: Standard errors
            feature_names: Feature names
            vifs: Variance Inflation Factors
            t_crit: Critical t-value
            n: Number of observations
            p: Number of predictors

        Returns:
            List of CoefficientInfo objects
        """
        if not (coeffs is not None):
            raise ValueError("coeffs must be provided")
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
        return coef_info

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
        if not (X is not None):
            raise ValueError("X must be provided")
        n, p = X.shape

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_reg = ss_tot - ss_res

        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
        rmse = np.sqrt(ss_res / n)
        mae = np.mean(np.abs(residuals))
        mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / n

        intercept_se, coef_se = self._compute_standard_errors(X, n, p, mse)

        alpha = 1 - self.config.confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, n - p - 1) if n > p + 1 else 1.96
        vifs = self._calculate_vif(X)

        coef_info = self._build_coefficient_info(coeffs, coef_se, feature_names, vifs, t_crit, n, p)

        df_model = p
        df_residual = n - p - 1
        if df_residual > 0 and ss_res > 0:
            f_stat = (ss_reg / df_model) / (ss_res / df_residual)
            f_p_value = 1 - stats.f.cdf(f_stat, df_model, df_residual)
        else:
            f_stat = 0
            f_p_value = 1

        k = p + 2
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

    @jit(nopython=True, fastmath=True)
    def _calculate_vif(self, X: np.ndarray) -> np.ndarray:
        """Calculate Variance Inflation Factors."""
        if not (X is not None):
            raise ValueError("X must be provided")
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
        if not (X is not None):
            raise ValueError("X must be provided")
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
            np.sum(diff_residuals**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 2
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
        outliers = [i for i, r in enumerate(student_residuals) if abs(r) > outlier_threshold]

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
        if not (coeffs is not None):
            raise ValueError("coeffs must be provided")
        x_std = np.std(X, axis=0)
        x_std[x_std == 0] = 1

        std_coeffs = np.abs(coeffs * x_std)
        total = np.sum(std_coeffs)

        if total > 0:
            importance = std_coeffs / total
        else:
            importance = np.ones(len(coeffs)) / len(coeffs)

        return {name: float(imp) for name, imp in zip(names, importance, strict=False)}

    def _create_predict_func(
        self,
        coeffs: np.ndarray,
        intercept: float,
        feature_names: list[str],
        original_predictors: list[str],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create prediction function for the model."""

        if not (coeffs is not None):
            raise ValueError("coeffs must be provided")

        def predict(X: np.ndarray) -> np.ndarray:
            # Build features if needed
            X_features, _ = self._build_features(X, original_predictors)
            result: np.ndarray = X_features @ coeffs + intercept
            return result

        return predict
