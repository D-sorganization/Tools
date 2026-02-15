# mypy: disable-error-code="arg-type"
"""Regression fitting methods.

Provides OLS, Ridge, LASSO, Elastic Net fitting and feature selection
as a mixin for MultivariateRegressor.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

from data_processor.core.regression_types import (
    RegressionConfig,
    SelectionMethod,
)

logger = logging.getLogger(__name__)


class FittingMixin:
    """Fitting and feature selection methods for MultivariateRegressor.

    Expects the host class to provide:
    - self.config: RegressionConfig
    """

    config: RegressionConfig

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
        reg_matrix = np.eye(p + 1)
        reg_matrix[0, 0] = 0

        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y

        coeffs = np.linalg.solve(XtX + alpha * reg_matrix, Xty)

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
        selected: list[int] = []
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
        selected: list[int] = []
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
            return float(n * np.log(mse) + 2 * k)
        elif criterion == "bic":
            return float(n * np.log(mse) + k * np.log(n))
        else:  # r_squared (negative for minimization)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return float(-(1 - ss_res / ss_tot))
