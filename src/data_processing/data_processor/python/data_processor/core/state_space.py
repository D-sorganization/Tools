"""State Space Modeling Module.

Provides comprehensive state space model implementations for
time series analysis, system identification, and forecasting.

Features:
- Linear state space models (local level, local linear trend)
- ARIMA in state space form
- Structural time series models
- Parameter estimation via MLE
- Kalman filter/smoother for state estimation
- Model diagnostics and selection
- Forecasting with prediction intervals
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class StateSpaceModelType(Enum):
    """Available state space model types."""

    LOCAL_LEVEL = "local_level"
    LOCAL_LINEAR_TREND = "local_linear_trend"
    SEASONAL = "seasonal"
    ARIMA = "arima"
    STRUCTURAL = "structural"
    CUSTOM = "custom"


class OptimizationMethod(Enum):
    """Parameter optimization methods."""

    GRADIENT_DESCENT = "gradient_descent"
    BFGS = "bfgs"
    NELDER_MEAD = "nelder_mead"
    EM = "em"  # Expectation-Maximization


@dataclass
class StateSpaceConfig:
    """Configuration for state space models."""

    # Model type
    model_type: StateSpaceModelType = StateSpaceModelType.LOCAL_LEVEL

    # Seasonal settings
    seasonal_period: int | None = None

    # ARIMA settings
    ar_order: int = 0
    diff_order: int = 0
    ma_order: int = 0

    # Estimation settings
    optimization_method: OptimizationMethod = OptimizationMethod.BFGS
    max_iterations: int = 2000
    tolerance: float = 1e-4

    # Initial values
    initial_state: np.ndarray | None = None
    initial_state_cov: np.ndarray | None = None

    # Forecasting
    forecast_horizon: int = 10
    confidence_level: float = 0.95


@dataclass
class StateSpaceResult:
    """Results from state space model fitting."""

    # Fitted values
    fitted_values: np.ndarray
    residuals: np.ndarray

    # State estimates
    filtered_states: np.ndarray
    smoothed_states: np.ndarray
    filtered_state_cov: np.ndarray
    smoothed_state_cov: np.ndarray

    # Model parameters
    parameters: dict[str, float]
    log_likelihood: float

    # Model information
    model_type: StateSpaceModelType
    n_observations: int
    n_states: int

    # Diagnostics
    aic: float = 0.0
    bic: float = 0.0
    mse: float = 0.0
    mae: float = 0.0

    # Convergence
    converged: bool = True
    n_iterations: int = 0


@dataclass
class ForecastResult:
    """Results from state space forecasting."""

    # Point forecasts
    forecast: np.ndarray
    forecast_index: np.ndarray

    # Uncertainty
    forecast_std: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray

    # State forecasts
    state_forecast: np.ndarray
    state_forecast_cov: np.ndarray

    # Confidence level used
    confidence_level: float


class BaseStateSpaceModel(ABC):
    """Abstract base class for state space models.

    State space form:
        State equation:   x_t = T * x_{t-1} + R * eta_t
        Observation eq:   y_t = Z * x_t + eps_t

    Where:
        x_t: State vector (n_states x 1)
        y_t: Observation (n_obs x 1)
        T: Transition matrix (n_states x n_states)
        Z: Design matrix (n_obs x n_states)
        R: Selection matrix (n_states x n_state_shocks)
        eta_t: State disturbance ~ N(0, Q)
        eps_t: Observation error ~ N(0, H)
    """

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        """Initialize the model.

        Args:
            config: Model configuration
        """
        self.config = config or StateSpaceConfig()
        self._is_fitted = False

        # Model matrices (to be set by subclasses)
        self.T: np.ndarray | None = None  # Transition
        self.Z: np.ndarray | None = None  # Design
        self.R: np.ndarray | None = None  # Selection
        self.Q: np.ndarray | None = None  # State covariance
        self.H: np.ndarray | None = None  # Observation covariance

        # Dimensions
        self.n_states = 0
        self.n_obs = 1

        self._last_state: np.ndarray | None = None
        self._last_cov: np.ndarray | None = None

        # Estimated parameters
        self._parameters: dict[str, float] = {}

    @abstractmethod
    def _initialize_matrices(self, y: np.ndarray) -> None:
        """Initialize model matrices based on data."""

    @abstractmethod
    def _update_matrices(self, parameters: np.ndarray) -> None:
        """Update matrices with new parameter values."""

    @abstractmethod
    def _get_initial_parameters(self) -> np.ndarray:
        """Get initial parameter values for optimization."""

    @abstractmethod
    def _parameters_to_dict(self, parameters: np.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary."""

    def fit(self, y: np.ndarray) -> StateSpaceResult:
        """Fit the state space model to data.

        Args:
            y: Time series observations

        Returns:
            StateSpaceResult with fitted values and diagnostics
        """
        assert y is not None, "y must be provided"
        y = np.asarray(y, dtype=np.float64).flatten()
        n = len(y)

        # Initialize model
        self._initialize_matrices(y)

        # Get initial parameters
        initial_params = self._get_initial_parameters()

        # Optimize parameters
        if self.config.optimization_method == OptimizationMethod.EM:
            opt_params, log_lik, converged, n_iter = self._em_algorithm(
                y, initial_params
            )
        else:
            opt_params, log_lik, converged, n_iter = self._optimize_parameters(
                y, initial_params
            )

        # Update matrices with optimal parameters
        self._update_matrices(opt_params)
        self._parameters = self._parameters_to_dict(opt_params)

        # Run Kalman filter and smoother
        filtered_states, filtered_cov, log_lik = self._kalman_filter(y)
        smoothed_states, smoothed_cov = self._kalman_smoother(
            y, filtered_states, filtered_cov
        )

        # Calculate fitted values and residuals
        fitted = np.zeros(n)
        for t in range(n):
            fitted[t] = (self.Z @ smoothed_states[t]).item()

        residuals = y - fitted

        # Calculate diagnostics
        n_params = len(opt_params)
        aic = -2 * log_lik + 2 * n_params
        bic = -2 * log_lik + n_params * np.log(n)
        mse = float(np.mean(residuals**2))
        mae = float(np.mean(np.abs(residuals)))

        self._is_fitted = True

        return StateSpaceResult(
            fitted_values=fitted,
            residuals=residuals,
            filtered_states=filtered_states,
            smoothed_states=smoothed_states,
            filtered_state_cov=filtered_cov,
            smoothed_state_cov=smoothed_cov,
            parameters=self._parameters,
            log_likelihood=log_lik,
            model_type=self.config.model_type,
            n_observations=n,
            n_states=self.n_states,
            aic=aic,
            bic=bic,
            mse=mse,
            mae=mae,
            converged=converged,
            n_iterations=n_iter,
        )

    def forecast(
        self,
        steps: int | None = None,
        confidence_level: float | None = None,
    ) -> ForecastResult:
        """Generate forecasts.

        Args:
            steps: Number of steps ahead to forecast
            confidence_level: Confidence level for intervals

        Returns:
            ForecastResult with forecasts and intervals
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        steps = steps or self.config.forecast_horizon
        confidence_level = confidence_level or self.config.confidence_level

        # Get last filtered state
        # (Assuming _last_state and _last_cov are stored after fitting)
        state = self._last_state.copy()
        state_cov = self._last_cov.copy()

        # Forecast arrays
        forecast = np.zeros(steps)
        forecast_std = np.zeros(steps)
        state_forecast = np.zeros((steps, self.n_states))
        state_forecast_cov = np.zeros((steps, self.n_states, self.n_states))

        # Iterate forward
        for h in range(steps):
            # Predict state
            state = self.T @ state
            state_cov = self.T @ state_cov @ self.T.T + self.R @ self.Q @ self.R.T

            # Predict observation
            forecast[h] = (self.Z @ state).item()
            obs_var = self.Z @ state_cov @ self.Z.T + self.H
            forecast_std[h] = np.sqrt(obs_var[0, 0])

            state_forecast[h] = state.flatten()
            state_forecast_cov[h] = state_cov

        # Compute confidence intervals
        z_score = self._normal_ppf((1 + confidence_level) / 2)
        lower_ci = forecast - z_score * forecast_std
        upper_ci = forecast + z_score * forecast_std

        return ForecastResult(
            forecast=forecast,
            forecast_index=np.arange(steps),
            forecast_std=forecast_std,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            state_forecast=state_forecast,
            state_forecast_cov=state_forecast_cov,
            confidence_level=confidence_level,
        )

    def _kalman_filter(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Run Kalman filter.

        Args:
            y: Observations

        Returns:
            Tuple of (filtered_states, filtered_covariances, log_likelihood)
        """
        assert y is not None, "y must be provided"
        n = len(y)

        # Initialize
        if self.config.initial_state is not None:
            state = self.config.initial_state.copy()
        else:
            state = np.zeros((self.n_states, 1))

        if self.config.initial_state_cov is not None:
            state_cov = self.config.initial_state_cov.copy()
        else:
            # Diffuse initialization
            state_cov = np.eye(self.n_states) * 1e6

        # Storage
        filtered_states = np.zeros((n, self.n_states))
        filtered_cov = np.zeros((n, self.n_states, self.n_states))
        log_likelihood = 0.0

        for t in range(n):
            # Prediction step
            state_pred = self.T @ state
            cov_pred = self.T @ state_cov @ self.T.T + self.R @ self.Q @ self.R.T

            # Innovation
            y_pred = self.Z @ state_pred
            innovation = y[t] - y_pred[0, 0]

            # Innovation covariance
            F = self.Z @ cov_pred @ self.Z.T + self.H

            # Kalman gain
            K = cov_pred @ self.Z.T / F[0, 0]

            # Update step
            state = state_pred + K * innovation
            state_cov = (np.eye(self.n_states) - K @ self.Z) @ cov_pred

            # Store
            filtered_states[t] = state.flatten()
            filtered_cov[t] = state_cov

            # Log-likelihood contribution
            if F[0, 0] > 0:
                log_likelihood -= 0.5 * (
                    np.log(2 * np.pi) + np.log(F[0, 0]) + innovation**2 / F[0, 0]
                )

        # Store last state for forecasting
        self._last_state = state
        self._last_cov = state_cov

        return filtered_states, filtered_cov, log_likelihood

    def _kalman_smoother(
        self,
        y: np.ndarray,
        filtered_states: np.ndarray,
        filtered_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Kalman smoother (backward pass).

        Args:
            y: Observations
            filtered_states: Filtered state estimates
            filtered_cov: Filtered state covariances

        Returns:
            Tuple of (smoothed_states, smoothed_covariances)
        """
        assert y is not None, "y must be provided"
        n = len(y)

        # Initialize with last filtered values
        smoothed_states = filtered_states.copy()
        smoothed_cov = filtered_cov.copy()

        # Backward pass
        for t in range(n - 2, -1, -1):
            # Predicted state at t+1 given t
            state_pred = self.T @ filtered_states[t].reshape(-1, 1)
            cov_pred = self.T @ filtered_cov[t] @ self.T.T + self.R @ self.Q @ self.R.T

            # Smoother gain
            try:
                J = filtered_cov[t] @ self.T.T @ np.linalg.inv(cov_pred)
            except np.linalg.LinAlgError:
                J = filtered_cov[t] @ self.T.T @ np.linalg.pinv(cov_pred)

            # Smoothed estimates
            smoothed_states[t] = filtered_states[t] + J @ (
                smoothed_states[t + 1] - state_pred.flatten()
            )
            smoothed_cov[t] = (
                filtered_cov[t] + J @ (smoothed_cov[t + 1] - cov_pred) @ J.T
            )

        return smoothed_states, smoothed_cov

    def _optimize_parameters(
        self, y: np.ndarray, initial_params: np.ndarray
    ) -> tuple[np.ndarray, float, bool, int]:
        """Optimize parameters using scipy.optimize.minimize.

        Args:
            y: Observations
            initial_params: Initial parameter values

        Returns:
            Tuple of (optimal_params, log_likelihood, converged, n_iterations)
        """

        assert y is not None, "y must be provided"

        def objective(params):
            # Ensure positive variances if needed
            self._update_matrices(params)
            _, _, ll = self._kalman_filter(y)
            return -ll if np.isfinite(ll) else 1e10

        res = minimize(
            objective,
            initial_params,
            method=(
                "BFGS"
                if self.config.optimization_method == OptimizationMethod.BFGS
                else "Nelder-Mead"
            ),
            tol=self.config.tolerance,
            options={"maxiter": self.config.max_iterations},
        )

        # Fallback to Nelder-Mead if BFGS fails
        if not res.success and (
            self.config.optimization_method == OptimizationMethod.BFGS
        ):
            res = minimize(
                objective,
                initial_params,
                method="Nelder-Mead",
                tol=self.config.tolerance,
                options={"maxiter": self.config.max_iterations},
            )

        return res.x, -res.fun, res.success, res.nit

    def _em_algorithm(
        self, y: np.ndarray, initial_params: np.ndarray
    ) -> tuple[np.ndarray, float, bool, int]:
        """EM algorithm for parameter estimation.

        Args:
            y: Observations
            initial_params: Initial parameters

        Returns:
            Tuple of (optimal_params, log_likelihood, converged, n_iterations)
        """
        assert y is not None, "y must be provided"
        params = initial_params.copy()
        max_iter = self.config.max_iterations
        tol = self.config.tolerance

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step: Run Kalman filter and smoother
            self._update_matrices(params)
            filtered_states, filtered_cov, ll = self._kalman_filter(y)
            smoothed_states, smoothed_cov = self._kalman_smoother(
                y, filtered_states, filtered_cov
            )

            # Check convergence
            if abs(ll - prev_ll) < tol:
                return params, ll, True, iteration + 1

            prev_ll = ll

            # M-step: Update parameters
            params = self._em_m_step(y, smoothed_states, smoothed_cov)

        return params, prev_ll, False, max_iter

    def _em_m_step(
        self,
        y: np.ndarray,
        smoothed_states: np.ndarray,
        smoothed_cov: np.ndarray,
    ) -> np.ndarray:
        """M-step of EM algorithm.

        Default implementation - can be overridden by subclasses.
        """
        assert y is not None, "y must be provided"
        n = len(y)

        # Estimate observation variance
        residuals = np.zeros(n)
        for t in range(n):
            residuals[t] = y[t] - (self.Z @ smoothed_states[t]).item()

        obs_var = np.mean(residuals**2)

        # Estimate state variance from smoothed residuals
        state_residuals = np.zeros(n - 1)
        for t in range(1, n):
            pred = self.T @ smoothed_states[t - 1]
            state_residuals[t - 1] = np.sum((smoothed_states[t] - pred) ** 2)

        state_var = np.mean(state_residuals) / self.n_states

        return np.array([max(1e-10, state_var), max(1e-10, obs_var)])

    def _numerical_gradient(
        self, y: np.ndarray, params: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """Compute numerical gradient of log-likelihood."""
        assert y is not None, "y must be provided"
        grad = np.zeros(len(params))

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps

            params_minus = params.copy()
            params_minus[i] -= eps

            self._update_matrices(params_plus)
            _, _, ll_plus = self._kalman_filter(y)

            self._update_matrices(params_minus)
            _, _, ll_minus = self._kalman_filter(y)

            grad[i] = (ll_plus - ll_minus) / (2 * eps)

        return grad

    def _normal_ppf(self, p: float) -> float:
        """Inverse normal CDF."""
        assert p is not None, "p must be provided"
        if p <= 0 or p >= 1:
            return 0.0
        if p < 0.5:
            return -self._normal_ppf(1 - p)

        t = np.sqrt(-2 * np.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)


class LocalLevelModel(BaseStateSpaceModel):
    """Local Level Model (Random Walk plus Noise).

    y_t = mu_t + eps_t,    eps_t ~ N(0, sigma_eps^2)
    mu_t = mu_{t-1} + eta_t,   eta_t ~ N(0, sigma_eta^2)
    """

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        config = config or StateSpaceConfig(model_type=StateSpaceModelType.LOCAL_LEVEL)
        super().__init__(config)
        self.n_states = 1
        self.n_obs = 1

    def _initialize_matrices(self, y: np.ndarray) -> None:
        """Initialize model matrices."""
        assert y is not None, "y must be provided"
        self.T = np.array([[1.0]])
        self.Z = np.array([[1.0]])
        self.R = np.array([[1.0]])
        self.Q = np.array([[np.var(np.diff(y))]])
        self.H = np.array([[np.var(y) * 0.5]])

    def _update_matrices(self, parameters: np.ndarray) -> None:
        """Update Q and H with parameter values."""
        assert parameters is not None, "parameters must be provided"
        self.Q = np.array([[parameters[0] ** 2]])
        self.H = np.array([[parameters[1] ** 2]])

    def _get_initial_parameters(self) -> np.ndarray:
        """Initial parameter estimates."""
        return np.array([np.sqrt(np.abs(self.Q[0, 0])), np.sqrt(np.abs(self.H[0, 0]))])

    def _parameters_to_dict(self, parameters: np.ndarray) -> dict[str, float]:
        """Convert to dictionary."""
        assert parameters is not None, "parameters must be provided"
        return {
            "sigma_eta_sq": float(parameters[0] ** 2),
            "sigma_eps_sq": float(parameters[1] ** 2),
        }


class LocalLinearTrendModel(BaseStateSpaceModel):
    """Local Linear Trend Model.

    y_t = mu_t + eps_t
    mu_t = mu_{t-1} + nu_{t-1} + eta_t
    nu_t = nu_{t-1} + zeta_t
    """

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        config = config or StateSpaceConfig(
            model_type=StateSpaceModelType.LOCAL_LINEAR_TREND
        )
        super().__init__(config)
        self.n_states = 2
        self.n_obs = 1

    def _initialize_matrices(self, y: np.ndarray) -> None:
        """Initialize model matrices."""
        assert y is not None, "y must be provided"
        self.T = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Z = np.array([[1.0, 0.0]])
        self.R = np.eye(2)

        # Initial variance estimates
        var_y = np.var(y)
        var_diff = np.var(np.diff(y))
        var_diff2 = np.var(np.diff(np.diff(y)))

        self.Q = np.diag([var_diff * 0.1, var_diff2 * 0.1])
        self.H = np.array([[var_y * 0.5]])

    def _update_matrices(self, parameters: np.ndarray) -> None:
        """Update matrices with parameter values."""
        assert parameters is not None, "parameters must be provided"
        self.Q = np.array([[parameters[0] ** 2, 0], [0, parameters[1] ** 2]])
        self.H = np.array([[parameters[2] ** 2]])

    def _get_initial_parameters(self) -> np.ndarray:
        """Initial parameter estimates."""
        return np.array(
            [
                np.sqrt(np.abs(self.Q[0, 0])),
                np.sqrt(np.abs(self.Q[1, 1])),
                np.sqrt(np.abs(self.H[0, 0])),
            ]
        )

    def _parameters_to_dict(self, parameters: np.ndarray) -> dict[str, float]:
        """Convert to dictionary."""
        assert parameters is not None, "parameters must be provided"
        return {
            "sigma_eta_sq": float(parameters[0] ** 2),
            "sigma_zeta_sq": float(parameters[1] ** 2),
            "sigma_eps_sq": float(parameters[2] ** 2),
        }


class SeasonalModel(BaseStateSpaceModel):
    """Seasonal State Space Model.

    Includes level, trend, and seasonal components.
    """

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        config = config or StateSpaceConfig(model_type=StateSpaceModelType.SEASONAL)
        super().__init__(config)

        period = config.seasonal_period or 12
        self.period = period
        self.n_states = 2 + period - 1  # Level + trend + seasonal
        self.n_obs = 1

    def _initialize_matrices(self, y: np.ndarray) -> None:
        """Initialize model matrices."""
        assert y is not None, "y must be provided"
        period = self.period

        # Transition matrix
        self.T = np.zeros((self.n_states, self.n_states))
        # Level and trend
        self.T[0, 0] = 1.0
        self.T[0, 1] = 1.0
        self.T[1, 1] = 1.0
        # Seasonal
        self.T[2, 2 : 2 + period - 1] = -1.0
        for i in range(3, 2 + period - 1):
            self.T[i, i - 1] = 1.0

        # Design matrix
        self.Z = np.zeros((1, self.n_states))
        self.Z[0, 0] = 1.0  # Level
        self.Z[0, 2] = 1.0  # Seasonal

        # Selection matrix
        self.R = np.zeros((self.n_states, 3))
        self.R[0, 0] = 1.0  # Level disturbance
        self.R[1, 1] = 1.0  # Trend disturbance
        self.R[2, 2] = 1.0  # Seasonal disturbance

        # Initial variance estimates
        var_y = np.var(y)
        self.Q = np.diag([var_y * 0.1, var_y * 0.01, var_y * 0.1])
        self.H = np.array([[var_y * 0.5]])

    def _update_matrices(self, parameters: np.ndarray) -> None:
        """Update with parameter values."""
        assert parameters is not None, "parameters must be provided"
        self.Q = np.diag([parameters[0], parameters[1], parameters[2]])
        self.H = np.array([[parameters[3]]])

    def _get_initial_parameters(self) -> np.ndarray:
        """Initial parameter estimates."""
        return np.array([self.Q[0, 0], self.Q[1, 1], self.Q[2, 2], self.H[0, 0]])

    def _parameters_to_dict(self, parameters: np.ndarray) -> dict[str, float]:
        """Convert to dictionary."""
        assert parameters is not None, "parameters must be provided"
        return {
            "sigma_level_sq": float(parameters[0]),
            "sigma_trend_sq": float(parameters[1]),
            "sigma_seasonal_sq": float(parameters[2]),
            "sigma_obs_sq": float(parameters[3]),
        }


class ARIMAStateSpace(BaseStateSpaceModel):
    """ARIMA model in state space form.

    ARIMA(p, d, q) represented as a state space model.
    """

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        config = config or StateSpaceConfig(model_type=StateSpaceModelType.ARIMA)
        super().__init__(config)

        p = config.ar_order
        q = config.ma_order
        self.p = p
        self.q = q
        self.d = config.diff_order

        self.n_states = max(p, q + 1)
        self.n_obs = 1

    def _initialize_matrices(self, y: np.ndarray) -> None:
        """Initialize model matrices."""
        assert y is not None, "y must be provided"
        r = self.n_states

        # Difference the series if needed
        y_diff = y.copy()
        for _ in range(self.d):
            y_diff = np.diff(y_diff)

        # Initial AR and MA coefficients
        self._ar_coeffs = np.zeros(self.p)
        self._ma_coeffs = np.zeros(self.q)

        if self.p > 0:
            # Simple AR estimation using Yule-Walker
            self._ar_coeffs = self._estimate_ar(y_diff, self.p)

        # Transition matrix
        self.T = np.zeros((r, r))
        for i in range(min(self.p, r)):
            self.T[0, i] = self._ar_coeffs[i] if i < self.p else 0
        for i in range(1, r):
            self.T[i, i - 1] = 1.0

        # Design matrix
        self.Z = np.zeros((1, r))
        self.Z[0, 0] = 1.0

        # Selection matrix
        self.R = np.zeros((r, 1))
        self.R[0, 0] = 1.0
        for i in range(min(self.q, r - 1)):
            self.R[i + 1, 0] = self._ma_coeffs[i] if i < self.q else 0

        # Variances
        var_y = np.var(y_diff) if len(y_diff) > 0 else 1.0
        self.Q = np.array([[var_y]])
        self.H = np.array([[0.0]])  # Pure ARIMA has no observation noise

    def _update_matrices(self, parameters: np.ndarray) -> None:
        """Update with parameter values."""
        assert parameters is not None, "parameters must be provided"
        idx = 0

        # AR coefficients
        for i in range(self.p):
            self._ar_coeffs[i] = parameters[idx]
            self.T[0, i] = parameters[idx]
            idx += 1

        # MA coefficients
        for i in range(self.q):
            self._ma_coeffs[i] = parameters[idx]
            if i + 1 < self.n_states:
                self.R[i + 1, 0] = parameters[idx]
            idx += 1

        # Variance
        self.Q = np.array([[parameters[idx]]])

    def _get_initial_parameters(self) -> np.ndarray:
        """Initial parameter estimates."""
        params = list(self._ar_coeffs) + list(self._ma_coeffs) + [self.Q[0, 0]]
        return np.array(params)

    def _parameters_to_dict(self, parameters: np.ndarray) -> dict[str, float]:
        """Convert to dictionary."""
        assert parameters is not None, "parameters must be provided"
        result = {}
        idx = 0

        for i in range(self.p):
            result[f"ar_{i + 1}"] = float(parameters[idx])
            idx += 1

        for i in range(self.q):
            result[f"ma_{i + 1}"] = float(parameters[idx])
            idx += 1

        result["sigma_sq"] = float(parameters[idx])
        return result

    def _estimate_ar(self, y: np.ndarray, p: int) -> np.ndarray:
        """Estimate AR coefficients using Yule-Walker equations."""
        assert y is not None, "y must be provided"
        n = len(y)
        if n < p + 1:
            return np.zeros(p)

        # Compute autocorrelations
        acf = np.zeros(p + 1)
        y_centered = y - np.mean(y)
        var = np.var(y_centered)

        if var == 0:
            return np.zeros(p)

        for k in range(p + 1):
            acf[k] = np.sum(y_centered[k:] * y_centered[: n - k]) / ((n - k) * var)

        # Solve Yule-Walker equations
        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = acf[abs(i - j)]

        r = acf[1 : p + 1]

        try:
            ar_coeffs = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            ar_coeffs = np.zeros(p)

        return ar_coeffs


class StateSpaceModelFactory:
    """Factory for creating state space models."""

    @staticmethod
    def create(config: StateSpaceConfig) -> BaseStateSpaceModel:
        """Create a state space model based on configuration.

        Args:
            config: Model configuration

        Returns:
            Appropriate state space model instance
        """
        model_map = {
            StateSpaceModelType.LOCAL_LEVEL: LocalLevelModel,
            StateSpaceModelType.LOCAL_LINEAR_TREND: LocalLinearTrendModel,
            StateSpaceModelType.SEASONAL: SeasonalModel,
            StateSpaceModelType.ARIMA: ARIMAStateSpace,
        }

        model_class = model_map.get(config.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {config.model_type}")

        return model_class(config)


def fit_state_space(
    y: np.ndarray,
    model_type: str = "local_level",
    **kwargs: Any,
) -> StateSpaceResult:
    """Convenience function to fit a state space model.

    Args:
        y: Time series data
        model_type: Model type ('local_level', 'local_linear_trend',
            'seasonal', 'arima')
        **kwargs: Additional configuration options

    Returns:
        StateSpaceResult with fitted model

    Example:
        >>> y = np.cumsum(np.random.randn(100)) + np.random.randn(100) * 0.5
        >>> result = fit_state_space(y, model_type='local_level')
        >>> print(f"AIC: {result.aic:.2f}")
    """
    assert y is not None, "y must be provided"
    type_map = {
        "local_level": StateSpaceModelType.LOCAL_LEVEL,
        "local_linear_trend": StateSpaceModelType.LOCAL_LINEAR_TREND,
        "seasonal": StateSpaceModelType.SEASONAL,
        "arima": StateSpaceModelType.ARIMA,
    }

    ss_type = type_map.get(model_type.lower(), StateSpaceModelType.LOCAL_LEVEL)
    config = StateSpaceConfig(model_type=ss_type, **kwargs)

    model = StateSpaceModelFactory.create(config)
    return model.fit(y)


__all__ = [
    "StateSpaceModelType",
    "OptimizationMethod",
    "StateSpaceConfig",
    "StateSpaceResult",
    "ForecastResult",
    "BaseStateSpaceModel",
    "LocalLevelModel",
    "LocalLinearTrendModel",
    "SeasonalModel",
    "ARIMAStateSpace",
    "StateSpaceModelFactory",
    "fit_state_space",
]
