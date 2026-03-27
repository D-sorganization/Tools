from numba import jit

"""Kalman Filtering Module.

Provides Kalman filtering for optimal state estimation in time-series data.
Excellent for:
- Sensor fusion in robotics
- Tracking with measurement noise
- Handling missing data naturally
- Systems with known dynamics

Includes:
- Standard Kalman Filter
- Extended Kalman Filter (EKF) for nonlinear systems
- Unscented Kalman Filter (UKF) for highly nonlinear systems
- Kalman Smoother for offline processing
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KalmanFilterType(Enum):
    """Types of Kalman filters available."""

    STANDARD = "standard"
    EXTENDED = "extended"
    UNSCENTED = "unscented"


class KalmanFilterConfig:
    """Configuration for Kalman filter."""

    # State dimension
    state_dim: int = 1

    # Measurement dimension
    measurement_dim: int = 1

    def __init__(self, **kwargs) -> None:
        if "obs_dim" in kwargs:
            self.measurement_dim = kwargs.pop("obs_dim")

        # Set all passed values
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Process noise covariance (Q)
    process_noise: float | np.ndarray = 1.0

    # Measurement noise covariance (R)
    measurement_noise: float | np.ndarray = 1.0

    # Initial state estimate
    initial_state: np.ndarray | None = None

    # Initial covariance estimate
    initial_covariance: float | np.ndarray = 1.0

    # State transition matrix (A/F) - for standard KF
    state_transition: np.ndarray | None = None

    # Measurement matrix (H) - for standard KF
    measurement_matrix: np.ndarray | None = None

    # Control input matrix (B) - optional
    control_matrix: np.ndarray | None = None

    # Filter type
    filter_type: KalmanFilterType = KalmanFilterType.STANDARD

    # UKF specific parameters
    ukf_alpha: float = 0.001
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0

    @property
    def obs_dim(self) -> int:
        """Alias for measurement_dim for backward compatibility."""
        return self.measurement_dim

    @obs_dim.setter
    def obs_dim(self, value: int) -> None:
        self.measurement_dim = value


@dataclass
class KalmanFilterResult:
    """Results from Kalman filtering."""

    # Filtered states (posterior estimates)
    filtered_states: np.ndarray

    # State covariances
    filtered_covariances: np.ndarray

    # Predicted states (prior estimates)
    predicted_states: np.ndarray

    # Kalman gains
    kalman_gains: np.ndarray

    # Innovation (measurement residual)
    innovations: np.ndarray

    # Innovation covariance
    innovation_covariances: np.ndarray

    # Log likelihood (for model comparison)
    log_likelihood: float

    # Smoothed states (if smoother was run)
    smoothed_states: np.ndarray | None = None
    smoothed_covariances: np.ndarray | None = None


class KalmanFilter:
    """Standard Kalman Filter implementation.

    Optimal linear estimator for systems of the form:
        x[k] = A * x[k-1] + B * u[k] + w[k]  (state equation)
        z[k] = H * x[k] + v[k]                (measurement equation)

    where w ~ N(0, Q) and v ~ N(0, R)
    """

    def __init__(self, config: KalmanFilterConfig) -> None:
        """Initialize the Kalman filter.

        Args:
            config: Filter configuration
        """
        if not (config is not None):
            raise ValueError("config must be provided")
        self.config = config
        self._initialize_matrices()

    def _initialize_matrices(self) -> None:
        """Initialize filter matrices."""
        n = self.config.state_dim
        m = self.config.measurement_dim

        # State transition matrix (default: identity for random walk)
        if self.config.state_transition is not None:
            self.A = np.asarray(self.config.state_transition)
        else:
            self.A = np.eye(n)

        # Measurement matrix (default: observe all states)
        if self.config.measurement_matrix is not None:
            self.H = np.asarray(self.config.measurement_matrix)
        else:
            self.H = np.eye(m, n)

        # Process noise covariance
        if isinstance(self.config.process_noise, (int, float)):
            self.Q = np.eye(n) * self.config.process_noise
        else:
            self.Q = np.asarray(self.config.process_noise)

        # Measurement noise covariance
        if isinstance(self.config.measurement_noise, (int, float)):
            self.R = np.eye(m) * self.config.measurement_noise
        else:
            self.R = np.asarray(self.config.measurement_noise)

        # Control matrix (optional)
        self.B = None
        if self.config.control_matrix is not None:
            self.B = np.asarray(self.config.control_matrix)

        # Initial state
        if self.config.initial_state is not None:
            self.x0 = np.asarray(self.config.initial_state).reshape(-1)
        else:
            self.x0 = np.zeros(n)

        # Initial covariance
        if isinstance(self.config.initial_covariance, (int, float)):
            self.P0 = np.eye(n) * self.config.initial_covariance
        else:
            self.P0 = np.asarray(self.config.initial_covariance)

    def set_transition_matrix(self, A: np.ndarray) -> None:
        """Set the state transition matrix."""
        self.A = np.asarray(A)

    def set_observation_matrix(self, H: np.ndarray) -> None:
        """Set the measurement/observation matrix."""
        self.H = np.asarray(H)

    def set_process_noise(self, Q: np.ndarray | float) -> None:
        """Set the process noise covariance."""
        if isinstance(Q, (int, float)):
            self.Q = np.eye(self.config.state_dim) * Q
        else:
            self.Q = np.asarray(Q)

    def set_observation_noise(self, R: np.ndarray | float) -> None:
        """Set the measurement/observation noise covariance."""
        if isinstance(R, (int, float)):
            self.R = np.eye(self.config.measurement_dim) * R
        else:
            self.R = np.asarray(R)

    @jit(nopython=True, fastmath=True)
    def filter(
        self,
        measurements: np.ndarray,
        control_inputs: np.ndarray | None = None,
    ) -> KalmanFilterResult:
        """Run the Kalman filter on measurements.

        Args:
            measurements: Array of measurements (T x m)
            control_inputs: Optional control inputs (T x p)

        Returns:
            KalmanFilterResult with filtered states and diagnostics
        """
        if not (measurements is not None):
            raise ValueError("measurements must be provided")
        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, self.config.measurement_dim)
        elif (
            measurements.ndim == 2
            and measurements.shape[1] != self.config.measurement_dim
            and measurements.shape[0] == self.config.measurement_dim
        ):
            measurements = measurements.T

        T = measurements.shape[0]
        n = self.config.state_dim
        m = self.config.measurement_dim

        # Storage
        filtered_states = np.zeros((T, n))
        filtered_covariances = np.zeros((T, n, n))
        predicted_states = np.zeros((T, n))
        kalman_gains = np.zeros((T, n, m))
        innovations = np.zeros((T, m))
        innovation_covariances = np.zeros((T, m, m))

        # Initialize
        x = self.x0.copy()
        P = self.P0.copy()
        log_likelihood = 0.0

        for t in range(T):
            # Predict
            x_pred = self.A @ x
            if self.B is not None and control_inputs is not None:
                x_pred += self.B @ control_inputs[t]
            P_pred = self.A @ P @ self.A.T + self.Q

            predicted_states[t] = x_pred

            # Update (if measurement is not NaN)
            z = measurements[t]
            if not np.any(np.isnan(z)):
                # Innovation
                y = z - self.H @ x_pred
                S = self.H @ P_pred @ self.H.T + self.R

                # Kalman gain
                K = P_pred @ self.H.T @ np.linalg.inv(S)

                # Update
                x = x_pred + K @ y
                P = (np.eye(n) - K @ self.H) @ P_pred

                # Store innovation info
                innovations[t] = y
                innovation_covariances[t] = S
                kalman_gains[t] = K

                # Log likelihood
                log_likelihood += self._log_likelihood_contribution(y, S)
            else:
                # No measurement - use prediction
                x = x_pred
                P = P_pred
                innovations[t] = np.nan
                innovation_covariances[t] = np.nan

            filtered_states[t] = x
            filtered_covariances[t] = P

        return KalmanFilterResult(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            kalman_gains=kalman_gains,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            log_likelihood=log_likelihood,
        )

    @jit(nopython=True, fastmath=True)
    def smooth(self, filter_result: KalmanFilterResult) -> KalmanFilterResult:
        """Run Rauch-Tung-Striebel smoother for offline processing.

        The smoother uses future measurements to improve past estimates.

        Args:
            filter_result: Result from forward filter pass

        Returns:
            Updated result with smoothed estimates
        """
        if not (filter_result is not None):
            raise ValueError("filter_result must be provided")
        T = filter_result.filtered_states.shape[0]
        n = self.config.state_dim

        smoothed_states = np.zeros((T, n))
        smoothed_covariances = np.zeros((T, n, n))

        # Initialize with final filtered estimate
        smoothed_states[-1] = filter_result.filtered_states[-1]
        smoothed_covariances[-1] = filter_result.filtered_covariances[-1]

        # Backward pass
        for t in range(T - 2, -1, -1):
            x_filt = filter_result.filtered_states[t]
            P_filt = filter_result.filtered_covariances[t]

            # Predicted state at t+1 from state at t
            x_pred = self.A @ x_filt
            P_pred = self.A @ P_filt @ self.A.T + self.Q

            # Smoother gain
            J = P_filt @ self.A.T @ np.linalg.inv(P_pred)

            # Smooth
            smoothed_states[t] = x_filt + J @ (smoothed_states[t + 1] - x_pred)
            smoothed_covariances[t] = P_filt + J @ (smoothed_covariances[t + 1] - P_pred) @ J.T

        # Update result
        return KalmanFilterResult(
            filtered_states=filter_result.filtered_states,
            filtered_covariances=filter_result.filtered_covariances,
            predicted_states=filter_result.predicted_states,
            kalman_gains=filter_result.kalman_gains,
            innovations=filter_result.innovations,
            innovation_covariances=filter_result.innovation_covariances,
            log_likelihood=filter_result.log_likelihood,
            smoothed_states=smoothed_states,
            smoothed_covariances=smoothed_covariances,
        )

    def _log_likelihood_contribution(
        self,
        innovation: np.ndarray,
        innovation_cov: np.ndarray,
    ) -> float:
        """Calculate log likelihood contribution from one observation."""
        if not (innovation is not None):
            raise ValueError("innovation must be provided")
        m = len(innovation)
        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            return -np.inf
        mahalanobis = innovation @ np.linalg.inv(innovation_cov) @ innovation
        return -0.5 * (m * np.log(2 * np.pi) + logdet + mahalanobis)


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems.

    Uses linearization (Jacobians) at each time step for systems:
        x[k] = f(x[k-1], u[k]) + w[k]  (nonlinear state transition)
        z[k] = h(x[k]) + v[k]          (nonlinear measurement)
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int | None = None,
        f: Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None = None,
        h: Callable[[np.ndarray], np.ndarray] | None = None,
        F_jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
        H_jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        obs_dim: int | None = None,
    ) -> None:
        """Initialize EKF."""
        if not (state_dim is not None):
            raise ValueError("state_dim must be provided")
        self.n = state_dim
        self.m = measurement_dim or obs_dim or state_dim
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q if Q is not None else np.eye(self.n)
        self.R = R if R is not None else np.eye(self.m)
        self.x0 = x0 if x0 is not None else np.zeros(self.n)
        self.P0 = P0 if P0 is not None else np.eye(self.n)

    @jit(nopython=True, fastmath=True)
    def filter(
        self,
        measurements: np.ndarray,
        control_inputs: np.ndarray | None = None,
        transition_func: Callable | None = None,
        observation_func: Callable | None = None,
        transition_jacobian: Callable | None = None,
        observation_jacobian: Callable | None = None,
    ) -> KalmanFilterResult:
        """Run EKF on measurements."""
        if not (measurements is not None):
            raise ValueError("measurements must be provided")
        f = transition_func or self.f
        h = observation_func or self.h
        F_jac = transition_jacobian or self.F_jacobian
        H_jac = observation_jacobian or self.H_jacobian

        if f is None or h is None:
            raise ValueError("Transition and observation functions must be provided")

        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, self.m)
        elif (
            measurements.ndim == 2
            and measurements.shape[1] != self.m
            and measurements.shape[0] == self.m
        ):
            measurements = measurements.T

        T = measurements.shape[0]

        filtered_states = np.zeros((T, self.n))
        filtered_covariances = np.zeros((T, self.n, self.n))
        predicted_states = np.zeros((T, self.n))
        kalman_gains = np.zeros((T, self.n, self.m))
        innovations = np.zeros((T, self.m))
        innovation_covariances = np.zeros((T, self.m, self.m))

        x = self.x0.copy()
        P = self.P0.copy()
        log_likelihood = 0.0

        for t in range(T):
            u = control_inputs[t] if control_inputs is not None else None

            # Predict
            x_pred = f(x, u) if u is not None else f(x)
            F = F_jac(x) if F_jac else np.eye(self.n)
            P_pred = F @ P @ F.T + self.Q

            predicted_states[t] = x_pred

            # Update
            z = measurements[t]
            if not np.any(np.isnan(z)):
                H = H_jac(x_pred) if H_jac else np.eye(self.m, self.n)
                y = z - h(x_pred)
                S = H @ P_pred @ H.T + self.R
                K = P_pred @ H.T @ np.linalg.inv(S)

                x = x_pred + K @ y
                P = (np.eye(self.n) - K @ H) @ P_pred

                innovations[t] = y
                innovation_covariances[t] = S
                kalman_gains[t] = K

                sign, logdet = np.linalg.slogdet(S)
                if sign > 0:
                    log_likelihood += -0.5 * (
                        self.m * np.log(2 * np.pi) + logdet + y @ np.linalg.inv(S) @ y
                    )
            else:
                x = x_pred
                P = P_pred

            filtered_states[t] = x
            filtered_covariances[t] = P

        return KalmanFilterResult(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            kalman_gains=kalman_gains,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            log_likelihood=log_likelihood,
        )


class UnscentedKalmanFilter:
    """Unscented Kalman Filter for highly nonlinear systems.

    Uses sigma points to propagate mean and covariance through
    nonlinear functions without requiring Jacobians.
    """

    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        f: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        """Initialize UKF."""
        if not (state_dim is not None):
            raise ValueError("state_dim must be provided")
        self.n = state_dim
        self.m = measurement_dim
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.x0 = x0 if x0 is not None else np.zeros(state_dim)
        self.P0 = P0 if P0 is not None else np.eye(state_dim)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim

        # Weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        n = self.n
        lambda_ = self.lambda_

        # Mean weights
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))

        # Covariance weights
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))

    @jit(nopython=True, fastmath=True)
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points."""
        if not (x is not None):
            raise ValueError("x must be provided")
        n = self.n
        lambda_ = self.lambda_

        sigma_pts = np.zeros((2 * n + 1, n))
        sigma_pts[0] = x

        sqrt_P = np.linalg.cholesky((n + lambda_) * P)

        for i in range(n):
            sigma_pts[i + 1] = x + sqrt_P[:, i]
            sigma_pts[n + i + 1] = x - sqrt_P[:, i]

        return sigma_pts

    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    @jit(nopython=True, fastmath=True)
    def filter(
        self,
        measurements: np.ndarray,
        control_inputs: np.ndarray | None = None,
    ) -> KalmanFilterResult:
        """Run UKF on measurements."""
        if not (measurements is not None):
            raise ValueError("measurements must be provided")
        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, self.m)
        elif (
            measurements.ndim == 2
            and measurements.shape[1] != self.m
            and measurements.shape[0] == self.m
        ):
            measurements = measurements.T

        T = measurements.shape[0]

        filtered_states = np.zeros((T, self.n))
        filtered_covariances = np.zeros((T, self.n, self.n))
        predicted_states = np.zeros((T, self.n))
        kalman_gains = np.zeros((T, self.n, self.m))
        innovations = np.zeros((T, self.m))
        innovation_covariances = np.zeros((T, self.m, self.m))

        x = self.x0.copy()
        P = self.P0.copy()
        log_likelihood = 0.0

        for t in range(T):
            u = control_inputs[t] if control_inputs is not None else None

            # Generate sigma points
            sigma_pts = self._sigma_points(x, P)

            # Predict sigma points through state transition
            sigma_pts_pred = np.array([self.f(sp, u) for sp in sigma_pts])

            # Predicted mean and covariance
            x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_pts_pred, axis=0)
            P_pred = self.Q.copy()
            for i, sp in enumerate(sigma_pts_pred):
                diff = sp - x_pred
                P_pred += self.Wc[i] * np.outer(diff, diff)

            predicted_states[t] = x_pred

            # Update
            z = measurements[t]
            if not np.any(np.isnan(z)):
                # Sigma points through measurement function
                sigma_pts_meas = np.array([self.h(sp) for sp in sigma_pts_pred])

                # Predicted measurement mean and covariance
                z_pred = np.sum(self.Wm[:, np.newaxis] * sigma_pts_meas, axis=0)
                Pzz = self.R.copy()
                Pxz = np.zeros((self.n, self.m))

                for i in range(len(sigma_pts_pred)):
                    z_diff = sigma_pts_meas[i] - z_pred
                    x_diff = sigma_pts_pred[i] - x_pred
                    Pzz += self.Wc[i] * np.outer(z_diff, z_diff)
                    Pxz += self.Wc[i] * np.outer(x_diff, z_diff)

                # Kalman gain and update
                K = Pxz @ np.linalg.inv(Pzz)
                y = z - z_pred
                x = x_pred + K @ y
                P = P_pred - K @ Pzz @ K.T

                innovations[t] = y
                innovation_covariances[t] = Pzz
                kalman_gains[t] = K

                sign, logdet = np.linalg.slogdet(Pzz)
                if sign > 0:
                    log_likelihood += -0.5 * (
                        self.m * np.log(2 * np.pi) + logdet + y @ np.linalg.inv(Pzz) @ y
                    )
            else:
                x = x_pred
                P = P_pred

            filtered_states[t] = x
            filtered_covariances[t] = P

        return KalmanFilterResult(
            filtered_states=filtered_states,
            filtered_covariances=filtered_covariances,
            predicted_states=predicted_states,
            kalman_gains=kalman_gains,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            log_likelihood=log_likelihood,
        )


def apply_kalman_filter(
    df: pd.DataFrame,
    signal_column: str,
    process_noise: float = 1.0,
    measurement_noise: float = 1.0,
    smooth: bool = True,
) -> pd.DataFrame:
    """Apply Kalman filter to a signal in a DataFrame.

    Convenience function for simple 1D filtering.

    Args:
        df: DataFrame with signal
        signal_column: Column to filter
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
        smooth: Whether to apply RTS smoother

    Returns:
        DataFrame with filtered signal columns added
    """
    if not (df is not None):
        raise ValueError("df must be provided")
    config = KalmanFilterConfig(
        state_dim=1,
        measurement_dim=1,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    kf = KalmanFilter(config)
    measurements = df[signal_column].values.reshape(-1, 1)

    result = kf.filter(measurements)
    if smooth:
        result = kf.smooth(result)

    output_df = df.copy()
    output_df[f"{signal_column}_kf_filtered"] = result.filtered_states.flatten()

    if result.smoothed_states is not None:
        output_df[f"{signal_column}_kf_smoothed"] = result.smoothed_states.flatten()

    # Add confidence intervals (2 sigma)
    std = np.sqrt(result.filtered_covariances[:, 0, 0])
    output_df[f"{signal_column}_kf_std"] = std
    output_df[f"{signal_column}_kf_lower"] = result.filtered_states.flatten() - 2 * std
    output_df[f"{signal_column}_kf_upper"] = result.filtered_states.flatten() + 2 * std

    return output_df


def kalman_smooth(
    signal: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.25,
) -> np.ndarray:
    """Convenience function for Kalman smoothing of a 1D signal.

    Args:
        signal: 1D array of measurements
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance

    Returns:
        Smoothed 1D signal
    """
    if not (signal is not None):
        raise ValueError("signal must be provided")
    signal = np.asarray(signal).flatten()
    n = len(signal)

    config = KalmanFilterConfig(
        state_dim=1,
        measurement_dim=1,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_state=np.array([signal[0]]) if n > 0 else None,
        initial_covariance=1.0,
    )

    kf = KalmanFilter(config)
    result = kf.filter(signal.reshape(-1, 1))
    result = kf.smooth(result)

    if result.smoothed_states is not None:
        return result.smoothed_states.flatten()
    return result.filtered_states.flatten()


def estimate_kalman_params(
    signal: np.ndarray,
    method: str = "innovation",
) -> tuple[float, float]:
    """Estimate optimal Kalman filter parameters from data.

    Args:
        signal: Signal to estimate parameters for
        method: Estimation method ("innovation" or "em")

    Returns:
        Tuple of (process_noise, measurement_noise) estimates
    """
    # Simple innovation-based estimation
    if not (signal is not None):
        raise ValueError("signal must be provided")
    signal = np.asarray(signal).flatten()
    signal = signal[~np.isnan(signal)]

    if len(signal) < 10:
        return 1.0, 1.0

    # Estimate measurement noise from high-frequency variation
    diff1 = np.diff(signal)
    measurement_noise = np.var(diff1) / 2

    # Estimate process noise from smoother variation
    diff2 = np.diff(signal, n=2)
    process_noise = np.var(diff2) / 4

    # Ensure positive
    return max(process_noise, 1e-6), max(measurement_noise, 1e-6)


__all__ = [
    "KalmanFilterType",
    "KalmanFilterConfig",
    "KalmanFilterResult",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "apply_kalman_filter",
    "kalman_smooth",
    "estimate_kalman_params",
]
