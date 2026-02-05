"""Uncertainty Quantification Module.

Provides comprehensive uncertainty quantification capabilities for
data analysis, including confidence intervals, prediction intervals,
bootstrap methods, and Bayesian approaches.

Features:
- Bootstrap confidence intervals (percentile, BCa, studentized)
- Monte Carlo uncertainty propagation
- Bayesian credible intervals
- Prediction intervals for regression
- Sensitivity analysis
- Error propagation
- Confidence bands for fitted curves
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BootstrapMethod(Enum):
    """Bootstrap confidence interval methods."""

    PERCENTILE = "percentile"
    BASIC = "basic"
    BCA = "bca"  # Bias-corrected and accelerated
    STUDENTIZED = "studentized"


class UncertaintyMethod(Enum):
    """Uncertainty quantification methods."""

    BOOTSTRAP = "bootstrap"
    MONTE_CARLO = "monte_carlo"
    BAYESIAN = "bayesian"
    ANALYTICAL = "analytical"
    DELTA_METHOD = "delta_method"


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""

    # Bootstrap settings
    n_bootstrap: int = 1000
    bootstrap_method: BootstrapMethod = BootstrapMethod.BCA
    random_seed: int | None = None

    # Monte Carlo settings
    n_monte_carlo: int = 10000

    # Confidence level
    confidence_level: float = 0.95

    # Bayesian settings
    n_posterior_samples: int = 5000
    prior_std: float = 1.0

    # Delta method
    delta_method_step: float = 1e-6


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""

    lower: float
    upper: float
    point_estimate: float
    confidence_level: float
    method: str

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        """Margin of error (half width)."""
        return self.width / 2

    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower <= value <= self.upper


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""

    # Point estimate
    point_estimate: float
    standard_error: float

    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Bootstrap distribution
    bootstrap_samples: np.ndarray
    bootstrap_statistics: np.ndarray

    # Bias and acceleration (for BCa)
    bias: float = 0.0
    acceleration: float = 0.0

    # Method used
    method: BootstrapMethod = BootstrapMethod.PERCENTILE


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo uncertainty propagation."""

    # Central estimates
    mean: float
    median: float
    std: float

    # Confidence interval
    ci_lower: float
    ci_upper: float
    confidence_level: float

    # Distribution
    samples: np.ndarray
    percentiles: dict[int, float]

    # Higher moments
    skewness: float = 0.0
    kurtosis: float = 0.0


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    # Sensitivity indices
    first_order: dict[str, float]
    total_order: dict[str, float]

    # Interactions (second-order)
    interactions: dict[str, float]

    # Most influential parameters
    ranking: list[str]

    # Variance decomposition
    variance_explained: dict[str, float]


@dataclass
class PredictionInterval:
    """Prediction interval for regression."""

    # Predictions
    predicted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    # Confidence bands for mean
    mean_lower: np.ndarray
    mean_upper: np.ndarray

    # Confidence level
    confidence_level: float

    # Residual standard error
    residual_std: float


class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification engine.

    Provides methods for bootstrap analysis, Monte Carlo propagation,
    and analytical uncertainty estimation.
    """

    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        """Initialize the quantifier.

        Args:
            config: Configuration options
        """
        self.config = config or UncertaintyConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def bootstrap_ci(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        method: BootstrapMethod | None = None,
    ) -> BootstrapResult:
        """Compute bootstrap confidence interval.

        Args:
            data: Input data array
            statistic: Function that computes the statistic of interest
            method: Bootstrap method to use

        Returns:
            BootstrapResult with confidence interval and distribution
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        method = method or self.config.bootstrap_method

        # Point estimate
        theta_hat = statistic(data)

        # Generate bootstrap samples
        bootstrap_stats = np.zeros(self.config.n_bootstrap)
        bootstrap_samples = np.zeros((self.config.n_bootstrap, n))

        for i in range(self.config.n_bootstrap):
            # Resample with replacement
            indices = self._rng.integers(0, n, size=n)
            bootstrap_samples[i] = data[indices]
            bootstrap_stats[i] = statistic(data[indices])

        # Standard error
        se = np.std(bootstrap_stats, ddof=1)

        # Compute CI based on method
        alpha = 1 - self.config.confidence_level

        if method == BootstrapMethod.PERCENTILE:
            ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            bias = 0.0
            acceleration = 0.0

        elif method == BootstrapMethod.BASIC:
            q_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            q_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            ci_lower = 2 * theta_hat - q_upper
            ci_upper = 2 * theta_hat - q_lower
            bias = 0.0
            acceleration = 0.0

        elif method == BootstrapMethod.BCA:
            ci_lower, ci_upper, bias, acceleration = self._bca_interval(
                data, statistic, bootstrap_stats, theta_hat, alpha
            )

        elif method == BootstrapMethod.STUDENTIZED:
            ci_lower, ci_upper = self._studentized_interval(
                data,
                statistic,
                bootstrap_samples,
                bootstrap_stats,
                theta_hat,
                se,
                alpha,
            )
            bias = 0.0
            acceleration = 0.0

        else:
            raise ValueError(f"Unknown bootstrap method: {method}")

        return BootstrapResult(
            point_estimate=theta_hat,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            bootstrap_samples=bootstrap_samples,
            bootstrap_statistics=bootstrap_stats,
            bias=bias,
            acceleration=acceleration,
            method=method,
        )

    def monte_carlo_propagation(
        self,
        func: Callable[..., float],
        param_distributions: dict[str, tuple[str, dict[str, Any]]],
    ) -> MonteCarloResult:
        """Propagate uncertainty through a function using Monte Carlo.

        Args:
            func: Function to propagate uncertainty through
            param_distributions: Dictionary mapping parameter names to
                (distribution_type, distribution_params) tuples

        Returns:
            MonteCarloResult with propagated uncertainty

        Example:
            >>> def f(x, y): return x * y
            >>> result = uq.monte_carlo_propagation(
            ...     f,
            ...     {'x': ('normal', {'loc': 5, 'scale': 0.5}),
            ...      'y': ('uniform', {'low': 1, 'high': 3})}
            ... )
        """
        n_samples = self.config.n_monte_carlo

        # Generate samples for each parameter
        param_samples = {}
        for name, (dist_type, params) in param_distributions.items():
            param_samples[name] = self._sample_distribution(
                dist_type, params, n_samples
            )

        # Evaluate function for each sample
        outputs = np.zeros(n_samples)
        for i in range(n_samples):
            kwargs = {name: samples[i] for name, samples in param_samples.items()}
            outputs[i] = func(**kwargs)

        # Compute statistics
        alpha = 1 - self.config.confidence_level

        mean = float(np.mean(outputs))
        median = float(np.median(outputs))
        std = float(np.std(outputs, ddof=1))

        ci_lower = float(np.percentile(outputs, 100 * alpha / 2))
        ci_upper = float(np.percentile(outputs, 100 * (1 - alpha / 2)))

        # Percentiles
        percentiles = {
            p: float(np.percentile(outputs, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        # Higher moments
        skewness = float(self._skewness(outputs))
        kurtosis = float(self._kurtosis(outputs))

        return MonteCarloResult(
            mean=mean,
            median=median,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            samples=outputs,
            percentiles=percentiles,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def error_propagation(
        self,
        func: Callable[..., float],
        values: dict[str, float],
        uncertainties: dict[str, float],
        method: str = "linear",
    ) -> tuple[float, float]:
        """Propagate errors through a function.

        Args:
            func: Function to propagate errors through
            values: Dictionary of parameter central values
            uncertainties: Dictionary of parameter uncertainties (std)
            method: 'linear' for linear approximation, 'quadrature' for quadrature

        Returns:
            Tuple of (result, uncertainty)
        """
        # Compute central value
        central = func(**values)

        if method == "linear":
            # Linear error propagation using partial derivatives
            variance = 0.0

            for name, unc in uncertainties.items():
                # Numerical partial derivative
                deriv = self._numerical_derivative(func, values, name)
                variance += (deriv * unc) ** 2

            return central, np.sqrt(variance)

        elif method == "quadrature":
            # Add in quadrature (assumes independent errors)
            variance = 0.0

            for name, unc in uncertainties.items():
                deriv = self._numerical_derivative(func, values, name)
                variance += (deriv * unc) ** 2

            return central, np.sqrt(variance)

        else:
            # Use Monte Carlo for nonlinear case
            distributions = {
                name: ("normal", {"loc": val, "scale": uncertainties.get(name, 0)})
                for name, val in values.items()
            }
            result = self.monte_carlo_propagation(func, distributions)
            return result.mean, result.std

    def sensitivity_analysis(
        self,
        func: Callable[..., float],
        param_bounds: dict[str, tuple[float, float]],
        n_samples: int | None = None,
    ) -> SensitivityResult:
        """Perform global sensitivity analysis using Sobol indices.

        Args:
            func: Function to analyze
            param_bounds: Dictionary mapping parameters to (min, max) bounds
            n_samples: Number of samples for analysis

        Returns:
            SensitivityResult with sensitivity indices
        """
        n_samples = n_samples or self.config.n_monte_carlo // 10
        params = list(param_bounds.keys())
        n_params = len(params)

        # Generate Sobol samples
        A = self._sobol_sample(n_samples, n_params, param_bounds, params)
        B = self._sobol_sample(n_samples, n_params, param_bounds, params)

        # Evaluate model on A and B
        y_A = np.array([func(**dict(zip(params, row, strict=False))) for row in A])
        y_B = np.array([func(**dict(zip(params, row, strict=False))) for row in B])

        total_variance = np.var(np.concatenate([y_A, y_B]))

        # First-order and total-order indices
        first_order = {}
        total_order = {}

        for i, param in enumerate(params):
            # AB_i: A with i-th column replaced by B
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]

            y_AB_i = np.array(
                [func(**dict(zip(params, row, strict=False))) for row in AB_i]
            )

            # First-order index
            if total_variance > 0:
                first_order[param] = float(
                    np.mean(y_B * (y_AB_i - y_A)) / total_variance
                )

                # Total-order index
                total_order[param] = float(
                    0.5 * np.mean((y_A - y_AB_i) ** 2) / total_variance
                )
            else:
                first_order[param] = 0.0
                total_order[param] = 0.0

        # Interactions (simplified - just residual from first-order)
        interactions = {}
        for param in params:
            interactions[param] = max(0, total_order[param] - first_order[param])

        # Ranking
        ranking = sorted(params, key=lambda p: total_order[p], reverse=True)

        # Variance explained
        variance_explained = {param: first_order[param] for param in params}

        return SensitivityResult(
            first_order=first_order,
            total_order=total_order,
            interactions=interactions,
            ranking=ranking,
            variance_explained=variance_explained,
        )

    def prediction_intervals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_new: np.ndarray,
        model_type: str = "linear",
    ) -> PredictionInterval:
        """Compute prediction intervals for regression.

        Args:
            X: Training features
            y: Training targets
            X_new: New features for prediction
            model_type: Type of model ('linear', 'polynomial')

        Returns:
            PredictionInterval with predictions and intervals
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X_new = np.asarray(X_new, dtype=np.float64)

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        n, p = X.shape
        n_new = X_new.shape[0]

        # Add intercept
        X_design = np.column_stack([np.ones(n), X])
        X_new_design = np.column_stack([np.ones(n_new), X_new])

        # Fit linear model
        try:
            beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X_design) @ y

        # Predictions
        y_pred = X_design @ beta
        y_new_pred = X_new_design @ beta

        # Residuals and MSE
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)
        residual_std = np.sqrt(mse)

        # Covariance matrix of beta
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_design.T @ X_design)

        # t-value for confidence level
        alpha = 1 - self.config.confidence_level
        t_val = self._t_ppf(1 - alpha / 2, n - p - 1)

        # Standard errors and intervals
        se_mean = np.zeros(n_new)
        se_pred = np.zeros(n_new)

        for i in range(n_new):
            x_i = X_new_design[i : i + 1, :]
            var_mean = x_i @ XtX_inv @ x_i.T
            se_mean[i] = np.sqrt(mse * var_mean[0, 0])
            se_pred[i] = np.sqrt(mse * (1 + var_mean[0, 0]))

        # Intervals
        mean_lower = y_new_pred - t_val * se_mean
        mean_upper = y_new_pred + t_val * se_mean
        pred_lower = y_new_pred - t_val * se_pred
        pred_upper = y_new_pred + t_val * se_pred

        return PredictionInterval(
            predicted=y_new_pred,
            lower=pred_lower,
            upper=pred_upper,
            mean_lower=mean_lower,
            mean_upper=mean_upper,
            confidence_level=self.config.confidence_level,
            residual_std=residual_std,
        )

    def bayesian_credible_interval(
        self,
        data: np.ndarray,
        prior_mean: float = 0.0,
        prior_std: float | None = None,
    ) -> ConfidenceInterval:
        """Compute Bayesian credible interval for mean.

        Uses conjugate normal-normal model.

        Args:
            data: Observed data
            prior_mean: Prior mean for the parameter
            prior_std: Prior standard deviation

        Returns:
            ConfidenceInterval (credible interval)
        """
        data = np.asarray(data, dtype=np.float64)
        n = len(data)

        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)

        prior_std = prior_std or self.config.prior_std
        prior_var = prior_std**2

        # Posterior parameters (conjugate update)
        posterior_var = 1 / (1 / prior_var + n / sample_var)
        posterior_mean = posterior_var * (
            prior_mean / prior_var + n * sample_mean / sample_var
        )
        posterior_std = np.sqrt(posterior_var)

        # Credible interval
        alpha = 1 - self.config.confidence_level
        z = self._normal_ppf(1 - alpha / 2)

        lower = posterior_mean - z * posterior_std
        upper = posterior_mean + z * posterior_std

        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            point_estimate=float(posterior_mean),
            confidence_level=self.config.confidence_level,
            method="bayesian",
        )

    def delta_method_ci(
        self,
        func: Callable[..., float],
        estimates: dict[str, float],
        covariance: np.ndarray,
        param_names: list[str],
    ) -> ConfidenceInterval:
        """Delta method confidence interval for transformed parameter.

        Args:
            func: Transformation function
            estimates: Parameter estimates
            covariance: Covariance matrix of estimates
            param_names: Names of parameters in order

        Returns:
            ConfidenceInterval for transformed parameter
        """
        # Compute gradient
        gradient = np.zeros(len(param_names))
        step = self.config.delta_method_step

        for i, name in enumerate(param_names):
            estimates_plus = estimates.copy()
            estimates_plus[name] += step

            estimates_minus = estimates.copy()
            estimates_minus[name] -= step

            gradient[i] = (func(**estimates_plus) - func(**estimates_minus)) / (
                2 * step
            )

        # Point estimate
        theta = func(**estimates)

        # Variance using delta method
        variance = gradient @ covariance @ gradient
        se = np.sqrt(variance)

        # Confidence interval
        alpha = 1 - self.config.confidence_level
        z = self._normal_ppf(1 - alpha / 2)

        lower = theta - z * se
        upper = theta + z * se

        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            point_estimate=float(theta),
            confidence_level=self.config.confidence_level,
            method="delta",
        )

    # Private helper methods

    def _bca_interval(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        bootstrap_stats: np.ndarray,
        theta_hat: float,
        alpha: float,
    ) -> tuple[float, float, float, float]:
        """Compute BCa confidence interval."""
        n = len(data)

        # Bias correction factor
        prop_less = np.mean(bootstrap_stats < theta_hat)
        z0 = self._normal_ppf(prop_less) if 0 < prop_less < 1 else 0.0

        # Acceleration factor using jackknife
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        jack_diff = jack_mean - jackknife_stats

        num = np.sum(jack_diff**3)
        denom = 6 * (np.sum(jack_diff**2)) ** 1.5

        a = num / denom if denom != 0 else 0.0

        # Adjusted percentiles
        z_alpha_lower = self._normal_ppf(alpha / 2)
        z_alpha_upper = self._normal_ppf(1 - alpha / 2)

        # BCa adjustment
        def adjusted_percentile(z_alpha: float) -> float:
            if a == 0:
                return self._normal_cdf(z0 + z_alpha)
            numer = z0 + z_alpha
            denom = 1 - a * (z0 + z_alpha)
            if denom == 0:
                return 0.5
            return self._normal_cdf(z0 + numer / denom)

        p_lower = adjusted_percentile(z_alpha_lower)
        p_upper = adjusted_percentile(z_alpha_upper)

        # Clip to valid range
        p_lower = np.clip(p_lower, 0.001, 0.999)
        p_upper = np.clip(p_upper, 0.001, 0.999)

        ci_lower = float(np.percentile(bootstrap_stats, 100 * p_lower))
        ci_upper = float(np.percentile(bootstrap_stats, 100 * p_upper))

        return ci_lower, ci_upper, z0, a

    def _studentized_interval(
        self,
        data: np.ndarray,
        statistic: Callable[[np.ndarray], float],
        bootstrap_samples: np.ndarray,
        bootstrap_stats: np.ndarray,
        theta_hat: float,
        se: float,
        alpha: float,
    ) -> tuple[float, float]:
        """Compute studentized bootstrap interval."""
        n_bootstrap = len(bootstrap_stats)

        # Compute standard error for each bootstrap sample
        t_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = bootstrap_samples[i]
            boot_stat = bootstrap_stats[i]

            # Nested bootstrap for SE (simplified: use jackknife)
            jack_stats = np.zeros(len(sample))
            for j in range(len(sample)):
                jack_sample = np.delete(sample, j)
                jack_stats[j] = statistic(jack_sample)

            boot_se = np.std(jack_stats) * np.sqrt(len(sample) - 1)

            if boot_se > 0:
                t_stats[i] = (boot_stat - theta_hat) / boot_se
            else:
                t_stats[i] = 0

        # Percentiles of t-distribution
        t_lower = np.percentile(t_stats, 100 * alpha / 2)
        t_upper = np.percentile(t_stats, 100 * (1 - alpha / 2))

        # Interval
        ci_lower = theta_hat - t_upper * se
        ci_upper = theta_hat - t_lower * se

        return float(ci_lower), float(ci_upper)

    def _sample_distribution(
        self, dist_type: str, params: dict[str, Any], n: int
    ) -> np.ndarray:
        """Sample from a distribution."""
        if dist_type == "normal":
            return self._rng.normal(params.get("loc", 0), params.get("scale", 1), n)
        elif dist_type == "uniform":
            return self._rng.uniform(params.get("low", 0), params.get("high", 1), n)
        elif dist_type == "lognormal":
            return self._rng.lognormal(params.get("mean", 0), params.get("sigma", 1), n)
        elif dist_type == "exponential":
            return self._rng.exponential(params.get("scale", 1), n)
        elif dist_type == "beta":
            return self._rng.beta(params.get("a", 1), params.get("b", 1), n)
        elif dist_type == "gamma":
            return self._rng.gamma(params.get("shape", 1), params.get("scale", 1), n)
        elif dist_type == "triangular":
            return self._rng.triangular(
                params.get("left", 0),
                params.get("mode", 0.5),
                params.get("right", 1),
                n,
            )
        else:
            raise ValueError(f"Unknown distribution: {dist_type}")

    def _numerical_derivative(
        self,
        func: Callable[..., float],
        values: dict[str, float],
        param_name: str,
    ) -> float:
        """Compute numerical partial derivative."""
        h = self.config.delta_method_step

        values_plus = values.copy()
        values_plus[param_name] += h

        values_minus = values.copy()
        values_minus[param_name] -= h

        return (func(**values_plus) - func(**values_minus)) / (2 * h)

    def _sobol_sample(
        self,
        n: int,
        d: int,
        bounds: dict[str, tuple[float, float]],
        params: list[str],
    ) -> np.ndarray:
        """Generate Sobol-like samples (simplified quasi-random)."""
        # Use stratified sampling as approximation
        samples = np.zeros((n, d))

        for i, param in enumerate(params):
            low, high = bounds[param]
            # Stratified sampling
            strata = np.linspace(0, 1, n + 1)
            uniforms = self._rng.uniform(strata[:-1], strata[1:])
            self._rng.shuffle(uniforms)
            samples[:, i] = low + uniforms * (high - low)

        return samples

    def _skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        return float(np.mean(((data - mean) / std) ** 3) * n**2 / ((n - 1) * (n - 2)))

    def _kurtosis(self, data: np.ndarray) -> float:
        """Compute excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        m4 = np.mean((data - mean) ** 4)
        return float(m4 / std**4 - 3)

    def _normal_ppf(self, p: float) -> float:
        """Inverse standard normal CDF."""
        if p <= 0 or p >= 1:
            return 0.0
        if p < 0.5:
            return -self._normal_ppf(1 - p)

        t = np.sqrt(-2 * np.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _t_ppf(self, p: float, df: int) -> float:
        """Inverse t-distribution CDF (approximation)."""
        if df > 30:
            return self._normal_ppf(p)

        # Use normal approximation with adjustment
        z = self._normal_ppf(p)
        g1 = (z**3 + z) / 4
        g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96

        return z + g1 / df + g2 / df**2


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> BootstrapResult:
    """Convenience function for bootstrap confidence interval.

    Args:
        data: Input data
        statistic: Statistic function
        confidence_level: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        BootstrapResult with confidence interval

    Example:
        >>> data = np.random.randn(100)
        >>> result = bootstrap_confidence_interval(data)
        >>> print(f"CI: ({result.ci_lower:.3f}, {result.ci_upper:.3f})")
    """
    config = UncertaintyConfig(
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )
    uq = UncertaintyQuantifier(config)
    return uq.bootstrap_ci(data, statistic)


def propagate_uncertainty(
    func: Callable[..., float],
    values: dict[str, float],
    uncertainties: dict[str, float],
) -> tuple[float, float]:
    """Convenience function for error propagation.

    Args:
        func: Function to propagate through
        values: Parameter values
        uncertainties: Parameter uncertainties

    Returns:
        Tuple of (result, uncertainty)

    Example:
        >>> def area(r): return np.pi * r**2
        >>> result, unc = propagate_uncertainty(area, {'r': 5.0}, {'r': 0.1})
    """
    uq = UncertaintyQuantifier()
    return uq.error_propagation(func, values, uncertainties)


__all__ = [
    "BootstrapMethod",
    "UncertaintyMethod",
    "UncertaintyConfig",
    "ConfidenceInterval",
    "BootstrapResult",
    "MonteCarloResult",
    "SensitivityResult",
    "PredictionInterval",
    "UncertaintyQuantifier",
    "bootstrap_confidence_interval",
    "propagate_uncertainty",
]
