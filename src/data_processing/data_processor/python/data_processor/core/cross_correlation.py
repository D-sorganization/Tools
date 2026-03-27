"""Cross-Correlation Analysis Module.

Provides comprehensive cross-correlation and lag analysis capabilities
for understanding relationships between multiple time series.

Features:
- Cross-correlation function (CCF) computation
- Lag detection and optimal lag finding
- Partial cross-correlation
- Rolling/windowed cross-correlation
- Granger causality testing
- Transfer entropy estimation
- Multi-variate cross-correlation matrices
- Statistical significance testing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from data_processor.contracts import require

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Normalization methods for cross-correlation."""

    NONE = "none"
    BIASED = "biased"  # Divide by n
    UNBIASED = "unbiased"  # Divide by n-lag
    COEFF = "coeff"  # Pearson correlation coefficient


class CausalityMethod(Enum):
    """Methods for causality testing."""

    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    CONVERGENT_CROSS_MAPPING = "ccm"


@dataclass
class CrossCorrelationConfig:
    """Configuration for cross-correlation analysis."""

    # Correlation settings
    max_lag: int | None = None  # Auto-determined if None
    normalization: NormalizationMethod = NormalizationMethod.COEFF
    detrend: bool = True
    remove_mean: bool = True

    # Significance testing
    significance_level: float = 0.05
    num_permutations: int = 1000  # For permutation testing

    # Granger causality
    granger_max_lag: int = 10
    granger_criterion: str = "aic"  # 'aic', 'bic', 'hqic'

    # Rolling correlation
    rolling_window: int | None = None
    rolling_min_periods: int | None = None

    # Transfer entropy
    te_history_length: int = 1
    te_bins: int = 10


@dataclass
class CrossCorrelationResult:
    """Results from cross-correlation analysis."""

    # CCF values
    lags: np.ndarray
    ccf_values: np.ndarray

    # Optimal lag
    optimal_lag: int
    max_correlation: float
    correlation_at_zero: float

    # Significance
    confidence_interval: tuple[float, float]
    significant_lags: list[int]
    p_values: np.ndarray | None = None

    # Metadata
    series_x_name: str = "X"
    series_y_name: str = "Y"

    def is_significant_at_lag(self, lag: int) -> bool:
        """Check if correlation is significant at given lag."""
        return lag in self.significant_lags


@dataclass
class GrangerCausalityResult:
    """Results from Granger causality test."""

    # X causes Y
    x_causes_y: bool
    x_causes_y_pvalue: float
    x_causes_y_fstat: float

    # Y causes X
    y_causes_x: bool
    y_causes_x_pvalue: float
    y_causes_x_fstat: float

    # Optimal lags
    optimal_lag_xy: int
    optimal_lag_yx: int

    # Direction
    causal_direction: str  # 'X->Y', 'Y->X', 'bidirectional', 'none'

    # Model details
    aic_values: dict[str, float] = field(default_factory=dict)


@dataclass
class TransferEntropyResult:
    """Results from transfer entropy analysis."""

    # Transfer entropy values
    te_x_to_y: float
    te_y_to_x: float
    net_te: float  # X->Y minus Y->X

    # Significance
    te_x_to_y_pvalue: float
    te_y_to_x_pvalue: float

    # Direction
    dominant_direction: str  # 'X->Y', 'Y->X', 'none'


@dataclass
class RollingCorrelationResult:
    """Results from rolling cross-correlation."""

    timestamps: np.ndarray
    correlations: np.ndarray
    window_size: int

    # Statistics
    mean_correlation: float
    std_correlation: float
    correlation_stability: float  # 1 - coefficient of variation


class CrossCorrelationAnalyzer:
    """Comprehensive cross-correlation analysis engine.

    Provides tools for analyzing relationships between time series
    including lag detection, causality testing, and significance analysis.
    """

    def __init__(self, config: CrossCorrelationConfig | None = None) -> None:
        """Initialize the analyzer.

        Args:
            config: Configuration options
        """
        self.config = config or CrossCorrelationConfig()

    def cross_correlate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int | None = None,
    ) -> CrossCorrelationResult:
        """Compute cross-correlation function between two series.

        Args:
            x: First time series
            y: Second time series
            max_lag: Maximum lag to compute (both directions)

        Returns:
            CrossCorrelationResult with CCF values and analysis
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        require(len(x) == len(y), "Series must have the same length", len(x))
        require(len(x) >= 3, "Need at least 3 data points", len(x))

        n = len(x)
        max_lag = max_lag or self.config.max_lag or n // 3

        # Preprocessing
        if self.config.remove_mean:
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)

        if self.config.detrend:
            x = self._detrend(x)
            y = self._detrend(y)

        # Compute CCF
        lags = np.arange(-max_lag, max_lag + 1)
        ccf = np.zeros(len(lags))

        for i, lag in enumerate(lags):
            # Use -lag to match convention: positive lag = Y lags X
            ccf[i] = self._compute_ccf_at_lag(x, y, -lag)

        # Find optimal lag
        optimal_idx = np.argmax(np.abs(ccf))
        optimal_lag = int(lags[optimal_idx])
        max_correlation = float(ccf[optimal_idx])

        # Correlation at zero lag
        zero_idx = np.where(lags == 0)[0][0]
        correlation_at_zero = float(ccf[zero_idx])

        # Compute confidence intervals
        ci = self._compute_confidence_interval(n, self.config.significance_level)

        # Find significant lags
        significant_lags = [
            int(lag) for lag, c in zip(lags, ccf, strict=False) if abs(c) > ci[1]
        ]

        # Compute p-values if needed
        p_values = self._compute_pvalues(ccf, n)

        return CrossCorrelationResult(
            lags=lags,
            ccf_values=ccf,
            optimal_lag=optimal_lag,
            max_correlation=max_correlation,
            correlation_at_zero=correlation_at_zero,
            confidence_interval=ci,
            significant_lags=significant_lags,
            p_values=p_values,
        )

    def lagged_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
    ) -> tuple[float, float]:
        """Compute correlation at a specific lag.

        Args:
            x: First time series
            y: Second time series
            lag: Lag value (positive = y leads x)

        Returns:
            Tuple of (correlation, p-value)
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n = len(x)

        if abs(lag) >= n:
            return 0.0, 1.0

        # Align series
        if lag >= 0:
            x_aligned = x[lag:]
            y_aligned = y[: n - lag]
        else:
            x_aligned = x[: n + lag]
            y_aligned = y[-lag:]

        # Compute correlation
        corr = np.corrcoef(x_aligned, y_aligned)[0, 1]

        # Compute p-value
        n_eff = len(x_aligned)
        if n_eff > 2 and abs(corr) < 1:
            t_stat = corr * np.sqrt((n_eff - 2) / (1 - corr**2))
            # Two-tailed p-value (approximation)
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), n_eff - 2))
        else:
            p_value = 1.0 if abs(corr) < 1 else 0.0

        return float(corr), float(p_value)

    def find_optimal_lag(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int | None = None,
        criterion: str = "max",
    ) -> tuple[int, float]:
        """Find the lag that maximizes correlation.

        Args:
            x: First time series
            y: Second time series
            max_lag: Maximum lag to search
            criterion: 'max' for maximum absolute, 'positive' for max positive

        Returns:
            Tuple of (optimal_lag, correlation_at_lag)
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        result = self.cross_correlate(x, y, max_lag)

        if criterion == "max":
            return result.optimal_lag, result.max_correlation
        elif criterion == "positive":
            positive_mask = result.ccf_values > 0
            if not np.any(positive_mask):
                return 0, 0.0
            positive_ccf = np.where(positive_mask, result.ccf_values, -np.inf)
            idx = np.argmax(positive_ccf)
            return int(result.lags[idx]), float(result.ccf_values[idx])
        else:
            return result.optimal_lag, result.max_correlation

    def rolling_cross_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window: int | None = None,
        lag: int = 0,
    ) -> RollingCorrelationResult:
        """Compute rolling (windowed) cross-correlation.

        Args:
            x: First time series
            y: Second time series
            window: Window size
            lag: Fixed lag to use

        Returns:
            RollingCorrelationResult with time-varying correlations
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n = len(x)
        window = window or self.config.rolling_window or n // 10
        min_periods = self.config.rolling_min_periods or window // 2

        # Align series for lag
        if lag >= 0:
            x_work = x[lag:]
            y_work = y[: n - lag]
        else:
            x_work = x[: n + lag]
            y_work = y[-lag:]

        n_work = len(x_work)
        correlations = np.full(n_work, np.nan)

        for i in range(window - 1, n_work):
            start = i - window + 1
            x_window = x_work[start : i + 1]
            y_window = y_work[start : i + 1]

            valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
            if np.sum(valid_mask) >= min_periods:
                correlations[i] = np.corrcoef(
                    x_window[valid_mask], y_window[valid_mask]
                )[0, 1]

        timestamps = np.arange(n_work)

        # Calculate statistics
        valid_corr = correlations[~np.isnan(correlations)]
        mean_corr = float(np.mean(valid_corr)) if len(valid_corr) > 0 else 0.0
        std_corr = float(np.std(valid_corr)) if len(valid_corr) > 0 else 0.0

        stability = 1 - abs(std_corr / mean_corr) if mean_corr != 0 else 0.0

        return RollingCorrelationResult(
            timestamps=timestamps,
            correlations=correlations,
            window_size=window,
            mean_correlation=mean_corr,
            std_correlation=std_corr,
            correlation_stability=stability,
        )

    def granger_causality_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: int | None = None,
    ) -> GrangerCausalityResult:
        """Perform Granger causality test.

        Tests whether past values of X help predict Y (and vice versa)
        beyond what past values of Y alone can predict.

        Args:
            x: First time series
            y: Second time series
            max_lag: Maximum lag for VAR model

        Returns:
            GrangerCausalityResult with causality test results
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        max_lag = max_lag or self.config.granger_max_lag

        # Find optimal lag using information criterion
        optimal_lag_xy = self._select_lag_order(
            y, x, max_lag, self.config.granger_criterion
        )
        optimal_lag_yx = self._select_lag_order(
            x, y, max_lag, self.config.granger_criterion
        )

        # Test X causes Y
        f_stat_xy, p_value_xy = self._granger_test(y, x, optimal_lag_xy)

        # Test Y causes X
        f_stat_yx, p_value_yx = self._granger_test(x, y, optimal_lag_yx)

        # Determine significance
        alpha = self.config.significance_level
        x_causes_y = p_value_xy < alpha
        y_causes_x = p_value_yx < alpha

        # Determine causal direction
        if x_causes_y and y_causes_x:
            direction = "bidirectional"
        elif x_causes_y:
            direction = "X->Y"
        elif y_causes_x:
            direction = "Y->X"
        else:
            direction = "none"

        return GrangerCausalityResult(
            x_causes_y=x_causes_y,
            x_causes_y_pvalue=float(p_value_xy),
            x_causes_y_fstat=float(f_stat_xy),
            y_causes_x=y_causes_x,
            y_causes_x_pvalue=float(p_value_yx),
            y_causes_x_fstat=float(f_stat_yx),
            optimal_lag_xy=optimal_lag_xy,
            optimal_lag_yx=optimal_lag_yx,
            causal_direction=direction,
        )

    def transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        history_length: int | None = None,
    ) -> TransferEntropyResult:
        """Compute transfer entropy between two series.

        Transfer entropy measures the reduction in uncertainty of Y
        given the past of both X and Y versus the past of Y alone.

        Args:
            x: Source time series
            y: Target time series
            history_length: Number of past values to consider

        Returns:
            TransferEntropyResult with transfer entropy values
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        k = history_length or self.config.te_history_length
        bins = self.config.te_bins

        # Compute TE(X->Y)
        te_xy = self._compute_transfer_entropy(x, y, k, bins)

        # Compute TE(Y->X)
        te_yx = self._compute_transfer_entropy(y, x, k, bins)

        # Compute significance using permutation test
        p_xy = self._permutation_test_te(x, y, k, bins, te_xy)
        p_yx = self._permutation_test_te(y, x, k, bins, te_yx)

        # Net transfer entropy
        net_te = te_xy - te_yx

        # Determine dominant direction
        alpha = self.config.significance_level
        if p_xy < alpha and (p_yx >= alpha or te_xy > te_yx):
            direction = "X->Y"
        elif p_yx < alpha and (p_xy >= alpha or te_yx > te_xy):
            direction = "Y->X"
        else:
            direction = "none"

        return TransferEntropyResult(
            te_x_to_y=float(te_xy),
            te_y_to_x=float(te_yx),
            net_te=float(net_te),
            te_x_to_y_pvalue=float(p_xy),
            te_y_to_x_pvalue=float(p_yx),
            dominant_direction=direction,
        )

    def partial_cross_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        max_lag: int | None = None,
    ) -> CrossCorrelationResult:
        """Compute partial cross-correlation controlling for a third variable.

        Args:
            x: First time series
            y: Second time series
            z: Control variable
            max_lag: Maximum lag

        Returns:
            CrossCorrelationResult for partial correlation
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # Residualize x and y with respect to z
        x_resid = self._residualize(x, z)
        y_resid = self._residualize(y, z)

        # Compute cross-correlation of residuals
        return self.cross_correlate(x_resid, y_resid, max_lag)

    def multi_series_correlation_matrix(
        self,
        series_dict: dict[str, np.ndarray],
        lag: int = 0,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute correlation matrix for multiple series at given lag.

        Args:
            series_dict: Dictionary mapping names to time series
            lag: Lag to use for correlations

        Returns:
            Tuple of (correlation_matrix, series_names)
        """
        if not (series_dict is not None):
            raise ValueError("series_dict must be provided")
        names = list(series_dict.keys())
        n_series = len(names)

        corr_matrix = np.eye(n_series)

        for i in range(n_series):
            for j in range(i + 1, n_series):
                corr, _ = self.lagged_correlation(
                    series_dict[names[i]], series_dict[names[j]], lag
                )
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix, names

    def find_lead_lag_relationship(
        self,
        series_dict: dict[str, np.ndarray],
        max_lag: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Find lead-lag relationships among multiple series.

        Args:
            series_dict: Dictionary mapping names to time series
            max_lag: Maximum lag to search

        Returns:
            Dictionary with pairwise lead-lag relationships
        """
        if not (series_dict is not None):
            raise ValueError("series_dict must be provided")
        names = list(series_dict.keys())
        results = {}

        for i, name_x in enumerate(names):
            for j, name_y in enumerate(names):
                if i >= j:
                    continue

                x = series_dict[name_x]
                y = series_dict[name_y]

                ccf_result = self.cross_correlate(x, y, max_lag)

                key = f"{name_x}_vs_{name_y}"
                results[key] = {
                    "optimal_lag": ccf_result.optimal_lag,
                    "max_correlation": ccf_result.max_correlation,
                    "leader": name_x if ccf_result.optimal_lag > 0 else name_y,
                    "follower": name_y if ccf_result.optimal_lag > 0 else name_x,
                    "lag_magnitude": abs(ccf_result.optimal_lag),
                    "is_significant": abs(ccf_result.max_correlation)
                    > ccf_result.confidence_interval[1],
                }

        return results

    # Private helper methods

    def _compute_ccf_at_lag(self, x: np.ndarray, y: np.ndarray, lag: int) -> float:
        """Compute cross-correlation at a specific lag."""
        if not (x is not None):
            raise ValueError("x must be provided")
        n = len(x)

        if abs(lag) >= n:
            return 0.0

        # Align series
        if lag >= 0:
            x_aligned = x[lag:]
            y_aligned = y[: n - lag]
        else:
            x_aligned = x[: n + lag]
            y_aligned = y[-lag:]

        # Compute correlation based on normalization method
        norm = self.config.normalization

        if norm == NormalizationMethod.NONE:
            return float(np.sum(x_aligned * y_aligned))

        elif norm == NormalizationMethod.BIASED:
            return float(np.sum(x_aligned * y_aligned) / n)

        elif norm == NormalizationMethod.UNBIASED:
            n_eff = len(x_aligned)
            return float(np.sum(x_aligned * y_aligned) / n_eff)

        else:  # COEFF
            return float(np.corrcoef(x_aligned, y_aligned)[0, 1])

    def _detrend(self, data: np.ndarray) -> np.ndarray:
        """Remove linear trend from data."""
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, data, 1)
        return data - (slope * x + intercept)

    def _compute_confidence_interval(self, n: int, alpha: float) -> tuple[float, float]:
        """Compute confidence interval for CCF."""
        # Approximate using normal distribution
        if not (n is not None):
            raise ValueError("n must be provided")
        z = self._normal_ppf(1 - alpha / 2)
        ci = z / np.sqrt(n)
        return (-ci, ci)

    def _compute_pvalues(self, ccf: np.ndarray, n: int) -> np.ndarray:
        """Compute p-values for CCF values."""
        # Using Fisher's z-transformation approximation
        if not (ccf is not None):
            raise ValueError("ccf must be provided")
        p_values = np.zeros(len(ccf))

        for i, r in enumerate(ccf):
            if abs(r) >= 1:
                p_values[i] = 0.0
            else:
                # t-statistic
                t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                p_values[i] = 2 * (1 - self._t_cdf(abs(t_stat), n - 2))

        return p_values

    def _normal_ppf(self, p: float) -> float:
        """Approximate inverse normal CDF."""
        # Rational approximation
        if not (p is not None):
            raise ValueError("p must be provided")
        if p <= 0 or p >= 1:
            return 0.0

        if p < 0.5:
            return -self._normal_ppf(1 - p)

        t = np.sqrt(-2 * np.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        return t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Use normal approximation for large df
        if not (t is not None):
            raise ValueError("t must be provided")
        if df > 30:
            return self._normal_cdf(t)

        # Simple approximation
        x = df / (df + t**2)
        return 1 - 0.5 * self._incomplete_beta(df / 2, 0.5, x)

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _incomplete_beta(self, a: float, b: float, x: float) -> float:
        """Approximate incomplete beta function."""
        # Simple numerical integration
        if not (a is not None):
            raise ValueError("a must be provided")
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        n_points = 100
        t = np.linspace(0, x, n_points)
        dt = x / n_points

        integrand = t ** (a - 1) * (1 - t) ** (b - 1)
        integral = np.sum(integrand) * dt

        # Normalize by beta function (approximation)
        beta_func = np.exp(
            self._log_gamma(a) + self._log_gamma(b) - self._log_gamma(a + b)
        )

        return integral / beta_func if beta_func > 0 else 0.0

    def _log_gamma(self, x: float) -> float:
        """Approximate log gamma function."""
        # Stirling's approximation
        if not (x is not None):
            raise ValueError("x must be provided")
        if x <= 0:
            return 0.0
        return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2 * np.pi)

    def _granger_test(
        self, y: np.ndarray, x: np.ndarray, lag: int
    ) -> tuple[float, float]:
        """Perform Granger causality F-test."""
        if not (y is not None):
            raise ValueError("y must be provided")
        if lag < 1:
            lag = 1

        # Create lagged matrices
        y_lags = self._create_lag_matrix(y, lag)
        x_lags = self._create_lag_matrix(x, lag)

        # Dependent variable
        y_dep = y[lag:]

        # Restricted model: y ~ y_lags only
        rss_restricted = self._ols_residual_ss(y_dep, y_lags)

        # Unrestricted model: y ~ y_lags + x_lags
        combined = np.hstack([y_lags, x_lags])
        rss_unrestricted = self._ols_residual_ss(y_dep, combined)

        # F-statistic
        n_obs = len(y_dep)
        df1 = lag  # Number of restrictions
        df2 = n_obs - 2 * lag - 1  # Residual degrees of freedom

        if df2 <= 0 or rss_unrestricted <= 0:
            return 0.0, 1.0

        f_stat = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)

        # P-value from F-distribution (approximation)
        p_value = 1 - self._f_cdf(f_stat, df1, df2)

        return f_stat, p_value

    def _select_lag_order(
        self, y: np.ndarray, x: np.ndarray, max_lag: int, criterion: str
    ) -> int:
        """Select optimal lag order using information criterion."""
        if not (y is not None):
            raise ValueError("y must be provided")
        n = len(y)
        best_lag = 1
        best_ic = np.inf

        for lag in range(1, min(max_lag + 1, n // 3)):
            y_lags = self._create_lag_matrix(y, lag)
            x_lags = self._create_lag_matrix(x, lag)
            y_dep = y[lag:]

            combined = np.hstack([y_lags, x_lags])
            rss = self._ols_residual_ss(y_dep, combined)

            n_obs = len(y_dep)
            k = 2 * lag + 1  # Number of parameters

            # Compute information criterion
            if rss > 0:
                log_likelihood = -n_obs / 2 * np.log(rss / n_obs)

                if criterion == "aic":
                    ic = -2 * log_likelihood + 2 * k
                elif criterion == "bic":
                    ic = -2 * log_likelihood + k * np.log(n_obs)
                else:  # hqic
                    ic = -2 * log_likelihood + 2 * k * np.log(np.log(n_obs))

                if ic < best_ic:
                    best_ic = ic
                    best_lag = lag

        return best_lag

    def _create_lag_matrix(self, data: np.ndarray, lag: int) -> np.ndarray:
        """Create matrix of lagged values."""
        if not (data is not None):
            raise ValueError("data must be provided")
        n = len(data)
        matrix = np.zeros((n - lag, lag))

        for i in range(lag):
            matrix[:, i] = data[lag - i - 1 : n - i - 1]

        return matrix

    def _ols_residual_ss(self, y: np.ndarray, X: np.ndarray) -> float:
        """Compute residual sum of squares from OLS."""
        # Add constant
        if not (y is not None):
            raise ValueError("y must be provided")
        n = len(y)
        X_with_const = np.hstack([np.ones((n, 1)), X])

        # Solve normal equations
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            residuals = y - X_with_const @ beta
            return float(np.sum(residuals**2))
        except np.linalg.LinAlgError:
            return np.inf

    def _f_cdf(self, f: float, df1: int, df2: int) -> float:
        """Approximate F-distribution CDF."""
        if not (f is not None):
            raise ValueError("f must be provided")
        if f <= 0:
            return 0.0

        # Use incomplete beta function
        x = df2 / (df2 + df1 * f)
        return 1 - self._incomplete_beta(df2 / 2, df1 / 2, x)

    def _compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        k: int,
        bins: int,
    ) -> float:
        """Compute transfer entropy from source to target."""
        # Discretize
        if not (source is not None):
            raise ValueError("source must be provided")
        source_binned = self._discretize(source, bins)
        target_binned = self._discretize(target, bins)

        # Create joint distributions
        # TE(X->Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})

        # Compute conditional entropies
        h_y_given_ypast = self._conditional_entropy(
            target_binned[k:], target_binned[:-k], bins
        )

        # Combined conditioning
        combined_past = source_binned[:-k] * bins + target_binned[:-k]
        h_y_given_both = self._conditional_entropy(
            target_binned[k:], combined_past, bins * bins
        )

        te = h_y_given_ypast - h_y_given_both

        return max(0, te)  # TE should be non-negative

    def _discretize(self, data: np.ndarray, bins: int) -> np.ndarray:
        """Discretize continuous data into bins."""
        if not (data is not None):
            raise ValueError("data must be provided")
        percentiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(data, percentiles)
        return np.digitize(data, edges[1:-1])

    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray, y_bins: int) -> float:
        """Compute conditional entropy H(X|Y)."""
        # Joint probability
        if not (x is not None):
            raise ValueError("x must be provided")
        n = len(x)
        x_bins = int(np.max(x)) + 1

        joint_counts = np.zeros((x_bins, y_bins))
        for i in range(n):
            xi = int(x[i]) if x[i] < x_bins else x_bins - 1
            yi = int(y[i]) if y[i] < y_bins else y_bins - 1
            joint_counts[xi, yi] += 1

        joint_prob = joint_counts / n
        y_prob = np.sum(joint_prob, axis=0)

        # Conditional entropy
        h = 0.0
        for j in range(y_bins):
            if y_prob[j] > 0:
                for i in range(x_bins):
                    if joint_prob[i, j] > 0:
                        cond_prob = joint_prob[i, j] / y_prob[j]
                        h -= joint_prob[i, j] * np.log2(cond_prob)

        return h

    def _permutation_test_te(
        self,
        source: np.ndarray,
        target: np.ndarray,
        k: int,
        bins: int,
        observed_te: float,
    ) -> float:
        """Permutation test for transfer entropy significance."""
        if not (source is not None):
            raise ValueError("source must be provided")
        n_perms = min(self.config.num_permutations, 100)  # Limit for speed
        count_greater = 0

        for _ in range(n_perms):
            # Shuffle source (breaking temporal dependency)
            source_shuffled = np.random.permutation(source)
            te_perm = self._compute_transfer_entropy(source_shuffled, target, k, bins)
            if te_perm >= observed_te:
                count_greater += 1

        return (count_greater + 1) / (n_perms + 1)

    def _residualize(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute residuals of y regressed on x."""
        # Add constant
        if not (y is not None):
            raise ValueError("y must be provided")
        n = len(y)
        X = np.column_stack([np.ones(n), x])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return y - X @ beta
        except np.linalg.LinAlgError:
            return y - np.mean(y)


def cross_correlate(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int | None = None,
) -> CrossCorrelationResult:
    """Convenience function for cross-correlation.

    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag to compute

    Returns:
        CrossCorrelationResult with CCF analysis

    Example:
        >>> x = np.random.randn(100)
        >>> y = np.roll(x, 5) + np.random.randn(100) * 0.5
        >>> result = cross_correlate(x, y)
        >>> print(f"Optimal lag: {result.optimal_lag}")
    """
    if not (x is not None):
        raise ValueError("x must be provided")
    analyzer = CrossCorrelationAnalyzer()
    return analyzer.cross_correlate(x, y, max_lag)


def granger_causality(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
) -> GrangerCausalityResult:
    """Convenience function for Granger causality test.

    Args:
        x: First time series
        y: Second time series
        max_lag: Maximum lag for VAR model

    Returns:
        GrangerCausalityResult with causality test results
    """
    if not (x is not None):
        raise ValueError("x must be provided")
    config = CrossCorrelationConfig(granger_max_lag=max_lag)
    analyzer = CrossCorrelationAnalyzer(config)
    return analyzer.granger_causality_test(x, y, max_lag)


__all__ = [
    "NormalizationMethod",
    "CausalityMethod",
    "CrossCorrelationConfig",
    "CrossCorrelationResult",
    "GrangerCausalityResult",
    "TransferEntropyResult",
    "RollingCorrelationResult",
    "CrossCorrelationAnalyzer",
    "cross_correlate",
    "granger_causality",
]
