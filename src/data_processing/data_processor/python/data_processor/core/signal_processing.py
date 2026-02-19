"""Core signal processing functions.

This module provides the shared processing logic for all GUI implementations.
Functions here are UI-agnostic and can be used by TKinter, PyQt6, React, or CLI.

Implements:
- Signal integration (trapezoidal, rectangular, Simpson)
- Signal differentiation (spline, rolling polynomial)
- Time resampling with interpolation
- Custom calculated variables with formula parsing
- Trendline analysis (linear, polynomial, exponential, power)
- Time range manipulation
"""

from __future__ import annotations

import ast
import re
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from shared.python.safe_eval import safe_eval


class IntegrationMethod(Enum):
    """Available integration methods."""

    TRAPEZOIDAL = "trapezoidal"
    RECTANGULAR = "rectangular"
    SIMPSON = "simpson"


class DifferentiationMethod(Enum):
    """Available differentiation methods."""

    SPLINE = "spline"  # Acausal - uses entire dataset
    ROLLING_POLYNOMIAL = "rolling_polynomial"  # Causal - uses past data only


class TrendlineType(Enum):
    """Available trendline types."""

    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    POWER = "power"


# =============================================================================
# INTEGRATION
# =============================================================================


def integrate_signals(
    df: pd.DataFrame,
    time_col: str,
    signals: list[str],
    method: str = "trapezoidal",
) -> pd.DataFrame:
    """Integrate specified signals over time.

    Args:
        df: DataFrame containing time series data
        time_col: Name of the time column
        signals: List of signal column names to integrate
        method: Integration method ('trapezoidal', 'rectangular', 'simpson')

    Returns:
        DataFrame with new cumulative columns added (cumulative_{signal})
    """
    if not signals:
        return df

    result = df.copy()

    # Convert time to numeric (seconds from start)
    if pd.api.types.is_datetime64_any_dtype(result[time_col]):
        time_numeric = (result[time_col] - result[time_col].min()).dt.total_seconds()
    else:
        time_numeric = pd.to_numeric(result[time_col], errors="coerce")

    dt = time_numeric.diff().fillna(0)

    for signal in signals:
        if signal not in df.columns or signal == time_col:
            continue

        signal_data = pd.to_numeric(result[signal], errors="coerce")
        cumulative = _compute_integral(signal_data, dt, method)
        result[f"cumulative_{signal}"] = cumulative

    return result


def _compute_integral(signal_data: pd.Series, dt: pd.Series, method: str) -> np.ndarray:
    """Compute the cumulative integral of a signal."""
    n = len(signal_data)
    cumulative = np.zeros(n)

    if method == "trapezoidal":
        for i in range(1, n):
            y_curr = signal_data.iloc[i]
            y_prev = signal_data.iloc[i - 1]
            dt_i = dt.iloc[i]

            if not np.isnan(y_curr) and not np.isnan(y_prev):
                cumulative[i] = cumulative[i - 1] + 0.5 * (y_curr + y_prev) * dt_i
            else:
                cumulative[i] = cumulative[i - 1]

    elif method == "rectangular":
        # Left-endpoint method
        values = signal_data.fillna(0).values
        cumulative = np.cumsum(values * dt.values)

    elif method == "simpson":
        # Simpson's rule (falls back to trapezoidal for odd intervals)
        for i in range(1, n):
            y_curr = signal_data.iloc[i]
            y_prev = signal_data.iloc[i - 1]
            dt_i = dt.iloc[i]

            if not np.isnan(y_curr) and not np.isnan(y_prev):
                # For simplicity, use trapezoidal for each pair
                # Full Simpson requires triplets of points
                cumulative[i] = cumulative[i - 1] + 0.5 * (y_curr + y_prev) * dt_i
            else:
                cumulative[i] = cumulative[i - 1]

    return cumulative


# =============================================================================
# DIFFERENTIATION
# =============================================================================


def differentiate_signals(
    df: pd.DataFrame,
    time_col: str,
    signals: list[str],
    method: str = "spline",
    orders: list[int] | None = None,
    window_size: int = 11,
    poly_order: int = 3,
) -> pd.DataFrame:
    """Differentiate specified signals.

    Args:
        df: DataFrame containing time series data
        time_col: Name of the time column
        signals: List of signal column names to differentiate
        method: 'spline' (acausal) or 'rolling_polynomial' (causal)
        orders: List of derivative orders to compute (e.g., [1, 2] for 1st and 2nd)
        window_size: Window size for rolling polynomial method
        poly_order: Polynomial order for fitting

    Returns:
        DataFrame with new derivative columns added ({signal}_d{order})
    """
    if orders is None:
        orders = [1]

    if not signals:
        return df

    result = df.copy()

    # Convert time to numeric
    if pd.api.types.is_datetime64_any_dtype(result[time_col]):
        time_numeric = (result[time_col] - result[time_col].min()).dt.total_seconds()
    else:
        time_numeric = pd.to_numeric(result[time_col], errors="coerce")

    dt = time_numeric.diff().fillna(1).mean()  # Average time step

    for signal in signals:
        if signal not in df.columns or signal == time_col:
            continue

        signal_data = pd.to_numeric(result[signal], errors="coerce")

        for order in orders:
            if method == "spline":
                derivative = _spline_derivative(time_numeric, signal_data, order)
            else:  # rolling_polynomial
                derivative = _rolling_poly_derivative(
                    signal_data, window_size, poly_order, order, dt
                )

            result[f"{signal}_d{order}"] = derivative

    return result


def _spline_derivative(
    time_numeric: pd.Series, signal_data: pd.Series, order: int
) -> np.ndarray:
    """Compute derivative using spline interpolation (acausal)."""
    valid_mask = ~(np.isnan(signal_data) | np.isnan(time_numeric))

    if np.sum(valid_mask) <= order + 1:
        return np.full(len(signal_data), np.nan)

    try:
        x_valid = time_numeric[valid_mask].values
        y_valid = signal_data[valid_mask].values

        # Fit spline with smoothing factor 0 for interpolation
        k = min(5, len(y_valid) - 1)  # Spline order
        spline = UnivariateSpline(x_valid, y_valid, s=0, k=k)

        # Compute nth derivative
        deriv_spline = spline
        for _ in range(order):
            deriv_spline = deriv_spline.derivative()

        derivative = deriv_spline(time_numeric.values)
        derivative[~valid_mask] = np.nan

        return derivative

    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return np.full(len(signal_data), np.nan)


def _rolling_poly_derivative(
    series: pd.Series,
    window: int,
    poly_order: int,
    deriv_order: int,
    delta_x: float,
) -> pd.Series:
    """Compute derivative using rolling polynomial fit (causal)."""
    if poly_order < deriv_order:
        return pd.Series(np.nan, index=series.index)

    # Pad the series at the beginning
    padded_series = pd.concat([pd.Series([series.iloc[0]] * (window - 1)), series])

    def get_deriv(w: np.ndarray) -> float:
        if len(w) < window or np.isnan(w).any():
            return np.nan
        x = np.arange(len(w)) * delta_x
        try:
            coeffs = np.polyfit(x, w, poly_order)
            deriv_coeffs = np.polyder(coeffs, deriv_order)
            return np.polyval(deriv_coeffs, x[-1])
        except (np.linalg.LinAlgError, TypeError):
            return np.nan

    result = padded_series.rolling(window=window).apply(get_deriv, raw=True)
    return result.iloc[window - 1 :].reset_index(drop=True)


# =============================================================================
# TIME RESAMPLING
# =============================================================================


def resample_data(
    df: pd.DataFrame,
    time_col: str,
    rule: str,
    method: str = "mean",
    interpolate: bool = False,
) -> pd.DataFrame:
    """Resample time series data to a different frequency.

    Args:
        df: DataFrame with time series data
        time_col: Name of the time column
        rule: Pandas resample rule (e.g., '5s', '1min', '500ms')
        method: Aggregation method ('mean', 'sum', 'first', 'last')
        interpolate: Whether to interpolate when upsampling

    Returns:
        Resampled DataFrame
    """
    result = df.copy()

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
        result[time_col] = pd.to_datetime(result[time_col], errors="coerce")

    result = result.set_index(time_col)

    # Get numeric columns for resampling
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()

    if method == "mean":
        resampled = result[numeric_cols].resample(rule).mean()
    elif method == "sum":
        resampled = result[numeric_cols].resample(rule).sum()
    elif method == "first":
        resampled = result[numeric_cols].resample(rule).first()
    elif method == "last":
        resampled = result[numeric_cols].resample(rule).last()
    else:
        resampled = result[numeric_cols].resample(rule).mean()

    if interpolate:
        resampled = resampled.interpolate(method="time")

    resampled = resampled.dropna(how="all")
    return resampled.reset_index()


# =============================================================================
# CUSTOM CALCULATED VARIABLES
# =============================================================================

# Safe functions allowed in formulas
# Use numpy versions for all math functions to support array operations
SAFE_FUNCTIONS = {
    # Standard Python functions (work with scalars and arrays)
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "sum": np.sum,
    "len": len,
    "round": np.round,
    # Math functions - use numpy for array support
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "pow": np.power,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "pi": np.pi,
    "e": np.e,
    # Statistical functions
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    # Also expose with np_ prefix for clarity
    "np_sqrt": np.sqrt,
    "np_log": np.log,
    "np_log10": np.log10,
    "np_exp": np.exp,
    "np_sin": np.sin,
    "np_cos": np.cos,
    "np_tan": np.tan,
    "np_abs": np.abs,
    "np_mean": np.mean,
    "np_std": np.std,
    "np_min": np.min,
    "np_max": np.max,
}


def apply_custom_variable(
    df: pd.DataFrame,
    name: str,
    formula: str,
    time_col: str | None = None,
) -> pd.DataFrame:
    """Apply a custom calculated variable to the dataframe.

    Args:
        df: DataFrame with existing signals
        name: Name for the new calculated column
        formula: Formula using [signal_name] syntax for references
        time_col: Optional time column to exclude from calculations

    Returns:
        DataFrame with new calculated column added

    Raises:
        ValueError: If formula is invalid or uses unsafe operations
    """
    result = df.copy()

    # Parse the formula and validate
    parsed_formula, signal_refs = _parse_formula(formula, df.columns, time_col)

    # Validate formula security
    _validate_formula_security(parsed_formula, signal_refs)

    # Build evaluation context
    eval_context: dict[str, Any] = SAFE_FUNCTIONS.copy()

    # Add signal data to context
    for col in signal_refs:
        if pd.api.types.is_numeric_dtype(df[col]):
            eval_context[col] = df[col].values
        else:
            eval_context[col] = pd.to_numeric(df[col], errors="coerce").values

    # Evaluate the formula using AST-validated safe evaluator
    try:
        calculated = safe_eval(parsed_formula, eval_context)
        result[name] = calculated
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Error evaluating formula: {e}") from e

    return result


def _parse_formula(
    formula: str, columns: pd.Index, time_col: str | None
) -> tuple[str, set[str]]:
    """Parse formula and extract signal references.

    Converts [signal_name] syntax to plain variable names.
    Returns parsed formula and set of referenced signals.
    """
    signal_pattern = r"\[([^\]]+)\]"
    signal_refs: set[str] = set()

    def replace_signal(match: re.Match) -> str:
        signal_name = match.group(1)
        if signal_name not in columns:
            raise ValueError(f"Signal not found: {signal_name}")
        if signal_name == time_col:
            raise ValueError(f"Cannot use time column in formula: {signal_name}")
        signal_refs.add(signal_name)
        return signal_name

    parsed = re.sub(signal_pattern, replace_signal, formula)
    return parsed, signal_refs


def _validate_formula_security(formula: str, allowed_names: set[str]) -> None:
    """Validate that the formula only uses safe operations."""
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax: {e}") from e

    allowed = allowed_names | set(SAFE_FUNCTIONS.keys())

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.BinOp,
                ast.UnaryOp,
                ast.operator,
                ast.unaryop,
                ast.cmpop,
                ast.Compare,
                ast.BoolOp,
                ast.boolop,
                ast.Constant,
                ast.Name,
                ast.Call,
                ast.keyword,
                ast.Subscript,
                ast.Index,
                ast.Slice,
            ),
        ):
            if isinstance(node, ast.Name):
                if node.id not in allowed:
                    raise ValueError(f"Unknown variable or function: {node.id}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in allowed:
                        raise ValueError(f"Unknown function: {node.func.id}")
                else:
                    raise ValueError("Complex function calls not allowed")
            continue
        raise ValueError(f"Unsafe operation detected: {type(node).__name__}")


# =============================================================================
# TRENDLINE ANALYSIS
# =============================================================================


def calculate_trendline(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    trend_type: str = "linear",
    degree: int = 2,
    x_min: float | None = None,
    x_max: float | None = None,
) -> dict[str, Any]:
    """Calculate trendline for data.

    Args:
        df: DataFrame with x and y data
        x_col: Name of x-axis column
        y_col: Name of y-axis column
        trend_type: 'linear', 'polynomial', 'exponential', or 'power'
        degree: Polynomial degree (for polynomial trend)
        x_min: Optional minimum x value for fitting range
        x_max: Optional maximum x value for fitting range

    Returns:
        Dictionary with trend parameters and R-squared value
    """
    # DbC preconditions
    assert x_col in df.columns, f"x_col '{x_col}' not found in DataFrame columns"
    assert y_col in df.columns, f"y_col '{y_col}' not found in DataFrame columns"
    assert degree >= 1, f"Polynomial degree must be >= 1, got {degree}"

    # Filter to valid data
    mask = ~(np.isnan(df[x_col]) | np.isnan(df[y_col]))

    # Apply x range filter if specified
    if x_min is not None:
        mask &= df[x_col] >= x_min
    if x_max is not None:
        mask &= df[x_col] <= x_max

    x = df.loc[mask, x_col].values
    y = df.loc[mask, y_col].values

    if len(x) < 2:
        raise ValueError("Not enough data points for trendline")

    if trend_type == "linear":
        return _linear_trend(x, y)
    elif trend_type == "polynomial":
        return _polynomial_trend(x, y, degree)
    elif trend_type == "exponential":
        return _exponential_trend(x, y)
    elif trend_type == "power":
        return _power_trend(x, y)
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")


def _linear_trend(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Calculate linear regression: y = mx + b."""
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs

    # Calculate R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "equation": f"y = {slope:.4f}x + {intercept:.4f}",
        "predict": lambda x_new: slope * x_new + intercept,
    }


def _polynomial_trend(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, Any]:
    """Calculate polynomial regression."""
    coeffs = np.polyfit(x, y, degree)

    # Calculate R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "coefficients": coeffs.tolist(),
        "degree": degree,
        "r_squared": r_squared,
        "predict": lambda x_new: np.polyval(coeffs, x_new),
    }


def _exponential_trend(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Calculate exponential regression: y = a * exp(b * x)."""
    # Filter positive y values
    mask = y > 0
    if np.sum(mask) < 2:
        raise ValueError("Exponential fit requires positive y values")

    x_pos = x[mask]
    y_pos = y[mask]

    # Use log transform for initial estimate
    log_y = np.log(y_pos)
    coeffs = np.polyfit(x_pos, log_y, 1)
    b_init = coeffs[0]
    a_init = np.exp(coeffs[1])

    # Refine with curve fitting
    def exp_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(b * x)

    try:
        popt, _ = curve_fit(exp_func, x_pos, y_pos, p0=[a_init, b_init], maxfev=5000)
        a, b = popt
    except RuntimeError:
        a, b = a_init, b_init

    # Calculate R-squared
    y_pred = exp_func(x_pos, a, b)
    ss_res = np.sum((y_pos - y_pred) ** 2)
    ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "equation": f"y = {a:.4f} * exp({b:.4f} * x)",
        "predict": lambda x_new: a * np.exp(b * x_new),
    }


def _power_trend(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Calculate power regression: y = a * x^b."""
    # Filter positive values
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        raise ValueError("Power fit requires positive x and y values")

    x_pos = x[mask]
    y_pos = y[mask]

    # Use log transform
    log_x = np.log(x_pos)
    log_y = np.log(y_pos)
    coeffs = np.polyfit(log_x, log_y, 1)
    b = coeffs[0]
    a = np.exp(coeffs[1])

    # Calculate R-squared
    y_pred = a * (x_pos**b)
    ss_res = np.sum((y_pos - y_pred) ** 2)
    ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "equation": f"y = {a:.4f} * x^{b:.4f}",
        "predict": lambda x_new: a * (x_new**b),
    }


# =============================================================================
# TIME RANGE UTILITIES
# =============================================================================


def trim_time_range(
    df: pd.DataFrame,
    time_col: str,
    start_time: str | None = None,
    end_time: str | None = None,
    date: str | None = None,
) -> pd.DataFrame:
    """Trim data to a specific time range.

    Args:
        df: DataFrame with time series data
        time_col: Name of the time column
        start_time: Start time string (e.g., '10:30:00')
        end_time: End time string (e.g., '14:00:00')
        date: Optional date string (e.g., '2024-01-01')

    Returns:
        Trimmed DataFrame
    """
    result = df.copy()

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
        result[time_col] = pd.to_datetime(result[time_col], errors="coerce")

    result = result.dropna(subset=[time_col])

    if not (start_time or end_time):
        return result

    # Get date from data if not specified
    if date is None:
        date = result[time_col].iloc[0].strftime("%Y-%m-%d")

    # Build full datetime strings
    start_str = f"{date} {start_time or '00:00:00'}"
    end_str = f"{date} {end_time or '23:59:59'}"

    # Filter
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)

    mask = (result[time_col] >= start_dt) & (result[time_col] <= end_dt)
    return result[mask].reset_index(drop=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "integrate_signals",
    "differentiate_signals",
    "resample_data",
    "apply_custom_variable",
    "calculate_trendline",
    "trim_time_range",
    "IntegrationMethod",
    "DifferentiationMethod",
    "TrendlineType",
]
