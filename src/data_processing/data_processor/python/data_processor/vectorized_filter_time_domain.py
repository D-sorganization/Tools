"""Time-domain filter kernels for the vectorized filter engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, filtfilt, medfilt

try:
    from .constants import (
        DEFAULT_BW_CUTOFF,
        DEFAULT_BW_ORDER,
        DEFAULT_GAUSSIAN_MODE,
        DEFAULT_GAUSSIAN_SIGMA,
        DEFAULT_HAMPEL_THRESHOLD,
        DEFAULT_HAMPEL_WINDOW,
        DEFAULT_MA_WINDOW,
        DEFAULT_MEDIAN_KERNEL,
        DEFAULT_SAVGOL_POLYORDER,
        DEFAULT_SAVGOL_WINDOW,
        DEFAULT_ZSCORE_METHOD,
        DEFAULT_ZSCORE_THRESHOLD,
        MIN_BUTTERWORTH_DATA_MULTIPLIER,
        MIN_SIGNAL_DATA_POINTS,
        NORMAL_DISTRIBUTION_CONSTANT,
    )
except ImportError:
    from constants import (  # type: ignore[no-redef]
        DEFAULT_BW_CUTOFF,
        DEFAULT_BW_ORDER,
        DEFAULT_GAUSSIAN_MODE,
        DEFAULT_GAUSSIAN_SIGMA,
        DEFAULT_HAMPEL_THRESHOLD,
        DEFAULT_HAMPEL_WINDOW,
        DEFAULT_MA_WINDOW,
        DEFAULT_MEDIAN_KERNEL,
        DEFAULT_SAVGOL_POLYORDER,
        DEFAULT_SAVGOL_WINDOW,
        DEFAULT_ZSCORE_METHOD,
        DEFAULT_ZSCORE_THRESHOLD,
        MIN_BUTTERWORTH_DATA_MULTIPLIER,
        MIN_SIGNAL_DATA_POINTS,
        NORMAL_DISTRIBUTION_CONSTANT,
    )

try:
    from scipy.signal import savgol_filter as _savgol_filter
except ImportError:
    _savgol_filter = None


ParamGetter = Callable[[dict[str, Any], str, Any, float | None, float | None], Any]
SamplingRateCalculator = Callable[[pd.Series], float | None]
Logger = Callable[[str], Any]


def apply_moving_average_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized moving average filter."""
    window = safe_get_param(params, "ma_window", DEFAULT_MA_WINDOW, 3, 1000)

    if len(signal) < window:
        return signal

    try:
        if not signal.hasnans:
            filtered_values = uniform_filter1d(
                signal.values,
                size=window,
                mode="nearest",
            )
            result = pd.Series(filtered_values, index=signal.index)

            radius = window // 2
            if radius > 0 and len(signal) > window:
                top_subset = signal.iloc[:window]
                top_corrected = top_subset.rolling(
                    window=window, min_periods=1, center=True
                ).mean()
                result.iloc[:radius] = top_corrected.iloc[:radius]

                bottom_subset = signal.iloc[-window:]
                bottom_corrected = bottom_subset.rolling(
                    window=window, min_periods=1, center=True
                ).mean()
                result.iloc[-radius:] = bottom_corrected.iloc[-radius:]

            return result

        result = signal.rolling(window=window, min_periods=1, center=True).mean()
        if signal.hasnans:
            result[signal.isna()] = np.nan
        return result
    except (ValueError, TypeError, RuntimeError):
        clean_data = signal.dropna()
        if len(clean_data) < window:
            return signal
        filtered_data = uniform_filter1d(
            clean_data.values,
            size=window,
            mode="nearest",
        )
        result = pd.Series(index=signal.index, dtype=float)
        result.loc[clean_data.index] = filtered_data
        return result


def apply_butterworth_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    calculate_sampling_rate: SamplingRateCalculator,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized Butterworth filter."""
    order = safe_get_param(params, "bw_order", DEFAULT_BW_ORDER, 1, 10)
    cutoff = safe_get_param(params, "bw_cutoff", DEFAULT_BW_CUTOFF, 0.01, 0.99)
    filter_type = params.get("filter_type", "Butterworth Low-pass")
    btype = "low" if "Low-pass" in filter_type else "high"

    sampling_rate = calculate_sampling_rate(signal)
    if (
        sampling_rate is None
        or len(signal.dropna()) <= order * MIN_BUTTERWORTH_DATA_MULTIPLIER
    ):
        logger("Warning: Insufficient data for Butterworth filter")
        return signal

    try:
        b, a = butter(N=order, Wn=cutoff, btype=btype, fs=sampling_rate)
        clean_data = signal.dropna()
        filtered_data = filtfilt(b, a, clean_data.values)
        return pd.Series(filtered_data, index=clean_data.index)
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Butterworth filter failed: {error}")
        return signal


def apply_median_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized median filter."""
    kernel = safe_get_param(params, "median_kernel", DEFAULT_MEDIAN_KERNEL, 3, 101)
    if kernel % 2 == 0:
        kernel += 1

    clean_data = signal.dropna()
    if len(clean_data) <= kernel:
        logger(f"Warning: Signal too short for median filter (kernel={kernel})")
        return signal

    try:
        filtered_data = medfilt(clean_data.values, kernel_size=kernel)
        return pd.Series(filtered_data, index=clean_data.index)
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Median filter failed: {error}")
        return signal


def apply_hampel_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized Hampel filter with a rolling-median fallback."""
    window = safe_get_param(params, "hampel_window", DEFAULT_HAMPEL_WINDOW, 3, 100)
    threshold = safe_get_param(
        params, "hampel_threshold", DEFAULT_HAMPEL_THRESHOLD, 1.0, 10.0
    )

    clean_data = signal.dropna()
    if len(clean_data) < window:
        logger(f"Warning: Signal too short for Hampel filter (window={window})")
        return signal

    try:
        rolling_median = clean_data.rolling(window=window, center=True).median()
        rolling_mad = (
            (clean_data - rolling_median)
            .abs()
            .rolling(window=window, center=True)
            .median()
        )
        threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad
        outlier_mask = (clean_data - rolling_median).abs() > threshold_values

        filtered_signal = signal.copy()
        filtered_signal.loc[clean_data.index[outlier_mask]] = rolling_median[
            outlier_mask
        ]
        return filtered_signal
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Vectorized Hampel filter failed, using fallback: {error}")
        return apply_hampel_fallback(signal, params, safe_get_param)


def apply_hampel_fallback(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
) -> pd.Series:
    """Apply the simplified Hampel fallback."""
    window = safe_get_param(params, "hampel_window", DEFAULT_HAMPEL_WINDOW, 3, 100)
    threshold = safe_get_param(
        params, "hampel_threshold", DEFAULT_HAMPEL_THRESHOLD, 1.0, 10.0
    )

    clean_data = signal.dropna()
    filtered_signal = signal.copy()
    rolling_median = clean_data.rolling(window=window, center=True).median()
    rolling_mad = (
        (clean_data - rolling_median).abs().rolling(window=window, center=True).median()
    )
    threshold_values = threshold * NORMAL_DISTRIBUTION_CONSTANT * rolling_mad
    outlier_mask = (clean_data - rolling_median).abs() > threshold_values
    filtered_signal.loc[clean_data.index[outlier_mask]] = rolling_median[outlier_mask]
    return filtered_signal


def apply_zscore_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized Z-score filter."""
    threshold = safe_get_param(
        params, "zscore_threshold", DEFAULT_ZSCORE_THRESHOLD, 1.0, 10.0
    )
    method = params.get("zscore_method", DEFAULT_ZSCORE_METHOD)

    clean_data = signal.dropna()
    if len(clean_data) < 3:
        logger("Warning: Signal too short for Z-score filter")
        return signal

    try:
        if method == "modified":
            center = float(np.median(clean_data.values))
            mad = float(np.median(np.abs(clean_data.values - center)))
            if mad == 0:
                return signal
            scale = NORMAL_DISTRIBUTION_CONSTANT * mad
            z_scores = np.abs((clean_data.values - center) / scale)
        else:
            center = float(np.mean(clean_data.values))
            scale = float(np.std(clean_data.values))
            if scale == 0:
                return signal
            z_scores = np.abs((clean_data.values - center) / scale)

        filtered_signal = signal.copy()
        outlier_mask = z_scores > threshold

        if method == "Clip Outliers":
            deviations = clean_data.values - center
            clipped_values = center + np.sign(deviations) * threshold * scale
            filtered_signal.loc[clean_data.index[outlier_mask]] = clipped_values[
                outlier_mask
            ]
        elif method == "Replace with Median":
            median_value = float(np.median(clean_data.values))
            filtered_signal.loc[clean_data.index[outlier_mask]] = median_value
        else:
            filtered_signal.loc[clean_data.index[outlier_mask]] = np.nan

        return filtered_signal
    except (ValueError, TypeError, RuntimeError, IndexError) as error:
        logger(f"Z-score filter failed: {error}")
        return signal


def apply_savgol_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized Savitzky-Golay filter."""
    window = safe_get_param(params, "savgol_window", DEFAULT_SAVGOL_WINDOW, 3, 101)
    polyorder = safe_get_param(
        params, "savgol_polyorder", DEFAULT_SAVGOL_POLYORDER, 1, 6
    )

    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1

    clean_data = signal.dropna()
    if len(clean_data) <= window:
        logger(f"Warning: Signal too short for Savitzky-Golay filter (window={window})")
        return signal

    if _savgol_filter is None:
        logger("Warning: scipy.signal.savgol_filter unavailable")
        return signal

    try:
        filtered_data = _savgol_filter(clean_data.values, window, polyorder)
        return pd.Series(filtered_data, index=clean_data.index)
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Savitzky-Golay filter failed: {error}")
        return signal


def apply_gaussian_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
) -> pd.Series:
    """Apply a vectorized Gaussian filter."""
    sigma = safe_get_param(params, "gaussian_sigma", DEFAULT_GAUSSIAN_SIGMA, 0.1, 100.0)
    mode = params.get("gaussian_mode", DEFAULT_GAUSSIAN_MODE)

    clean_data = signal.dropna()
    if len(clean_data) < MIN_SIGNAL_DATA_POINTS - 1:
        logger("Warning: Signal too short for Gaussian filter")
        return signal

    try:
        filtered_data = gaussian_filter1d(clean_data.values, sigma=sigma, mode=mode)
        return pd.Series(filtered_data, index=clean_data.index)
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Gaussian filter failed, using moving average fallback: {error}")
        return signal.rolling(
            window=min(10, len(signal)),
            min_periods=1,
            center=True,
        ).mean()
