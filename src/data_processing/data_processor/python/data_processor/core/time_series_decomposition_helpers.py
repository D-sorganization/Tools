"""Low-level numeric helpers for time-series decomposition."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate a centered moving average."""
    if data is None:
        raise ValueError("data must be provided")
    if window < 1:
        window = 1
    if window > len(data):
        window = len(data)

    kernel = np.ones(window) / window
    averaged = np.convolve(data, kernel, mode="same")

    half = window // 2
    for i in range(half):
        averaged[i] = np.mean(data[: i + half + 1])
        averaged[-(i + 1)] = np.mean(data[-(i + half + 1) :])

    return averaged


@jit(nopython=True, fastmath=True)
def centered_moving_average(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate centered moving average for classical decomposition."""
    if data is None:
        raise ValueError("data must be provided")
    n = len(data)
    result = np.full(n, np.nan)
    half = period // 2

    for i in range(half, n - half):
        if period % 2 == 0:
            result[i] = (
                0.5 * data[i - half]
                + np.sum(data[i - half + 1 : i + half])
                + 0.5 * data[i + half]
            ) / period
        else:
            result[i] = np.mean(data[i - half : i + half + 1])

    first_valid = half
    last_valid = n - half - 1
    result[:first_valid] = result[first_valid]
    result[last_valid + 1 :] = result[last_valid]
    return result


@jit(nopython=True, fastmath=True)
def lowess_smooth(data: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """LOWESS (Locally Weighted Scatterplot Smoothing)."""
    if data is None:
        raise ValueError("data must be provided")
    n = len(data)
    x = np.arange(n)
    result = np.zeros(n)
    neighbors = max(int(frac * n), 2)

    for i in range(n):
        distances = np.abs(x - x[i])
        nearest_idx = np.argsort(distances)[:neighbors]
        max_dist = distances[nearest_idx[-1]]
        if max_dist == 0:
            max_dist = 1.0
        u = distances[nearest_idx] / max_dist
        weights = (1 - u**3) ** 3
        weights = np.clip(weights, 0, None)

        x_local = x[nearest_idx]
        y_local = data[nearest_idx]
        sum_w = np.sum(weights)
        if sum_w == 0:
            result[i] = data[i]
            continue

        sum_wx = np.sum(weights * x_local)
        sum_wy = np.sum(weights * y_local)
        sum_wxx = np.sum(weights * x_local * x_local)
        sum_wxy = np.sum(weights * x_local * y_local)
        denom = sum_w * sum_wxx - sum_wx * sum_wx
        if abs(denom) < 1e-10:
            result[i] = sum_wy / sum_w
        else:
            slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom
            intercept = (sum_wy - slope * sum_wx) / sum_w
            result[i] = intercept + slope * x[i]

    return result


def polynomial_trend(data: np.ndarray, degree: int) -> np.ndarray:
    """Fit a polynomial trend line."""
    if data is None:
        raise ValueError("data must be provided")
    x = np.arange(len(data))
    coefficients = np.polyfit(x, data, degree)
    return np.polyval(coefficients, x)


@jit(nopython=True, fastmath=True)
def exponential_smooth(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential smoothing."""
    if data is None:
        raise ValueError("data must be provided")
    result = np.zeros(len(data))
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


@jit(nopython=True, fastmath=True)
def hp_filter(data: np.ndarray, lambd: float) -> np.ndarray:
    """Apply the Hodrick-Prescott filter."""
    if data is None:
        raise ValueError("data must be provided")
    n = len(data)
    second_difference = np.zeros((n - 2, n))
    for i in range(n - 2):
        second_difference[i, i] = 1
        second_difference[i, i + 1] = -2
        second_difference[i, i + 2] = 1

    penalty = lambd * second_difference.T @ second_difference
    identity = np.eye(n)
    try:
        return np.linalg.solve(identity + penalty, data)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(identity + penalty, data, rcond=None)[0]


def extract_stl_seasonal(
    detrended: np.ndarray,
    period: int,
    smoother: Callable[[np.ndarray, float], np.ndarray] = lowess_smooth,
) -> np.ndarray:
    """Extract the STL seasonal component for a detrended series."""
    if detrended is None:
        raise ValueError("detrended must be provided")
    seasonal_indices = np.zeros(period)

    for offset in range(period):
        subseries = detrended[offset::period]
        smoothed = smoother(subseries, frac=0.5)
        seasonal_indices[offset] = np.mean(smoothed)

    seasonal_indices -= np.mean(seasonal_indices)
    return np.tile(seasonal_indices, len(detrended) // period + 1)[: len(detrended)]


@jit(nopython=True, fastmath=True)
def autocorrelation(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Calculate the autocorrelation function."""
    if data is None:
        raise ValueError("data must be provided")
    centered = data - np.mean(data)
    variance = np.var(centered)
    if variance == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    n = len(data)
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(centered[lag:] * centered[:-lag]) / ((n - lag) * variance)
    return acf


@jit(nopython=True, fastmath=True)
def find_acf_peaks(acf: np.ndarray) -> list[int]:
    """Find dominant peaks in an autocorrelation sequence."""
    if acf is None:
        raise ValueError("acf must be provided")
    peaks = []
    for i in range(2, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
            peaks.append(i)
    return peaks


def extrapolate_exponential(trend: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate a trend using a log-linear fit over the tail."""
    if trend is None:
        raise ValueError("trend must be provided")
    fit_length = min(len(trend), 50)
    x = np.arange(fit_length)
    y = trend[-fit_length:]
    shifted = y - np.min(y) + 1
    log_y = np.log(shifted)
    slope, intercept = np.polyfit(x, log_y, 1)
    future_x = np.arange(fit_length, fit_length + horizon)
    return np.exp(intercept + slope * future_x) + np.min(y) - 1


__all__ = [
    "autocorrelation",
    "centered_moving_average",
    "exponential_smooth",
    "extract_stl_seasonal",
    "extrapolate_exponential",
    "find_acf_peaks",
    "hp_filter",
    "lowess_smooth",
    "moving_average",
    "polynomial_trend",
]
