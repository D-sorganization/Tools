"""Frequency-domain filter kernels for the vectorized filter engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import windows

try:
    from .constants import (
        DEFAULT_FFT_FREQ_HIGH,
        DEFAULT_FFT_FREQ_LOW,
        DEFAULT_FFT_FREQ_UNIT,
        DEFAULT_FFT_TRANSITION_BW,
        DEFAULT_FFT_WINDOW_SHAPE,
        DEFAULT_FFT_ZERO_PHASE,
        MAX_FFT_FREQUENCY,
        MAX_FFT_TRANSITION_BW,
        MIN_FFT_FREQUENCY,
        MIN_FFT_TRANSITION_BW,
    )
except ImportError:
    from constants import (  # type: ignore[no-redef]
        DEFAULT_FFT_FREQ_HIGH,
        DEFAULT_FFT_FREQ_LOW,
        DEFAULT_FFT_FREQ_UNIT,
        DEFAULT_FFT_TRANSITION_BW,
        DEFAULT_FFT_WINDOW_SHAPE,
        DEFAULT_FFT_ZERO_PHASE,
        MAX_FFT_FREQUENCY,
        MAX_FFT_TRANSITION_BW,
        MIN_FFT_FREQUENCY,
        MIN_FFT_TRANSITION_BW,
    )


ParamGetter = Callable[[dict[str, Any], str, Any, float | None, float | None], Any]
SamplingRateCalculator = Callable[[pd.Series], float | None]
Logger = Callable[[str], Any]


def apply_fft_filter_vectorized(
    signal: pd.Series,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    calculate_sampling_rate: SamplingRateCalculator,
    logger: Logger,
) -> pd.Series:
    """Apply an FFT-based filter with the configured frequency window."""
    filter_type = params.get("filter_type", "FFT Low-pass")
    window_shape = params.get("fft_window_shape", DEFAULT_FFT_WINDOW_SHAPE)
    freq_low = safe_get_param(
        params,
        "fft_freq_low",
        DEFAULT_FFT_FREQ_LOW,
        MIN_FFT_FREQUENCY,
        MAX_FFT_FREQUENCY,
    )
    freq_high = safe_get_param(
        params,
        "fft_freq_high",
        DEFAULT_FFT_FREQ_HIGH,
        MIN_FFT_FREQUENCY,
        MAX_FFT_FREQUENCY,
    )
    transition_bw = safe_get_param(
        params,
        "fft_transition_bw",
        DEFAULT_FFT_TRANSITION_BW,
        MIN_FFT_TRANSITION_BW,
        MAX_FFT_TRANSITION_BW,
    )
    zero_phase = params.get("fft_zero_phase", DEFAULT_FFT_ZERO_PHASE)
    freq_unit = params.get("fft_freq_unit", DEFAULT_FFT_FREQ_UNIT)

    clean_data = signal.dropna()
    if len(clean_data) < 4:
        logger("Warning: Signal too short for FFT filter")
        return signal

    try:
        sample_rate = None
        if freq_unit == "Hz":
            sample_rate = calculate_sampling_rate(signal)
            if sample_rate is None:
                logger(
                    "Warning: Cannot determine sample rate, using normalized frequencies"
                )
                freq_unit = "normalized"

        if freq_unit == "Hz" and sample_rate is not None:
            freq_low = freq_low / (sample_rate / 2)
            freq_high = freq_high / (sample_rate / 2)
            transition_bw = transition_bw / (sample_rate / 2)

        freq_low = max(0.0, min(freq_low, 0.5))
        freq_high = max(freq_low, min(freq_high, 0.5))

        filter_coeffs = design_frequency_window(
            filter_type,
            freq_low,
            freq_high,
            window_shape,
            len(clean_data),
            transition_bw,
        )
        filtered_data = apply_fft_filter_core(
            clean_data.values,
            filter_coeffs,
            zero_phase,
        )
        return pd.Series(filtered_data, index=clean_data.index)
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"FFT filter failed: {error}")
        return signal


def design_frequency_window(
    filter_type: str,
    freq_low: float,
    freq_high: float,
    window_shape: str,
    n_samples: int,
    transition_bw: float,
) -> np.ndarray:
    """Design a frequency-domain window for FFT filtering."""
    freqs = np.abs(np.fft.fftfreq(n_samples))
    filter_response = np.zeros_like(freqs)

    if filter_type == "FFT Low-pass":
        filter_response[freqs <= freq_low] = 1.0
        transition_mask = (freqs > freq_low) & (freqs <= freq_low + transition_bw)
        filter_response[transition_mask] = 0.5 * (
            1 + np.cos(np.pi * (freqs[transition_mask] - freq_low) / transition_bw)
        )
    elif filter_type == "FFT High-pass":
        filter_response[freqs >= freq_high] = 1.0
        transition_mask = (freqs >= freq_high - transition_bw) & (freqs < freq_high)
        filter_response[transition_mask] = 0.5 * (
            1
            - np.cos(
                np.pi
                * (freqs[transition_mask] - freq_high + transition_bw)
                / transition_bw
            )
        )
    elif filter_type == "FFT Band-pass":
        filter_response[(freqs >= freq_low) & (freqs <= freq_high)] = 1.0
        low_transition = (freqs > freq_low - transition_bw) & (freqs <= freq_low)
        high_transition = (freqs >= freq_high) & (freqs < freq_high + transition_bw)
        filter_response[low_transition] = 0.5 * (
            1
            + np.cos(
                np.pi
                * (freqs[low_transition] - freq_low + transition_bw)
                / transition_bw
            )
        )
        filter_response[high_transition] = 0.5 * (
            1 - np.cos(np.pi * (freqs[high_transition] - freq_high) / transition_bw)
        )
    elif filter_type == "FFT Band-stop":
        filter_response[(freqs < freq_low) | (freqs > freq_high)] = 1.0
        low_transition = (freqs >= freq_low) & (freqs < freq_low + transition_bw)
        high_transition = (freqs > freq_high - transition_bw) & (freqs <= freq_high)
        filter_response[low_transition] = 0.5 * (
            1 - np.cos(np.pi * (freqs[low_transition] - freq_low) / transition_bw)
        )
        filter_response[high_transition] = 0.5 * (
            1
            + np.cos(
                np.pi
                * (freqs[high_transition] - freq_high + transition_bw)
                / transition_bw
            )
        )

    if window_shape != "Rectangular":
        filter_response = apply_window_function(filter_response, window_shape)

    return filter_response


def apply_window_function(filter_response: np.ndarray, window_shape: str) -> np.ndarray:
    """Apply a smoothing window to the filter response."""
    n = len(filter_response)
    if window_shape == "Gaussian":
        sigma = n / 8
        window = np.exp(-0.5 * ((np.arange(n) - n / 2) / sigma) ** 2)
    elif window_shape == "Hamming":
        window = windows.hamming(n)
    elif window_shape == "Hann":
        window = windows.hann(n)
    elif window_shape == "Blackman":
        window = windows.blackman(n)
    elif window_shape == "Kaiser":
        window = windows.kaiser(n, beta=8.6)
    elif window_shape == "Tukey":
        window = windows.tukey(n, alpha=0.5)
    elif window_shape == "Bartlett":
        window = windows.bartlett(n)
    else:
        return filter_response

    window_fft = np.fft.fft(window)
    response_fft = np.fft.fft(filter_response)
    smoothed_fft = response_fft * window_fft
    smoothed_response = np.real(np.fft.ifft(smoothed_fft))
    return smoothed_response / np.max(smoothed_response)


def apply_fft_filter_core(
    signal_data: np.ndarray,
    filter_coeffs: np.ndarray,
    zero_phase: bool,
) -> np.ndarray:
    """Apply the FFT filter core to raw signal data."""
    if len(filter_coeffs) != len(signal_data):
        old_indices = np.linspace(0, len(filter_coeffs) - 1, len(filter_coeffs))
        new_indices = np.linspace(0, len(filter_coeffs) - 1, len(signal_data))
        filter_coeffs = np.interp(new_indices, old_indices, filter_coeffs).astype(
            np.float64
        )

    signal_fft = np.fft.fft(signal_data)
    filtered_fft = signal_fft * filter_coeffs

    if zero_phase:
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        filtered_fft_rev = np.fft.fft(filtered_signal[::-1]) * filter_coeffs
        filtered_signal = np.real(np.fft.ifft(filtered_fft_rev))[::-1]
    else:
        filtered_signal = np.real(np.fft.ifft(filtered_fft))

    return filtered_signal


def calculate_frequency_response(
    filter_type: str,
    params: dict[str, Any],
    safe_get_param: ParamGetter,
    logger: Logger,
    n_freqs: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the frequency response of an FFT filter for preview."""
    try:
        window_shape = params.get("fft_window_shape", DEFAULT_FFT_WINDOW_SHAPE)
        freq_low = safe_get_param(
            params,
            "fft_freq_low",
            DEFAULT_FFT_FREQ_LOW,
            MIN_FFT_FREQUENCY,
            MAX_FFT_FREQUENCY,
        )
        freq_high = safe_get_param(
            params,
            "fft_freq_high",
            DEFAULT_FFT_FREQ_HIGH,
            MIN_FFT_FREQUENCY,
            MAX_FFT_FREQUENCY,
        )
        transition_bw = safe_get_param(
            params,
            "fft_transition_bw",
            DEFAULT_FFT_TRANSITION_BW,
            MIN_FFT_TRANSITION_BW,
            MAX_FFT_TRANSITION_BW,
        )

        filter_coeffs = design_frequency_window(
            filter_type,
            freq_low,
            freq_high,
            window_shape,
            n_freqs,
            transition_bw,
        )
        freqs = np.fft.fftfreq(n_freqs)[: n_freqs // 2]
        magnitude = np.abs(filter_coeffs[: n_freqs // 2])
        return freqs, magnitude
    except (ValueError, TypeError, RuntimeError) as error:
        logger(f"Frequency response calculation failed: {error}")
        return np.array([]), np.array([])
