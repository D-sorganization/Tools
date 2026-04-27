"""FFT-based frequency domain filtering operations.

Extracted from VectorizedFilterEngine to follow the Single Responsibility
Principle. This module handles all frequency-domain filter design, window
functions, and FFT-based signal filtering.

Design by Contract (DbC) guards validate inputs at the public boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.signal import windows

logger = logging.getLogger(__name__)


def design_frequency_window(
    filter_type: str,
    freq_low: float,
    freq_high: float,
    window_shape: str,
    n_samples: int,
    transition_bw: float,
) -> np.ndarray[Any, Any]:
    """Design frequency domain window for FFT filtering.

    Args:
        filter_type: Type of filter (Low-pass, High-pass, Band-pass, Band-stop)
        freq_low: Lower cutoff frequency (normalized)
        freq_high: Upper cutoff frequency (normalized)
        window_shape: Window function type
        n_samples: Number of samples in signal
        transition_bw: Transition bandwidth (normalized)

    Returns:
        Frequency domain filter coefficients
    """
    assert filter_type is not None, "filter_type must be provided"
    freqs = np.fft.fftfreq(n_samples)
    freqs = np.abs(freqs)

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
                / transition_bw,
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
                / transition_bw,
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
                / transition_bw,
            )
        )

    if window_shape != "Rectangular":
        filter_response = apply_window_function(filter_response, window_shape)

    return filter_response


def apply_window_function(
    filter_response: np.ndarray[Any, Any],
    window_shape: str,
) -> np.ndarray[Any, Any]:
    """Apply window function to smooth frequency response."""
    assert filter_response is not None, "filter_response must be provided"
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
    signal_data: np.ndarray[Any, Any],
    filter_coeffs: np.ndarray[Any, Any],
    zero_phase: bool,
) -> np.ndarray[Any, Any]:
    """Core FFT filtering implementation.

    Args:
        signal_data: Input signal data
        filter_coeffs: Frequency domain filter coefficients
        zero_phase: Whether to use zero-phase filtering

    Returns:
        Filtered signal data
    """
    assert signal_data is not None, "signal_data must be provided"
    if len(filter_coeffs) != len(signal_data):
        old_indices = np.linspace(0, len(filter_coeffs) - 1, len(filter_coeffs))
        new_indices = np.linspace(0, len(filter_coeffs) - 1, len(signal_data))
        filter_coeffs = np.interp(new_indices, old_indices, filter_coeffs).astype(
            np.float64,
        )

    signal_fft = np.fft.fft(signal_data)
    filtered_fft = signal_fft * filter_coeffs

    if zero_phase:
        filtered_signal = np.real(np.fft.ifft(filtered_fft))
        filtered_fft_rev = np.fft.fft(filtered_signal[::-1])
        filtered_fft_rev = filtered_fft_rev * filter_coeffs
        filtered_signal_rev = np.real(np.fft.ifft(filtered_fft_rev))
        filtered_signal = filtered_signal_rev[::-1]
    else:
        filtered_signal = np.real(np.fft.ifft(filtered_fft))

    return filtered_signal
