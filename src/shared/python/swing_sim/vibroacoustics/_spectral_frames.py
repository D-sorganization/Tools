"""Shared, explicitly windowed and segment-detrended spectral inputs."""

from __future__ import annotations

import numpy as np


def finite_output(value: np.ndarray) -> np.ndarray:
    """Refuse nonfinite numerical results; never replace them with zeros."""
    if not np.all(np.isfinite(value)):
        raise ValueError("spectral numerical result must be finite")
    return value


def detrended(samples: np.ndarray) -> np.ndarray:
    """Remove each last-axis mean, refusing numerical overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = samples - np.mean(samples, axis=-1, keepdims=True)
    return finite_output(result)


def spectral_frames(
    samples: np.ndarray, rate_hz: float, segment_length: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return rFFT frames, frequencies and density divisor for a symmetric Hann.

    Use length//2 stride, last-axis constant detrending and only full frames.
    Require at least three samples per window: the length-two symmetric Hann
    is identically zero. Every returned number must be finite.
    """
    if (
        isinstance(segment_length, bool)
        or not isinstance(segment_length, int)
        or segment_length < 3
        or segment_length > samples.size
    ):
        raise ValueError(
            "segment_length must be an integer >= 3 within recording length"
        )
    window = np.hanning(segment_length)
    frames = np.lib.stride_tricks.sliding_window_view(samples, segment_length)[
        :: segment_length // 2
    ]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        spectrum = np.fft.rfft(detrended(frames) * window, axis=-1)
        divisor = float(rate_hz * np.sum(window**2))
        frequencies = np.fft.rfftfreq(segment_length, d=1.0 / rate_hz)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("spectral normalization must be finite and positive")
    return finite_output(spectrum), finite_output(frequencies), divisor


__all__ = ()
