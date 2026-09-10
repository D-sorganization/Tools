"""Shared synchronized-array boundary and unnormalized cross spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._spectral_frames import finite_output, spectral_frames
from .measurement import WaveformRecording


@dataclass(frozen=True)
class SpectralPair:
    """FFT frames using identical windows; units remain caller declarations."""

    frequencies: np.ndarray
    input_fft: np.ndarray
    response_fft: np.ndarray
    density_divisor: float

    def input_cross_spectra(self) -> tuple[np.ndarray, np.ndarray]:
        """Preserve the legacy H1 domain without computing response autopower."""
        with np.errstate(over="ignore", invalid="ignore"):
            sxx = finite_output(np.mean(np.abs(self.input_fft) ** 2, axis=0))
            sxy = finite_output(
                np.mean(np.conj(self.input_fft) * self.response_fft, axis=0)
            )
        return sxx, sxy

    def cross_spectra(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return unnormalized mean Sxx, Syy, and conj(X)*Y; refuse overflow."""
        sxx, sxy = self.input_cross_spectra()
        with np.errstate(over="ignore", invalid="ignore"):
            syy = finite_output(np.mean(np.abs(self.response_fft) ** 2, axis=0))
        return sxx, syy, sxy


def spectral_pair(
    force: WaveformRecording, response: WaveformRecording, segment_length: int
) -> SpectralPair:
    """Require equal-length/rate recordings; alignment is an external obligation."""
    if not isinstance(force, WaveformRecording) or not isinstance(
        response, WaveformRecording
    ):
        raise TypeError("recordings must be WaveformRecording instances")
    if force.samples.shape != response.samples.shape:
        raise ValueError("recordings must be same length, one-dimensional arrays")
    if force.sample_rate_hz != response.sample_rate_hz:
        raise ValueError("recordings must share one sample rate")
    x_fft, frequencies, divisor = spectral_frames(
        force.samples, force.sample_rate_hz, segment_length
    )
    y_fft, _, _ = spectral_frames(
        response.samples, response.sample_rate_hz, segment_length
    )
    return SpectralPair(frequencies, x_fft, y_fft, divisor)


__all__ = ()
