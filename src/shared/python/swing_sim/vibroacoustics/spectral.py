"""Spectral estimation for vibroacoustic measurements (IA-T5, #5074).

Welch PSD with explicit normalization, modal decay estimation from the
analytic envelope, and H1 frequency-response estimation.  Structural
modes and force histories are inputs here; radiation and observer
response live in a later, independently qualified tier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from shared.python.swing_sim.vibroacoustics.measurement import WaveformRecording

from ._spectral_frames import detrended as _detrended
from ._spectral_frames import finite_output as _finite_output
from ._spectral_frames import spectral_frames as _spectral_frames

DEFAULT_SEGMENT_LENGTH = 4096


def _samples(recording: WaveformRecording) -> np.ndarray:
    if not isinstance(recording, WaveformRecording):
        raise TypeError("recording must be a WaveformRecording")
    samples: np.ndarray = recording.samples
    return samples


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return the (start, stop) bounds of the longest ``True`` run."""
    best_start = start = 0
    best_stop = 0
    for index, value in enumerate(mask):
        if not value:
            start = index + 1
        elif index - start >= best_stop - best_start:
            best_start, best_stop = start, index + 1
    return best_start, best_stop


def psd_welch(
    recording: WaveformRecording,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one-sided Welch PSD ``(frequencies_hz, psd)``.

    Each complete segment has its own mean removed before applying a symmetric
    Hann window. Stride is segment_length//2; no zero-padded tail is included.
    Power is normalized so ``sum(psd) * df`` estimates the mean square
    of a stationary signal (Parseval-consistent within window and
    segmenting tolerance).

    Raises:
        ValueError: If ``segment_length`` is not an integer >= 3 no longer
            than the recording, or a numerical result is nonfinite.
    """
    samples = _samples(recording)
    spectrum, frequencies, divisor = _spectral_frames(
        samples, recording.sample_rate_hz, segment_length
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        periodogram = np.abs(spectrum) ** 2 / divisor
        if segment_length % 2 == 0:
            periodogram[:, 1:-1] *= 2.0
        else:
            periodogram[:, 1:] *= 2.0
        density = np.mean(periodogram, axis=0)
    return frequencies, _finite_output(density)


@dataclass(frozen=True)
class ModalDecayFit:
    """Estimated parameters of one exponentially decaying modal response."""

    natural_frequency_hz: float
    damping_ratio: float


def estimate_modal_decay(recording: WaveformRecording) -> ModalDecayFit:
    """Estimate ``(natural_frequency_hz, damping_ratio)`` from a ring-down.

    The analytic envelope (Hilbert transform) supplies the exponential
    decay rate; the spectral peak supplies the damped natural frequency.
    The two combine into the undamped natural frequency and damping
    ratio.

    Raises:
        ValueError: If the signal does not decay or the peak carries no
            usable energy.
    """
    samples = _samples(recording)
    analytic = hilbert(_detrended(samples))
    envelope = _finite_output(np.abs(analytic))
    if not np.any(envelope > 0):
        raise ValueError("ring-down must carry nonzero energy")
    above = envelope >= float(np.max(envelope)) * 1e-2
    start, stop = _longest_true_run(above)
    if stop - start < 8:
        raise ValueError("ring-down is too short to estimate modal decay")
    time = np.arange(start, stop) / recording.sample_rate_hz
    slope, _ = np.polyfit(time, np.log(envelope[start:stop]), 1)
    if slope >= 0.0:
        raise ValueError("signal must be decaying to estimate modal decay")
    spectrum = np.abs(np.fft.rfft(_detrended(samples)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / recording.sample_rate_hz)
    peak_bin = int(np.argmax(spectrum[1:])) + 1
    omega_d = 2.0 * np.pi * float(frequencies[peak_bin])
    omega_n = float(np.sqrt(omega_d**2 + slope**2))
    if not np.isfinite(omega_n) or not np.isfinite(slope) or omega_n <= 0.0:
        raise ValueError("modal frequency must be positive")
    return ModalDecayFit(
        natural_frequency_hz=omega_n / (2.0 * np.pi),
        damping_ratio=float(-slope / omega_n),
    )


def estimate_frf_h1(
    force_recording: WaveformRecording,
    response_recording: WaveformRecording,
    *,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the H1 FRF estimate ``(frequencies_hz, magnitude)``.

    H1 = Syx / Sxx from Welch-averaged auto- and cross-spectra of the
    measured force and response recordings.

    Raises:
        ValueError: If the recordings differ in length or sampling rate,
            or ``segment_length`` is invalid, any excitation bin is zero, or
            the numerical result is nonfinite. Positive excitation alone does not
            qualify signal-to-noise ratio, identifiability or uncertainty.
    """
    force = _samples(force_recording)
    response = _samples(response_recording)
    if force.shape != response.shape or force.ndim != 1:
        raise ValueError("recordings must be same length, one-dimensional arrays")
    if force_recording.sample_rate_hz != response_recording.sample_rate_hz:
        raise ValueError("recordings must share one sample rate")
    x_spectrum, frequencies, _ = _spectral_frames(
        force, force_recording.sample_rate_hz, segment_length
    )
    y_spectrum, _, _ = _spectral_frames(
        response, response_recording.sample_rate_hz, segment_length
    )
    with np.errstate(over="ignore", invalid="ignore"):
        sxx = _finite_output(np.mean(np.abs(x_spectrum) ** 2, axis=0))
        syx = _finite_output(np.mean(np.conj(x_spectrum) * y_spectrum, axis=0))
    if np.any(sxx <= 0):
        raise ValueError("H1 requires nonzero excitation in every returned bin")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude = np.abs(syx / sxx)
    return frequencies, _finite_output(magnitude)
