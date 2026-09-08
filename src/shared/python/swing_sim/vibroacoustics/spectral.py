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

DEFAULT_SEGMENT_LENGTH = 4096


def _detrended(samples: np.ndarray) -> np.ndarray:
    return samples - float(np.mean(samples))


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

    Power is normalized so ``sum(psd) * df`` estimates the mean square
    of a stationary signal (Parseval-consistent within window and
    segmenting tolerance).

    Raises:
        ValueError: If ``segment_length`` is not a positive integer no
            longer than the recording.
    """
    samples = _samples(recording)
    if (
        isinstance(segment_length, bool)
        or not isinstance(segment_length, int)
        or segment_length < 2
    ):
        raise ValueError("segment_length must be an integer >= 2")
    if segment_length > samples.size:
        raise ValueError(
            f"segment_length {segment_length} exceeds recording length {samples.size}"
        )
    detrended = _detrended(samples)
    step = segment_length // 2
    count = 1 + (detrended.size - segment_length) // step
    window = np.hanning(segment_length)
    window_power = float(np.sum(window**2))
    frames = np.stack(
        [
            detrended[start : start + segment_length] * window
            for start in range(0, count * step, step)
        ]
    )
    spectrum = np.fft.rfft(frames, axis=1)
    periodogram = np.abs(spectrum) ** 2 / (recording.sample_rate_hz * window_power)
    if segment_length % 2 == 0:
        periodogram[:, 1:-1] *= 2.0
    else:
        periodogram[:, 1:] *= 2.0
    psd = np.mean(periodogram, axis=0)
    frequencies = np.fft.rfftfreq(segment_length, d=1.0 / recording.sample_rate_hz)
    return frequencies, psd


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
    envelope = np.abs(analytic)
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
    if omega_n <= 0.0:
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
            or ``segment_length`` is invalid.
    """
    force = _samples(force_recording)
    response = _samples(response_recording)
    if force.shape != response.shape or force.ndim != 1:
        raise ValueError("recordings must be same length, one-dimensional arrays")
    if force_recording.sample_rate_hz != response_recording.sample_rate_hz:
        raise ValueError("recordings must share one sample rate")
    if (
        isinstance(segment_length, bool)
        or not isinstance(segment_length, int)
        or segment_length < 2
    ):
        raise ValueError("segment_length must be an integer >= 2")
    if segment_length > force.size:
        raise ValueError(
            f"segment_length {segment_length} exceeds recording length {force.size}"
        )
    step = segment_length // 2
    count = 1 + (force.size - segment_length) // step
    window = np.hanning(segment_length)
    sxx = np.zeros(segment_length // 2 + 1)
    syx = np.zeros(segment_length // 2 + 1, dtype=complex)
    for start in range(0, count * step, step):
        x_frame = _detrended(force[start : start + segment_length]) * window
        y_frame = _detrended(response[start : start + segment_length]) * window
        x_spectrum = np.fft.rfft(x_frame)
        y_spectrum = np.fft.rfft(y_frame)
        sxx += np.abs(x_spectrum) ** 2
        syx += np.conj(x_spectrum) * y_spectrum
    transfer = syx / sxx
    frequencies = np.fft.rfftfreq(
        segment_length, d=1.0 / force_recording.sample_rate_hz
    )
    return frequencies, np.abs(transfer)
