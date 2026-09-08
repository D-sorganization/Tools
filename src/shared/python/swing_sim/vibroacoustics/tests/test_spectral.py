"""Spectral estimation contract: PSD, modal decay, FRF (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from shared.python.swing_sim.vibroacoustics import (
    SourceKind,
    WaveformRecording,
    estimate_frf_h1,
    estimate_modal_decay,
    psd_welch,
)

RATE = 4096.0


def _recording(samples: np.ndarray) -> WaveformRecording:
    return WaveformRecording(
        samples=samples,
        sample_rate_hz=RATE,
        unit="Pa",
        calibration_id="cal-1",
        sensitivity_per_unit=1.0,
        source_id="rec-spec",
        source_kind=SourceKind.MEASURED,
    )


def _damped_sine(
    n: int,
    natural_hz: float,
    zeta: float,
    rate: float = RATE,
) -> np.ndarray:
    t = np.arange(n) / rate
    omega_n = 2.0 * np.pi * natural_hz
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    return np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)


def test_psd_dominant_frequency_and_parseval() -> None:
    n = 8192
    freq = 100.0
    amplitude = 2.0
    t = np.arange(n) / RATE
    samples = amplitude * np.sin(2.0 * np.pi * freq * t)
    frequencies, psd = psd_welch(_recording(samples))
    peak_hz = frequencies[int(np.argmax(psd))]
    assert peak_hz == pytest.approx(freq, abs=RATE / n)
    # Parseval: sum(psd) * df equals mean square of the (deterministic)
    # segment within segmenting/window tolerance for a stationary tone.
    df = frequencies[1] - frequencies[0]
    total_power = float(np.sum(psd) * df)
    assert total_power == pytest.approx(float(np.mean(samples**2)), rel=0.35)
    assert psd.shape == frequencies.shape


def test_psd_rejects_mismatched_rate() -> None:
    with pytest.raises(ValueError, match="length"):
        psd_welch(_recording(np.zeros(8)), segment_length=16)


def test_modal_decay_recovers_damping_and_frequency() -> None:
    natural_hz = 250.0
    zeta = 0.03
    samples = _damped_sine(8192, natural_hz, zeta)
    fit = estimate_modal_decay(_recording(samples))
    assert fit.natural_frequency_hz == pytest.approx(natural_hz, rel=0.05)
    assert fit.damping_ratio == pytest.approx(zeta, rel=0.2)


def test_modal_decay_rejects_growing_signal() -> None:
    growing = _damped_sine(4096, 250.0, -0.05)
    with pytest.raises(ValueError, match="decaying"):
        estimate_modal_decay(_recording(growing))


def test_frf_h1_recovers_damped_oscillator_resonance() -> None:
    natural_hz = 300.0
    zeta = 0.05
    n = 16384
    rng = np.random.default_rng(20260908)
    force = rng.standard_normal(n)
    # Sample the analytic force-to-displacement impulse response of a unit-mass
    # oscillator; the time-step factor supplies rectangular quadrature units.
    t = np.arange(n) / RATE
    omega_n = 2.0 * np.pi * natural_hz
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    impulse = (
        np.exp(-zeta * omega_n * t)
        * np.sin(omega_d * t)
        / (omega_n * np.sqrt(1.0 - zeta**2))
    )
    response = fftconvolve(force, impulse, mode="full")[:n] / RATE
    # A causal prefix independently checks FFT padding and truncation against
    # direct linear convolution without a quadratic full-record fixture cost.
    prefix = 256
    direct = np.convolve(force[:prefix], impulse[:prefix])[:prefix] / RATE
    np.testing.assert_allclose(response[:prefix], direct, rtol=1e-11, atol=1e-18)
    frequencies, magnitude = estimate_frf_h1(_recording(force), _recording(response))
    band = (frequencies > 50.0) & (frequencies < 900.0)
    peak_hz = frequencies[band][int(np.argmax(magnitude[band]))]
    expected_peak = natural_hz * np.sqrt(1.0 - 2.0 * zeta**2)
    assert peak_hz == pytest.approx(expected_peak, rel=0.05)


def test_frf_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        estimate_frf_h1(_recording(np.zeros(16)), _recording(np.zeros(8)))
