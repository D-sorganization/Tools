"""Tests for psychoacoustics, loudness and sharpness (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics.psychoacoustics import (
    AcousticReferenceAlgorithm,
    calculate_sharpness,
    calculate_specific_loudness,
    compute_equivalent_sound_level,
    compute_peak_spl,
    compute_sound_exposure_level,
)


@pytest.mark.unit
def test_spl_and_sound_exposure_level_analytic_sine() -> None:
    sample_rate_hz = 48000.0
    duration = 1.0  # 1 second
    t = np.arange(0, duration, 1.0 / sample_rate_hz)
    p0 = 20e-6  # 20 uPa reference

    # Sine wave with RMS pressure = 1.0 Pa -> SPL = 20 log10(1 / 20e-6) = 93.9794 dB
    peak_p = np.sqrt(2.0)
    p_wave = peak_p * np.sin(2.0 * np.pi * 1000.0 * t)

    peak_spl = compute_peak_spl(p_wave, p0=p0)
    # Peak SPL is 20 log10(sqrt(2) / 20e-6) = 96.9897 dB
    assert peak_spl == pytest.approx(96.9897, abs=0.01)

    exp_val, sel_val = compute_sound_exposure_level(
        p_wave, sample_rate_hz=sample_rate_hz, p0=p0
    )
    # Exposure E = int p^2 dt = 1.0 Pa^2 * 1 s = 1.0 Pa^2 s
    # SEL = 10 log10(1.0 / 4e-10) = 93.9794 dB
    assert exp_val == pytest.approx(1.0, rel=1e-3)
    assert sel_val == pytest.approx(93.9794, abs=0.05)

    leq = compute_equivalent_sound_level(p_wave, sample_rate_hz=sample_rate_hz, p0=p0)
    assert leq == pytest.approx(93.9794, abs=0.05)


@pytest.mark.unit
def test_din_45692_sharpness_reference_fixture() -> None:
    # DIN 45692 reference fixture:
    # Standard reference sound is a 1 kHz narrow-band tone at 60 dB SPL,
    # which by definition has a sharpness of 1.0 acum (+/- 0.05 acum).
    sample_rate_hz = 48000.0
    duration = 0.5
    t = np.arange(0, duration, 1.0 / sample_rate_hz)
    p0 = 20e-6
    rms_p = p0 * 10.0 ** (60.0 / 20.0)  # 60 dB SPL = 0.02 Pa
    p_1k = np.sqrt(2.0) * rms_p * np.sin(2.0 * np.pi * 1000.0 * t)

    sharpness_1k = calculate_sharpness(
        pressure_samples=p_1k,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.DIN_45692,
    )

    # Reference condition must be approximately 1.0 acum
    assert sharpness_1k == pytest.approx(1.0, abs=0.08)

    # Frequency ordering: high-frequency content must have strictly higher sharpness
    # 4 kHz tone at same 60 dB SPL:
    p_4k = np.sqrt(2.0) * rms_p * np.sin(2.0 * np.pi * 4000.0 * t)
    sharpness_4k = calculate_sharpness(
        pressure_samples=p_4k,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.DIN_45692,
    )
    assert sharpness_4k > 1.8 * sharpness_1k

    # Low-frequency 250 Hz tone at 60 dB SPL must have lower sharpness
    p_250 = np.sqrt(2.0) * rms_p * np.sin(2.0 * np.pi * 250.0 * t)
    sharpness_250 = calculate_sharpness(
        pressure_samples=p_250,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.DIN_45692,
    )
    assert sharpness_250 < 0.8 * sharpness_1k


@pytest.mark.unit
def test_specific_loudness_iso_532_1_bands() -> None:
    sample_rate_hz = 48000.0
    duration = 0.2
    t = np.arange(0, duration, 1.0 / sample_rate_hz)
    p0 = 20e-6
    rms_p = p0 * 10.0 ** (70.0 / 20.0)  # 70 dB SPL
    p_wave = np.sqrt(2.0) * rms_p * np.sin(2.0 * np.pi * 1000.0 * t)

    spec_loudness, total_loudness = calculate_specific_loudness(
        pressure_samples=p_wave,
        sample_rate_hz=sample_rate_hz,
        algorithm=AcousticReferenceAlgorithm.ISO_532_1,
    )

    # Must produce 24 Bark critical bands
    assert len(spec_loudness) == 24
    assert total_loudness > 0.0
    # Peak specific loudness should be around band 8-9 (1 kHz is at ~8.5 Bark)
    assert np.argmax(spec_loudness) in (7, 8, 9)


@pytest.mark.unit
def test_refusal_of_invalid_algorithms_and_inputs() -> None:
    p_wave = np.ones(100)
    with pytest.raises(ValueError, match="reference algorithm"):
        calculate_sharpness(
            pressure_samples=p_wave,
            sample_rate_hz=48000.0,
            algorithm="spectral_centroid",  # type: ignore[arg-type]
        )
