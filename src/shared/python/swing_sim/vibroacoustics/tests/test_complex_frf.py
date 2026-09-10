"""Independent phase, support and coherence controls for Tools #5074."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy import signal

from shared.python.swing_sim.vibroacoustics import (
    SourceKind,
    WaveformRecording,
    estimate_frf_h1,
)
from shared.python.swing_sim.vibroacoustics.frf_estimation import (
    H1Bin,
    H1Estimate,
    H1Settings,
    estimate_complex_frf_h1,
)

RATE = 4096.0


def recording(samples: np.ndarray) -> WaveformRecording:
    """Declare synthetic samples without representing them as measurements."""
    return WaveformRecording(
        samples=samples,
        sample_rate_hz=RATE,
        unit="N",
        calibration_id="synthetic-unqualified",
        sensitivity_per_unit=1.0,
        source_id="synthetic-frf-control",
        source_kind=SourceKind.SYNTHESIZED,
    )


@pytest.mark.parametrize("length", [255, 256])
def test_complex_spectra_match_explicit_scipy_reference(length: int) -> None:
    """Odd windows also retain the repository's floor(L/2) stride."""
    rng = np.random.default_rng(5074)
    force = rng.normal(size=8192)
    response = signal.lfilter([0.2, -0.1], [1.0, -0.6], force)
    response += 0.15 * rng.normal(size=force.size)
    options = dict(
        fs=RATE,
        window=np.hanning(length),
        nperseg=length,
        noverlap=length - length // 2,
        nfft=length,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    frequencies, pxx = signal.welch(force, **options)
    _, pyy = signal.welch(response, **options)
    _, pxy = signal.csd(force, response, **options)
    result = estimate_complex_frf_h1(
        recording(force), recording(response), settings=H1Settings(length, 0.0, 0.0)
    )
    np.testing.assert_array_equal(result.frequencies_hz, frequencies)
    np.testing.assert_allclose([b.input_psd for b in result.bins], pxx, rtol=1e-12)
    np.testing.assert_allclose([b.response_psd for b in result.bins], pyy, rtol=1e-12)
    np.testing.assert_allclose([b.h1 for b in result.bins], pxy / pxx, rtol=1e-12)
    np.testing.assert_allclose(
        [b.coherence for b in result.bins], abs(pxy) ** 2 / (pxx * pyy), rtol=1e-12
    )
    assert result.segment_count == 1 + (force.size - length) // (length // 2)
    _, magnitude = estimate_frf_h1(
        recording(force), recording(response), segment_length=length
    )
    np.testing.assert_allclose([abs(b.h1) for b in result.bins], magnitude, rtol=1e-12)


def test_known_delay_has_negative_phase_and_correct_gain() -> None:
    """At a resolved tone, y[n]=2*x[n-4] implies H=2*exp(-i*omega*4)."""
    length, tone_bin, delay = 1024, 57, 4
    omega = 2 * np.pi * tone_bin / length
    sample = np.arange(8192)
    result = estimate_complex_frf_h1(
        recording(np.sin(omega * sample)),
        recording(2 * np.sin(omega * (sample - delay))),
        settings=H1Settings(length, 1e-6, 1e-6),
    )
    selected = result.bins[tone_bin]
    assert selected.h1 == pytest.approx(2 * np.exp(-1j * omega * delay), abs=2e-6)
    assert selected.coherence == pytest.approx(1, abs=1e-12)
    assert result.bins[5].h1 is None
    assert result.bins[5].coherence is None


def test_zero_response_is_zero_transfer_but_undefined_coherence() -> None:
    force = recording(np.random.default_rng(9).normal(size=512))
    result = estimate_complex_frf_h1(
        force, recording(np.zeros(512)), settings=H1Settings(128, 0.0, 0.0)
    )
    assert all(item.h1 == 0j for item in result.bins)
    assert all(item.coherence is None for item in result.bins)


def test_silence_reports_unsupported_bins_without_fake_zero_transfer() -> None:
    silence = recording(np.zeros(512))
    result = estimate_complex_frf_h1(
        silence, silence, settings=H1Settings(128, 0.0, 0.0)
    )
    assert all(item.h1 is None and item.coherence is None for item in result.bins)
    assert all(item.input_psd == item.response_psd == 0 for item in result.bins)


def test_response_floor_does_not_erase_defined_h1() -> None:
    force = recording(np.random.default_rng(3).normal(size=512))
    result = estimate_complex_frf_h1(force, force, settings=H1Settings(128, 0.0, 1e20))
    assert all(item.h1 == pytest.approx(1) for item in result.bins)
    assert all(item.coherence is None for item in result.bins)


@pytest.mark.parametrize("field", ["minimum_input_psd", "minimum_response_psd"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "0", 1j])
def test_settings_refuse_invalid_density_floor(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="PSD"):
        replace(H1Settings(128, 0.0, 0.0), **{field: value})


@pytest.mark.parametrize("length", [True, 2, 3.0, -2])
def test_settings_refuse_invalid_segment_length(length: object) -> None:
    with pytest.raises((TypeError, ValueError), match="segment"):
        H1Settings(length, 0.0, 0.0)


def test_one_segment_cannot_masquerade_as_coherence_evidence() -> None:
    source = recording(np.arange(128.0))
    with pytest.raises(ValueError, match="two complete"):
        estimate_complex_frf_h1(source, source, settings=H1Settings(128, 0.0, 0.0))


def test_pair_and_settings_contracts_refuse_implicit_conversion() -> None:
    source = recording(np.arange(512.0))
    with pytest.raises(TypeError, match="settings"):
        estimate_complex_frf_h1(source, source, settings={})
    with pytest.raises(TypeError, match="WaveformRecording"):
        estimate_complex_frf_h1(
            source.samples, source, settings=H1Settings(128, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="same length"):
        estimate_complex_frf_h1(
            source, recording(np.arange(256.0)), settings=H1Settings(128, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="sample rate"):
        estimate_complex_frf_h1(
            source,
            replace(source, sample_rate_hz=2048),
            settings=H1Settings(128, 0.0, 0.0),
        )


def test_results_are_immutable_and_do_not_change_input_identity() -> None:
    source = recording(np.random.default_rng(11).normal(size=512))
    original = source.samples.tobytes()
    result = estimate_complex_frf_h1(source, source, settings=H1Settings(128, 0.0, 0.0))
    assert isinstance(result.bins, tuple) and isinstance(result.frequencies_hz, tuple)
    with pytest.raises(FrozenInstanceError):
        result.segment_count = 0
    with pytest.raises(FrozenInstanceError):
        result.bins[0].h1 = 0j
    assert source.samples.tobytes() == original
    assert source.source_kind is SourceKind.SYNTHESIZED


def test_spectral_overflow_is_refused_even_when_threshold_would_hide_it() -> None:
    source = recording(np.tile([1e200, -1e200], 256))
    with pytest.raises(ValueError, match="finite"):
        estimate_complex_frf_h1(source, source, settings=H1Settings(128, 1e300, 1e300))


@pytest.mark.parametrize(
    "values",
    [
        (-1.0, 1.0, 1j, 1.0),
        (1.0, 1.0, complex(float("nan")), 1.0),
        (1.0, 1.0, 1j, 1.1),
        (1.0, 1.0, None, 0.5),
        (0.0, 1.0, 0j, None),
        (1.0, 0.0, 0j, 0.0),
        (1.0, 1.0, True, None),
    ],
)
def test_bin_construction_refuses_impossible_or_nonfinite_results(
    values: tuple,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        H1Bin(*values)


def test_result_construction_refuses_mutable_or_inconsistent_axes() -> None:
    item = H1Bin(1.0, 1.0, 1 + 0j, 1.0)
    valid = H1Estimate((0.0, 1.0), (item, item), 3, ("N", "Pa"))
    assert valid.units == ("N", "Pa")
    for change in (
        {"frequencies_hz": [0.0, 1.0]},
        {"frequencies_hz": (1.0, 0.0)},
        {"bins": (item,)},
        {"bins": (item, object())},
        {"segment_count": True},
        {"units": ("N", "")},
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(valid, **change)


@pytest.mark.parametrize("amplitude", [1e-140, 1e140])
def test_coherence_avoids_product_overflow_and_underflow(amplitude: float) -> None:
    source = recording(amplitude * np.random.default_rng(54).normal(size=512))
    result = estimate_complex_frf_h1(source, source, settings=H1Settings(128, 0.0, 0.0))
    assert all(item.h1 == pytest.approx(1.0) for item in result.bins)
    assert all(item.coherence == pytest.approx(1.0, abs=1e-14) for item in result.bins)


def test_psd_floor_is_strict_and_retains_unmasked_power() -> None:
    source = recording(np.random.default_rng(82).normal(size=512))
    baseline = estimate_complex_frf_h1(
        source, source, settings=H1Settings(128, 0.0, 0.0)
    )
    floor = baseline.bins[9].input_psd
    masked = estimate_complex_frf_h1(
        source, source, settings=H1Settings(128, floor, floor)
    )
    assert masked.bins[9].h1 is None
    assert masked.bins[9].coherence is None
    assert masked.bins[9].input_psd == floor
    for original, selected in zip(baseline.bins, masked.bins, strict=True):
        assert (selected.h1 is not None) == (original.input_psd > floor)


def test_legacy_magnitude_does_not_require_finite_response_autopower() -> None:
    source = np.random.default_rng(3).normal(size=512)
    _, magnitude = estimate_frf_h1(
        recording(source), recording(1e200 * source), segment_length=128
    )
    np.testing.assert_allclose(magnitude, 1e200, rtol=1e-14)
