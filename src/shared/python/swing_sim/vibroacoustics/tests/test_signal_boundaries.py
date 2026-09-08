"""Numerical counterexamples for IA-T5; these are synthetic, not measurements."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from scipy import signal

from shared.python.swing_sim.vibroacoustics import (
    align_time_shift,
    estimate_frf_h1,
    estimate_modal_decay,
    psd_welch,
)

from .test_measurement import _recording


@pytest.mark.parametrize(
    "samples",
    [
        np.array([1 + 2j, 2 + 3j]),
        [True, 2.0],
        np.array([True, False]),
        ["1", "2"],
        np.array([1.0, 2.0], dtype=object),
    ],
)
def test_samples_refuse_nonreal_or_coerced_values(samples: object) -> None:
    with pytest.raises(ValueError, match="real"):
        _recording(samples=samples)


def test_recorded_samples_are_owned_and_cannot_be_made_writable() -> None:
    samples = np.array([1.0, 2.0, 3.0])
    recording = _recording(samples=samples)
    samples[0] = 99
    np.testing.assert_array_equal(recording.samples, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        recording.samples[0] = 99
    with pytest.raises(ValueError):
        recording.samples.setflags(write=True)


@pytest.mark.parametrize("lag", [-13, -3, 0, 3, 13])
def test_alignment_has_signed_linear_not_circular_lags(lag: int) -> None:
    reference, response = np.zeros(32), np.zeros(32)
    reference[15], response[15 + lag] = 1, 1
    assert align_time_shift(reference, response) == lag


def test_alignment_refuses_uninformative_and_ambiguous_records() -> None:
    with pytest.raises(ValueError):
        align_time_shift(np.zeros(16), np.zeros(16))
    with pytest.raises(ValueError):
        align_time_shift(np.array([0, 1, 0]), np.array([1, 0, 1]))


def test_alignment_refuses_complex_input() -> None:
    with pytest.raises(ValueError, match="real"):
        align_time_shift(np.array([1 + 2j, 0]), np.array([0, 1 + 2j]))


@pytest.mark.parametrize("length", [2, True, 3.0])
def test_spectral_estimators_refuse_degenerate_or_noninteger_windows(
    length: Any,
) -> None:
    recording = _recording(samples=np.arange(8.0))
    with pytest.raises(ValueError, match="segment_length"):
        psd_welch(recording, segment_length=length)
    with pytest.raises(ValueError, match="segment_length"):
        estimate_frf_h1(recording, recording, segment_length=length)


@pytest.mark.parametrize("length", [7, 8])
def test_psd_agrees_with_explicit_segment_detrending_and_window(length: int) -> None:
    # Segment-dependent offsets make global detrending an observable error.
    samples = np.arange(41.0) + np.sin(np.arange(41.0))
    recording = _recording(samples=samples)
    expected_f, expected = signal.welch(
        samples,
        fs=recording.sample_rate_hz,
        window=np.hanning(length),
        nperseg=length,
        noverlap=length - length // 2,
        detrend="constant",
        scaling="density",
        average="mean",
    )
    frequencies, density = psd_welch(recording, segment_length=length)
    np.testing.assert_allclose(frequencies, expected_f, rtol=1e-14)
    np.testing.assert_allclose(density, expected, rtol=1e-12, atol=1e-18)


def test_psd_zero_is_finite_but_unexcited_frf_is_refused() -> None:
    quiet = _recording(samples=np.zeros(16))
    _, density = psd_welch(quiet, segment_length=8)
    np.testing.assert_array_equal(density, 0)
    with pytest.raises(ValueError, match="excitation"):
        estimate_frf_h1(quiet, quiet, segment_length=8)


def test_h1_gain_matches_explicit_cross_spectral_reference() -> None:
    rng = np.random.default_rng(5074)
    force = _recording(samples=rng.normal(size=256))
    response = dataclasses.replace(force, samples=-3.25 * force.samples)
    _, magnitude = estimate_frf_h1(force, response, segment_length=32)
    np.testing.assert_allclose(magnitude, 3.25, rtol=1e-12)


def test_zero_ringdown_cannot_produce_nan_parameters() -> None:
    with pytest.raises(ValueError):
        estimate_modal_decay(_recording(samples=np.zeros(16)))


@pytest.mark.parametrize("estimator", [psd_welch, estimate_modal_decay])
def test_spectral_overflow_is_a_refusal(estimator: object) -> None:
    huge = _recording(samples=np.tile([1e308, -1e308], 4096))
    with pytest.raises(ValueError):
        estimator(huge)  # type: ignore[operator]
