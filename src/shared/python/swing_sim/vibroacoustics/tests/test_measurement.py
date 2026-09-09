"""Calibrated measurement recording contract (IA-T5, #5074)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics import (
    ClippingReport,
    SourceKind,
    SynthesizedSourceError,
    WaveformRecording,
    align_time_shift,
    as_measured,
    bandwidth_report,
    clipping_fraction,
    raw_data_hash,
)


def _sine(
    n: int = 256,
    freq_hz: float = 40.0,
    rate: float = 1024.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = np.arange(n) / rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _recording(**overrides: object) -> WaveformRecording:
    # Adversarial contract tests intentionally pass values outside the API types.
    defaults: dict[str, Any] = {
        "samples": _sine(),
        "sample_rate_hz": 1024.0,
        "unit": "Pa",
        "calibration_id": "cal-1",
        "sensitivity_per_unit": 0.5,
        "source_id": "rec-001",
        "source_kind": SourceKind.MEASURED,
    }
    defaults.update(overrides)
    return WaveformRecording(**defaults)


def test_recording_round_trips_calibrated_provenance() -> None:
    recording = _recording()
    assert recording.unit == "Pa"
    assert recording.calibration_id == "cal-1"
    assert recording.sensitivity_per_unit == 0.5
    assert recording.sample_rate_hz == 1024.0
    assert recording.source_kind is SourceKind.MEASURED


def test_non_finite_or_multidimensional_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        _recording(samples=np.array([0.0, float("nan"), 1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        _recording(samples=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        _recording(samples=np.array([float("inf")]))


def test_sample_rate_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        _recording(sample_rate_hz=0.0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        _recording(sample_rate_hz=float("inf"))


def test_raw_data_hash_is_deterministic_and_content_bound() -> None:
    first = raw_data_hash(_recording())
    second = raw_data_hash(_recording())
    changed = raw_data_hash(_recording(samples=_sine(amplitude=1.001)))
    assert first == second
    assert first != changed
    assert len(first) == 64


def test_clipping_fraction_counts_samples_at_or_over_rail() -> None:
    samples = np.array([0.5, 1.0, -1.0, 0.25, 1.5, -1.5, 0.0, 0.75])
    report = clipping_fraction(_recording(samples=samples), rail=1.0)
    assert isinstance(report, ClippingReport)
    assert report.clipped_samples == 4
    assert report.total_samples == 8
    assert report.fraction == pytest.approx(0.5)
    assert report.clipped is True


def test_clean_signal_is_not_clipped() -> None:
    report = clipping_fraction(_recording(), rail=2.0)
    assert report.clipped is False
    assert report.fraction == 0.0


def test_bandwidth_report_exposes_nyquist_margin() -> None:
    report = bandwidth_report(_recording(), max_expected_hz=400.0)
    assert report.nyquist_hz == pytest.approx(512.0)
    assert report.max_expected_hz == pytest.approx(400.0)
    assert report.adequate is True
    assert bandwidth_report(_recording(), max_expected_hz=600.0).adequate is False


def test_time_alignment_recovers_known_integer_delay() -> None:
    reference = np.zeros(512)
    reference[100:140] = _sine(40, amplitude=1.0)
    delayed = np.zeros(512)
    delayed[177:217] = reference[100:140]
    lag = align_time_shift(reference, delayed)
    assert lag == 77


def test_alignment_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        align_time_shift(np.zeros(16), np.zeros(10))


def test_as_measured_passes_measured_sources() -> None:
    recording = _recording()
    assert as_measured(recording) is recording


def test_synthesized_sources_are_refused_as_measurements() -> None:
    synthesized = _recording(source_kind=SourceKind.SYNTHESIZED)
    with pytest.raises(SynthesizedSourceError, match="synthesized"):
        as_measured(synthesized)
