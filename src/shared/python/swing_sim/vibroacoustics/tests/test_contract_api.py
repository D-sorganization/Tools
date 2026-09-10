"""Pinned downstream-facing API for vibroacoustics (IA-T5, #5074)."""

from __future__ import annotations

import pytest

import shared.python.swing_sim.vibroacoustics as vibro

EXPECTED_PUBLIC_API = {
    "BandwidthReport",
    "ClippingReport",
    "H1Bin",
    "H1Estimate",
    "H1Settings",
    "SourceKind",
    "SynthesizedSourceError",
    "WaveformRecording",
    "align_time_shift",
    "as_measured",
    "bandwidth_report",
    "clipping_fraction",
    "estimate_complex_frf_h1",
    "estimate_frf_h1",
    "estimate_modal_decay",
    "psd_welch",
    "raw_data_hash",
}


@pytest.mark.contract
def test_public_api_is_explicit_and_pinned() -> None:
    assert set(vibro.__all__) == EXPECTED_PUBLIC_API
    for symbol in vibro.__all__:
        assert getattr(vibro, symbol) is not None
