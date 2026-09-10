"""Vibroacoustic measurement ingestion for impact studies (IA-T5, #5074).

Imports calibrated pressure/waveform measurements with provenance and
provides spectral estimation.  Synthesized sources are structurally
distinct and refused as measurements: synthesized tones are never
labelled as predicted acoustics.  Radiation and observer modeling live
in a later, independently qualified tier.
"""

from __future__ import annotations

from shared.python.swing_sim.vibroacoustics.frf_estimation import (
    H1Bin,
    H1Estimate,
    H1Settings,
    estimate_complex_frf_h1,
)
from shared.python.swing_sim.vibroacoustics.measurement import (
    BandwidthReport,
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
from shared.python.swing_sim.vibroacoustics.spectral import (
    ModalDecayFit,
    estimate_frf_h1,
    estimate_modal_decay,
    psd_welch,
)

__all__ = [
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
]
