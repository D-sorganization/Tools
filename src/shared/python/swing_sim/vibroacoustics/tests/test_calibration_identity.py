"""Full-precision provenance identity and legacy preservation (#5157)."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics import (
    SourceKind,
    WaveformRecording,
    raw_data_hash,
)
from shared.python.swing_sim.vibroacoustics.waveform_calibration import (
    CalibratedWaveform,
    IndependentSampleUncertainty,
)

from .test_waveform_calibration import calibration_example, raw_example


def test_identity_binds_every_acquisition_component_and_samples() -> None:
    raw, calibration = raw_example(), calibration_example()
    baseline = CalibratedWaveform(raw, calibration).identity_sha256
    altered = (
        replace(raw, samples=raw.samples + 1e-12),
        replace(raw, channel=replace(raw.channel, sensor_chain_id="different")),
        replace(raw, channel=replace(raw.channel, channel_id="different")),
        replace(raw, channel=replace(raw.channel, setup_sha256="a" * 64)),
        replace(raw, clock=replace(raw.clock, first_sample_time_s=1e-12)),
        replace(raw, clock=replace(raw.clock, sample_rate_hz=4096.000000001)),
        replace(raw, clock=replace(raw.clock, timing_evidence_sha256="b" * 64)),
        replace(raw, provenance=replace(raw.provenance, source_id="different")),
        replace(
            raw, provenance=replace(raw.provenance, source_kind=SourceKind.MEASURED)
        ),
        replace(raw, provenance=replace(raw.provenance, raw_file_sha256="c" * 64)),
    )
    identities = {
        CalibratedWaveform(item, calibration).identity_sha256 for item in altered
    }
    assert baseline not in identities
    assert len(identities) == len(altered)
    assert len(baseline) == 64


def test_identity_binds_sub_quantization_calibration_change_even_at_zero_input() -> (
    None
):
    raw, calibration = raw_example(np.zeros(4)), calibration_example()
    first = CalibratedWaveform(raw, calibration)
    changed = replace(
        calibration,
        transform=replace(calibration.transform, gain=float(np.nextafter(2.0, 3.0))),
    )
    second = CalibratedWaveform(raw, changed)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.identity_sha256 != second.identity_sha256
    assert first.identity_sha256 == CalibratedWaveform(raw, calibration).identity_sha256


def test_identity_binds_uncertainty_domain_and_certificate() -> None:
    raw, calibration = raw_example(), calibration_example()
    baseline = CalibratedWaveform(raw, calibration).identity_sha256
    alternatives = (
        replace(calibration, calibration_id="other"),
        replace(
            calibration,
            evidence=replace(calibration.evidence, certificate_sha256="a" * 64),
        ),
        replace(
            calibration,
            evidence=replace(calibration.evidence, source_kind=SourceKind.MEASURED),
        ),
        replace(
            calibration,
            domain=replace(calibration.domain, indication_range=(-6.0, 6.0)),
        ),
        replace(
            calibration, transform=replace(calibration.transform, uncertainty=None)
        ),
        replace(
            calibration, transform=replace(calibration.transform, offset=-1.0 + 1e-12)
        ),
    )
    assert all(
        CalibratedWaveform(raw, item).identity_sha256 != baseline
        for item in alternatives
    )
    known = CalibratedWaveform(
        raw, calibration, IndependentSampleUncertainty(np.zeros(4))
    )
    assert known.identity_sha256 != baseline


def test_synthetic_calibration_cannot_promote_measured_input() -> None:
    raw = raw_example()
    measured = replace(
        raw, provenance=replace(raw.provenance, source_kind=SourceKind.MEASURED)
    )
    result = CalibratedWaveform(measured, calibration_example())
    assert result.source_kind is SourceKind.SYNTHESIZED


@pytest.mark.parametrize("value", ["", " ", "bad\nidentifier", "\ud800"])
def test_identity_labels_refuse_ambiguous_text(value: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(raw_example().channel, sensor_chain_id=value)


def test_legacy_hash_bytes_are_unchanged() -> None:
    legacy = WaveformRecording(
        np.array([1.0, -2.0]), 4096.0, "V", "legacy", 0.5, "raw", SourceKind.SYNTHESIZED
    )
    # Legacy intentionally omits sensitivity; the new identity is separate.
    assert raw_data_hash(legacy) == raw_data_hash(
        replace(legacy, sensitivity_per_unit=2.0)
    )
    assert raw_data_hash(legacy) == (
        "d52731135e7064ab1e8dd46a194eb73373347ca360f672a31ab9bf8304103ed0"
    )
    assert legacy.samples.tobytes() == np.array([1.0, -2.0]).tobytes()
