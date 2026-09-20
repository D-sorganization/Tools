"""Tests for calibrated sound recordings and synchronization (IA-T5, #5074)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics._acquisition_records import (
    AcquisitionChannel,
    AcquisitionClock,
    RawWaveform,
    WaveformProvenance,
)
from shared.python.swing_sim.vibroacoustics._affine_calibration_records import (
    AffineCalibration,
    CalibrationDomain,
    CalibrationEvidence,
    CalibrationRecord,
)
from shared.python.swing_sim.vibroacoustics.calibrated_sound import (
    CalibratedSoundRecording,
    synchronize_timebases,
)
from shared.python.swing_sim.vibroacoustics.measurement import (
    SourceKind,
    SynthesizedSourceError,
)
from shared.python.swing_sim.vibroacoustics.observer import ObserverLocation
from shared.python.swing_sim.vibroacoustics.waveform_calibration import (
    CalibratedWaveform,
)


def _create_mock_calibrated_waveform(
    samples: np.ndarray,
    unit: str = "Pa",
    source_kind: SourceKind = SourceKind.MEASURED,
) -> CalibratedWaveform:
    clock = AcquisitionClock(
        clock_id="master_daq_clock",
        first_sample_time_s=0.0,
        sample_rate_hz=48000.0,
        timing_evidence_sha256="b" * 64,
    )
    channel = AcquisitionChannel(
        sensor_chain_id="GRAS_46AE_PXIe4499",
        channel_id="mic_ch1",
        unit="V",
        setup_sha256="a" * 64,
    )
    provenance = WaveformProvenance(
        source_id="impact_test_01",
        source_kind=source_kind,
        raw_file_sha256="c" * 64,
    )
    raw = RawWaveform(
        channel=channel,
        clock=clock,
        provenance=provenance,
        samples=samples,
    )
    transform = AffineCalibration(
        gain=50.0,  # 50 Pa / V (20 mV/Pa sensitivity)
        offset=0.0,
        units=("V", unit),
        uncertainty=None,
    )
    domain = CalibrationDomain(
        indication_range=(-10.0, 10.0),
        frequency_band_hz=(20.0, 20000.0),
        time_interval_s=(-1.0, 100.0),
        clock_id="master_daq_clock",
    )
    evidence = CalibrationEvidence(
        certificate_sha256="d" * 64,
        method_id="IEC_61094_4",
        source_kind=source_kind,
    )
    calibration = CalibrationRecord(
        calibration_id="NIST_2026_MIC_01",
        transform=transform,
        domain=domain,
        evidence=evidence,
    )
    return CalibratedWaveform(raw=raw, calibration=calibration)


@pytest.mark.unit
def test_calibrated_sound_recording_preserves_identity_and_units() -> None:
    samples = np.array([0.01, 0.02, 0.01, -0.01, -0.02])
    waveform = _create_mock_calibrated_waveform(samples, unit="Pa")
    obs = ObserverLocation(name="golfer_ear", coordinates_m=np.array([0.0, -0.3, 1.6]))

    recording = CalibratedSoundRecording(waveform=waveform, observer=obs)

    assert recording.unit == "Pa"
    assert recording.identity_sha256 == waveform.identity_sha256
    assert recording.observer.name == "golfer_ear"
    assert recording.as_measured() is recording


@pytest.mark.unit
def test_calibrated_sound_refuses_non_pressure_units() -> None:
    samples = np.array([1.0, 2.0, 3.0])
    waveform = _create_mock_calibrated_waveform(
        samples, unit="N"
    )  # Force, not pressure
    obs = ObserverLocation(name="mic", coordinates_m=np.array([1.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="unit must be 'Pa'"):
        CalibratedSoundRecording(waveform=waveform, observer=obs)


@pytest.mark.unit
def test_calibrated_sound_refuses_synthesized_as_measured() -> None:
    samples = np.array([0.01, -0.01])
    waveform = _create_mock_calibrated_waveform(
        samples, unit="Pa", source_kind=SourceKind.SYNTHESIZED
    )
    obs = ObserverLocation(name="mic", coordinates_m=np.array([1.0, 0.0, 0.0]))
    recording = CalibratedSoundRecording(waveform=waveform, observer=obs)

    assert recording.source_kind == SourceKind.SYNTHESIZED
    with pytest.raises(SynthesizedSourceError, match="synthesized"):
        recording.as_measured()


@pytest.mark.unit
def test_phase_sensitive_timebase_synchronization() -> None:
    # 2 recordings with known distance and delay
    sample_rate_hz = 48000.0
    t = np.arange(0, 0.05, 1.0 / sample_rate_hz)
    # Reference pulse
    pulse = np.exp(-((t - 0.01) ** 2) / (2 * (0.0005**2)))

    # Receiver at distance 0.6864 m -> expected delay 0.6864 / 343.2
    # = 0.002 s = 96 samples
    delay_samples = 96
    observed_pulse = np.roll(pulse, delay_samples)

    ref_wav = _create_mock_calibrated_waveform(pulse / 50.0)
    obs_wav = _create_mock_calibrated_waveform(observed_pulse / 50.0)

    ref_rec = CalibratedSoundRecording(
        waveform=ref_wav,
        observer=ObserverLocation(
            name="source", coordinates_m=np.array([0.0, 0.0, 0.0])
        ),
    )
    obs_rec = CalibratedSoundRecording(
        waveform=obs_wav,
        observer=ObserverLocation(
            name="mic", coordinates_m=np.array([0.6864, 0.0, 0.0])
        ),
    )

    residual_lag, propagation_delay_s = synchronize_timebases(
        reference=ref_rec,
        observed=obs_rec,
        distance_m=0.6864,
        sound_speed_mps=343.2,
    )

    # Propagation delay should be 2 ms (96 samples)
    assert propagation_delay_s == pytest.approx(0.002, abs=1e-5)
    # Residual lag after removing acoustic propagation delay should be 0!
    assert residual_lag == 0
