"""Calibration conversion, domains and shared uncertainty (Tools #5157)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics import SourceKind
from shared.python.swing_sim.vibroacoustics.waveform_calibration import (
    AcquisitionChannel,
    AcquisitionClock,
    AffineCalibration,
    CalibratedWaveform,
    CalibrationDomain,
    CalibrationEvidence,
    CalibrationRecord,
    GainOffsetUncertainty,
    IndependentSampleUncertainty,
    RawWaveform,
    WaveformProvenance,
)


def raw_example(samples: np.ndarray | None = None) -> RawWaveform:
    """An explicitly synthetic acquisition, with no physical certificate claim."""
    return RawWaveform(
        np.array([-2.0, 0.0, 1.0, 3.0]) if samples is None else samples,
        AcquisitionChannel("sensor-chain-1", "channel-1", "V", "1" * 64),
        AcquisitionClock("clock-1", 0.0, 4096.0, "2" * 64),
        WaveformProvenance("synthetic-record-1", SourceKind.SYNTHESIZED, "3" * 64),
    )


def calibration_example() -> CalibrationRecord:
    return CalibrationRecord(
        "calibration-1",
        AffineCalibration(2.0, -1.0, ("V", "Pa"), GainOffsetUncertainty(0.1, 0.2, 0.4)),
        CalibrationDomain((-5.0, 5.0), (0.0, 1500.0), (-1.0, 1.0), "clock-1"),
        CalibrationEvidence(
            "4" * 64, "synthetic-affine-reference", SourceKind.SYNTHESIZED
        ),
    )


def test_affine_conversion_and_polarity_retain_raw_values() -> None:
    raw = raw_example()
    calibration = calibration_example()
    result = CalibratedWaveform(raw, calibration)
    np.testing.assert_array_equal(result.values, [-5.0, -1.0, 1.0, 5.0])
    np.testing.assert_array_equal(raw.samples, [-2.0, 0.0, 1.0, 3.0])
    assert result.unit == "Pa"
    assert result.source_kind is SourceKind.SYNTHESIZED
    negative = replace(calibration, transform=replace(calibration.transform, gain=-2.0))
    np.testing.assert_array_equal(
        CalibratedWaveform(raw, negative).values, [3.0, -1.0, -3.0, -7.0]
    )


def test_full_covariance_matches_independent_jacobian_matrix() -> None:
    raw, calibration = raw_example(), calibration_example()
    sigma = np.array([0.1, 0.2, 0.3, 0.4])
    result = CalibratedWaveform(raw, calibration, IndependentSampleUncertainty(sigma))
    # Independent GUM first-order J C J^T with explicit gain/offset covariance.
    count = raw.samples.size
    jacobian = np.column_stack((2.0 * np.eye(count), raw.samples, np.ones(count)))
    covariance = np.zeros((count + 2, count + 2))
    covariance[:count, :count] = np.diag(sigma**2)
    covariance[count:, count:] = [[0.01, 0.008], [0.008, 0.04]]
    expected = jacobian @ covariance @ jacobian.T
    actual = np.array(
        [[result.covariance(i, j) for j in range(count)] for i in range(count)]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        result.marginal_standard_uncertainty(), np.sqrt(np.diag(expected)), rtol=1e-13
    )
    weights = np.array([0.5, -0.3, 0.2, 0.6])
    assert result.linear_standard_uncertainty(weights) == pytest.approx(
        np.sqrt(weights @ expected @ weights), rel=1e-13
    )
    assert abs(actual[0, 3]) > 0


def test_common_offset_uncertainty_does_not_average_away() -> None:
    raw = raw_example(np.ones(100))
    calibration = calibration_example()
    transform = replace(
        calibration.transform, uncertainty=GainOffsetUncertainty(0.0, 0.3, 0.0)
    )
    result = CalibratedWaveform(
        raw,
        replace(calibration, transform=transform),
        IndependentSampleUncertainty(np.zeros(100)),
    )
    assert result.linear_standard_uncertainty(np.full(100, 0.01)) == pytest.approx(
        0.3, abs=1e-14
    )
    assert result.covariance(0, 99) == pytest.approx(0.09)


@pytest.mark.parametrize("correlation", [-1.0, 1.0])
def test_singular_valid_calibration_covariance_is_supported(correlation: float) -> None:
    raw, calibration = raw_example(), calibration_example()
    transform = replace(
        calibration.transform, uncertainty=GainOffsetUncertainty(0.1, 0.2, correlation)
    )
    result = CalibratedWaveform(
        raw,
        replace(calibration, transform=transform),
        IndependentSampleUncertainty(np.zeros(4)),
    )
    np.testing.assert_allclose(
        result.marginal_standard_uncertainty(),
        abs(0.1 * raw.samples + correlation * 0.2),
        atol=1e-15,
    )


def test_unknown_uncertainty_remains_unknown_with_explicit_partial_component() -> None:
    result = CalibratedWaveform(raw_example(), calibration_example())
    assert result.covariance(0, 1) is None
    assert result.marginal_standard_uncertainty() is None
    assert result.linear_standard_uncertainty(np.ones(4)) is None
    assert result.calibration_covariance(0, 1) == pytest.approx(0.024)
    calibration = calibration_example()
    absent = replace(
        calibration, transform=replace(calibration.transform, uncertainty=None)
    )
    result = CalibratedWaveform(
        raw_example(), absent, IndependentSampleUncertainty(np.zeros(4))
    )
    assert result.calibration_covariance(0, 1) is None
    assert result.covariance(0, 0) is None


def test_mutation_and_reenabled_writes_cannot_alter_records() -> None:
    samples = np.array([0.0, 1.0, 2.0, 3.0])
    sigma = np.ones(4)
    raw = raw_example(samples)
    result = CalibratedWaveform(
        raw, calibration_example(), IndependentSampleUncertainty(sigma)
    )
    samples[:] = 9
    sigma[:] = 9
    np.testing.assert_array_equal(raw.samples, [0, 1, 2, 3])
    for array in (
        raw.samples,
        result.values,
        result.sample_uncertainty.standard_uncertainty,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_raw_indication_time_clock_and_unit_domains_are_enforced() -> None:
    raw, calibration = raw_example(), calibration_example()
    for altered in (
        replace(raw, samples=np.array([6.0])),
        replace(raw, clock=replace(raw.clock, first_sample_time_s=2.0)),
        replace(raw, clock=replace(raw.clock, clock_id="other-clock")),
        replace(raw, channel=replace(raw.channel, unit="counts")),
    ):
        with pytest.raises(ValueError):
            CalibratedWaveform(altered, calibration)
    endpoint = replace(
        calibration, domain=replace(calibration.domain, time_interval_s=(0.0, 2 / 4096))
    )
    with pytest.raises(ValueError, match="time"):
        CalibratedWaveform(raw, endpoint)


def test_frequency_band_is_retained_without_inferred_filter_or_phase() -> None:
    raw, calibration = raw_example(), calibration_example()
    broad = replace(
        calibration,
        domain=replace(calibration.domain, frequency_band_hz=(20.0, 10000.0)),
    )
    result = CalibratedWaveform(raw, broad)
    assert result.valid_frequency_band_hz == (20.0, 2048.0)
    np.testing.assert_array_equal(result.values, [-5, -1, 1, 5])
    unsupported = replace(
        calibration,
        domain=replace(calibration.domain, frequency_band_hz=(3000.0, 4000.0)),
    )
    with pytest.raises(ValueError, match="band"):
        CalibratedWaveform(raw, unsupported)


@pytest.mark.parametrize("value", [True, 0.0, float("inf"), float("nan"), "2"])
def test_invalid_gain_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(calibration_example().transform, gain=value)


@pytest.mark.parametrize("value", [-1.0, True, float("nan"), float("inf")])
def test_invalid_standard_uncertainty_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GainOffsetUncertainty(value, 0.1, 0.0)


def test_invalid_uncertainty_shape_indices_and_weights_are_refused() -> None:
    raw, calibration = raw_example(), calibration_example()
    with pytest.raises(ValueError, match="length"):
        CalibratedWaveform(raw, calibration, IndependentSampleUncertainty(np.ones(3)))
    result = CalibratedWaveform(raw, calibration)
    for index in (True, -1, 4, 1.5):
        with pytest.raises((TypeError, ValueError)):
            result.covariance(index, 0)
    for weights in (np.ones(3), np.array([1j] * 4), np.array([float("nan")] * 4)):
        with pytest.raises((TypeError, ValueError)):
            result.linear_standard_uncertainty(weights)


def test_overflow_and_inadequate_clock_resolution_are_refused() -> None:
    raw, calibration = raw_example(), calibration_example()
    large = replace(calibration, transform=replace(calibration.transform, gain=1e308))
    with pytest.raises(ValueError, match="finite"):
        CalibratedWaveform(raw, large)
    with pytest.raises(ValueError, match="resolution"):
        replace(raw, clock=replace(raw.clock, first_sample_time_s=1e18))
