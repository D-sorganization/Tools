"""Boundary and representability controls for declared calibration (#5157)."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics.waveform_calibration import (
    CalibratedWaveform,
    GainOffsetUncertainty,
    IndependentSampleUncertainty,
)

from .test_waveform_calibration import calibration_example, raw_example


@pytest.mark.parametrize("value", [0.0, -1.0, True, float("inf"), float("nan")])
def test_invalid_clock_rate_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(raw_example().clock, sample_rate_hz=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("indication_range", [-1, 1]),
        ("indication_range", (1.0, 1.0)),
        ("time_interval_s", (1.0, -1.0)),
        ("time_interval_s", (0.0, float("inf"))),
        ("frequency_band_hz", (-1.0, 100.0)),
        ("clock_id", " clock-1"),
    ],
)
def test_invalid_calibration_domain_is_refused(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(calibration_example().domain, **{field: value})


@pytest.mark.parametrize("value", [2.0, -2.0, True, float("nan")])
def test_invalid_coefficient_correlation_is_refused(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GainOffsetUncertainty(0.1, 0.1, value)


def test_undefined_correlation_of_zero_uncertainty_is_not_invented() -> None:
    with pytest.raises(ValueError, match="zero"):
        GainOffsetUncertainty(0.0, 0.1, 0.3)


@pytest.mark.parametrize("value", ["", "a" * 63, "A" * 64, "g" * 64])
def test_invalid_certificate_digest_is_refused(value: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(calibration_example().evidence, certificate_sha256=value)


def test_domain_endpoints_are_inclusive_and_conversion_is_not_reapplied() -> None:
    raw, calibration = raw_example(np.array([-5.0, 5.0])), calibration_example()
    domain = replace(calibration.domain, time_interval_s=(0.0, 1 / 4096))
    result = CalibratedWaveform(raw, replace(calibration, domain=domain))
    np.testing.assert_array_equal(result.values, [-11, 9])
    with pytest.raises(TypeError, match="RawWaveform"):
        CalibratedWaveform(result, calibration)


@pytest.mark.parametrize(
    "samples",
    [
        np.array([1j]),
        np.array([True]),
        np.array([float("nan")]),
        np.array([[1.0]]),
        np.array([]),
    ],
)
def test_raw_samples_are_not_silently_coerced(samples: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        raw_example(samples)


def test_uncertainty_model_requires_explicit_independence_type() -> None:
    with pytest.raises(TypeError, match="IndependentSampleUncertainty"):
        CalibratedWaveform(raw_example(), calibration_example(), np.ones(4))
    with pytest.raises(ValueError):
        IndependentSampleUncertainty(np.array([-1.0]))


def test_large_finite_standard_uncertainty_need_not_form_its_variance() -> None:
    raw, calibration = raw_example(), calibration_example()
    transform = replace(
        calibration.transform, uncertainty=GainOffsetUncertainty(1e200, 0.0, 0.0)
    )
    result = CalibratedWaveform(
        raw,
        replace(calibration, transform=transform),
        IndependentSampleUncertainty(np.zeros(4)),
    )
    np.testing.assert_allclose(
        result.marginal_standard_uncertainty(), abs(raw.samples) * 1e200
    )
    assert result.linear_standard_uncertainty(
        np.array([0.0, 0.0, 1.0, 0.0])
    ) == pytest.approx(1e200)
    with pytest.raises(ValueError, match="finite"):
        result.covariance(2, 2)


def test_record_types_cannot_be_replaced_by_unvalidated_mappings() -> None:
    raw, calibration = raw_example(), calibration_example()
    for field in ("channel", "clock", "provenance"):
        with pytest.raises(TypeError):
            replace(raw, **{field: {}})
    for field in ("transform", "domain", "evidence"):
        with pytest.raises(TypeError):
            replace(calibration, **{field: {}})
    with pytest.raises(TypeError):
        replace(calibration.transform, units=["V", "Pa"])
    with pytest.raises((TypeError, ValueError)):
        replace(raw.provenance, source_kind="measured")
