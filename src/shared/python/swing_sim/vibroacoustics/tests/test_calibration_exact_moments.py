"""Exact independent-block second moments versus the GUM approximation."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.vibroacoustics.waveform_calibration import (
    CalibratedWaveform,
    GainOffsetUncertainty,
    IndependentSampleUncertainty,
    UncertaintyPropagation,
)

from .test_waveform_calibration import calibration_example, raw_example


def test_exact_moments_retain_gain_variance_times_indication_covariance() -> None:
    raw, calibration = raw_example(), calibration_example()
    sigma = np.array([0.1, 0.2, 0.3, 0.4])
    source_uncertainty = IndependentSampleUncertainty(sigma)
    approximate = CalibratedWaveform(raw, calibration, source_uncertainty)
    exact = CalibratedWaveform(
        raw, calibration, source_uncertainty, UncertaintyPropagation.EXACT_INDEPENDENT
    )
    count = raw.size
    # E[G^2] E[XX'] + E[GB](mu_X + mu_X') + E[B^2] - mu_Y mu_Y'.
    second_x = np.diag(sigma**2) + np.outer(raw.samples, raw.samples)
    sum_x = raw.samples[:, None] + raw.samples[None, :]
    mean_y = 2 * raw.samples - 1
    expected = 4.01 * second_x + (-2 + 0.008) * sum_x + 1.04 - np.outer(mean_y, mean_y)
    observed = np.array(
        [[exact.covariance(i, j) for j in range(count)] for i in range(count)]
    )
    first_order = np.array(
        [[approximate.covariance(i, j) for j in range(count)] for i in range(count)]
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        observed - first_order, 0.01 * np.diag(sigma**2), atol=1e-15
    )
    np.testing.assert_allclose(
        exact.marginal_standard_uncertainty(), np.sqrt(np.diag(expected)), rtol=1e-12
    )
    weights = np.array([0.2, -0.1, 0.4, 0.5])
    assert exact.linear_standard_uncertainty(weights) == pytest.approx(
        np.sqrt(weights @ expected @ weights), rel=1e-12
    )
    assert exact.identity_sha256 != approximate.identity_sha256


def test_exact_affine_variance_agrees_with_complete_discrete_ensemble() -> None:
    # Independent two-point G and X are non-Gaussian; enumerate every outcome.
    raw = raw_example(np.array([1.0]))
    calibration = calibration_example()
    transform = replace(
        calibration.transform,
        gain=2.0,
        offset=0.0,
        uncertainty=GainOffsetUncertainty(1.0, 0.0, 0.0),
    )
    result = CalibratedWaveform(
        raw,
        replace(calibration, transform=transform),
        IndependentSampleUncertainty(np.array([0.5])),
        UncertaintyPropagation.EXACT_INDEPENDENT,
    )
    outcomes = np.array(
        [gain * indication for gain in (1.0, 3.0) for indication in (0.5, 1.5)]
    )
    assert result.values[0] == pytest.approx(float(np.mean(outcomes)))
    assert result.covariance(0, 0) == pytest.approx(float(np.var(outcomes)))
    assert result.covariance(0, 0) == pytest.approx(2.25)


def test_propagation_mode_requires_enum_and_keeps_unknown_components() -> None:
    with pytest.raises(TypeError, match="propagation"):
        CalibratedWaveform(raw_example(), calibration_example(), propagation="exact")
    result = CalibratedWaveform(
        raw_example(),
        calibration_example(),
        propagation=UncertaintyPropagation.EXACT_INDEPENDENT,
    )
    assert result.covariance(0, 0) is None
