"""Adverse, metamorphic, and countermodel tests for response-field claims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation.group_spec import PerturbationGroup
from shared.python.swing_sim.variation.noise_response import (
    ADEQUACY_ESTIMABLE,
    ADEQUACY_INSUFFICIENT_PAIRS,
    ADEQUACY_UNSUPPORTED_CORRELATED,
    ResponseFieldAccumulator,
    compute_position_noise_response_field,
)
from shared.python.swing_sim.variation.noise_response_record import (
    PositionNoiseResponseField,
)
from shared.python.swing_sim.variation.spec import NoiseSpec
from shared.python.swing_sim.variation.trace_resampling import TraceResamplingResult

from .noise_response_test_support import (
    BALL_SPEED,
    LAUNCH_ANGLE,
    ResponseFixtureConfig,
    build_response_inputs,
    default_fixture_config,
)


def _field(config: ResponseFixtureConfig) -> PositionNoiseResponseField:
    return compute_position_noise_response_field(build_response_inputs(config))


def test_rigid_translation_changes_neither_response_nor_scatter() -> None:
    config = default_fixture_config()
    translated = replace(
        config,
        baseline_positions_m=config.baseline_positions_m + np.array([7.0, -3.0, 11.0]),
    )

    reference = _field(config)
    shifted = _field(translated)

    np.testing.assert_allclose(
        shifted.signed_response_m_per_declared_scale,
        reference.signed_response_m_per_declared_scale,
    )
    np.testing.assert_allclose(
        shifted.matched_absolute_rms_scatter_m,
        reference.matched_absolute_rms_scatter_m,
    )


def test_rotation_transforms_signed_components_and_preserves_magnitudes() -> None:
    config = default_fixture_config()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(
        config,
        baseline_positions_m=config.baseline_positions_m @ rotation.T,
        coefficients=config.coefficients @ rotation.T,
    )

    reference = _field(config)
    transformed = _field(rotated)

    np.testing.assert_allclose(
        transformed.signed_response_m_per_declared_scale,
        reference.signed_response_m_per_declared_scale @ rotation.T,
    )
    np.testing.assert_allclose(
        transformed.response_magnitude_m_per_declared_scale,
        reference.response_magnitude_m_per_declared_scale,
    )
    np.testing.assert_allclose(
        transformed.matched_absolute_rms_scatter_m,
        reference.matched_absolute_rms_scatter_m,
    )


def test_position_unit_scaling_transforms_every_geometric_metric() -> None:
    config = default_fixture_config()
    scale = 1000.0
    scaled = replace(
        config,
        baseline_positions_m=config.baseline_positions_m * scale,
        coefficients=config.coefficients * scale,
    )

    reference = _field(config)
    transformed = _field(scaled)

    np.testing.assert_allclose(
        transformed.signed_response_m_per_declared_scale,
        reference.signed_response_m_per_declared_scale * scale,
    )
    np.testing.assert_allclose(
        transformed.response_magnitude_m_per_declared_scale,
        reference.response_magnitude_m_per_declared_scale * scale,
    )
    np.testing.assert_allclose(
        transformed.matched_absolute_rms_scatter_m,
        reference.matched_absolute_rms_scatter_m * scale,
    )


def test_equal_scatter_can_have_different_declared_scale_response() -> None:
    config = default_fixture_config()
    altered_coefficients = config.coefficients.copy()
    altered_coefficients[0] *= 2.0
    altered_specs = (
        NoiseSpec(BALL_SPEED, scale=4.0, spec_id="input.ball-speed"),
        NoiseSpec(LAUNCH_ANGLE, scale=4.0, spec_id="input.launch-angle"),
    )
    altered = replace(
        config,
        coefficients=altered_coefficients,
        specs=altered_specs,
    )

    reference = _field(config)
    transformed = _field(altered)

    np.testing.assert_allclose(
        transformed.matched_absolute_rms_scatter_m[0],
        reference.matched_absolute_rms_scatter_m[0],
    )
    np.testing.assert_allclose(
        transformed.response_magnitude_m_per_declared_scale[0],
        2.0 * reference.response_magnitude_m_per_declared_scale[0],
    )


def test_equal_response_can_have_different_exogenous_absolute_scatter() -> None:
    config = default_fixture_config()
    residual = np.zeros_like(config.baseline_positions_m)
    residual[:, :, :, 2] = np.array([-0.3, -0.1, 0.1, 0.3])[:, None, None]
    noisy = replace(config, baseline_positions_m=residual)

    reference = _field(config)
    transformed = _field(noisy)

    np.testing.assert_allclose(
        transformed.signed_response_m_per_declared_scale,
        reference.signed_response_m_per_declared_scale,
    )
    assert not np.allclose(
        transformed.matched_absolute_rms_scatter_m,
        reference.matched_absolute_rms_scatter_m,
    )
    assert np.mean(transformed.matched_absolute_rms_scatter_m) > np.mean(
        reference.matched_absolute_rms_scatter_m
    )


def test_missing_rows_preserve_matched_and_all_eligible_denominators() -> None:
    config = default_fixture_config()
    baseline_valid = np.array(
        [[True, True], [True, False], [True, False], [False, False]]
    )
    missing = replace(config, baseline_valid=baseline_valid)

    field = _field(missing)

    np.testing.assert_array_equal(field.availability_count[:, :, 0], [[3, 1], [3, 1]])
    np.testing.assert_array_equal(field.all_eligible_count, 4)
    np.testing.assert_array_equal(field.adequacy[:, 0], ADEQUACY_ESTIMABLE)
    np.testing.assert_array_equal(field.adequacy[:, 1], ADEQUACY_INSUFFICIENT_PAIRS)
    assert np.all(np.isnan(field.signed_response_m_per_declared_scale[:, 1]))
    assert np.all(np.isfinite(field.all_eligible_absolute_rms_scatter_m))


def test_nonlinear_interaction_countermodel_is_not_promoted_to_oat_response() -> None:
    x = np.array([0.25, 0.5, 0.75, 1.0])
    oat_at_zero_partner = x * 0.0
    simultaneous = x * x
    oat_slope = float(np.dot(x, oat_at_zero_partner) / np.dot(x, x))
    simultaneous_slope = float(np.dot(x, simultaneous) / np.dot(x, x))
    assert oat_slope == 0.0
    assert simultaneous_slope > 0.0

    config = default_fixture_config()
    grouped = replace(
        config,
        groups=(
            PerturbationGroup(
                group_id="interaction-design",
                spec_ids=("input.ball-speed", "input.launch-angle"),
                matrix=((1.0, 0.5), (0.5, 1.0)),
            ),
        ),
    )
    field = _field(grouped)
    np.testing.assert_array_equal(field.adequacy, ADEQUACY_UNSUPPORTED_CORRELATED)


def test_trial_order_frame_registry_and_resume_contract_drift_fail_closed() -> None:
    inputs = build_response_inputs(default_fixture_config())
    first = inputs[0]
    with pytest.raises(ContractViolationError, match="trial order mismatch"):
        replace(first, perturbed_trial_ids=tuple(reversed(first.trial_ids)))

    changed_trace = replace(first.perturbed.traces, coordinate_frame="swing.other")
    changed_result = TraceResamplingResult(
        changed_trace, first.perturbed.impact_alignment_error_s
    )
    with pytest.raises(ContractViolationError, match="frame mismatch"):
        replace(first, perturbed=changed_result)

    changed_metadata = replace(first.execution_metadata, registry_sha256="b" * 64)
    with pytest.raises(ContractViolationError, match="registry digest mismatch"):
        replace(first, execution_metadata=changed_metadata)

    accumulator = ResponseFieldAccumulator(inputs)
    accumulator.accept_trial_slice(0, 2)
    snapshot = accumulator.snapshot()
    crossed = list(inputs)
    crossed[0] = replace(crossed[0], source_sha256="c" * 64)
    with pytest.raises(ContractViolationError, match="snapshot contract drift"):
        ResponseFieldAccumulator.from_snapshot(tuple(crossed), snapshot)


def test_incomplete_design_and_zero_declared_scale_fail_closed() -> None:
    inputs = build_response_inputs(default_fixture_config())
    with pytest.raises(ContractViolationError, match="incomplete input design"):
        compute_position_noise_response_field(inputs[:1])
    with pytest.raises(ContractViolationError, match="scale must be finite and > 0"):
        NoiseSpec(BALL_SPEED, scale=0.0)
