"""Golden and immutable contracts for the geometric noise-response field."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.variation.noise_response import (
    ADEQUACY_ESTIMABLE,
    ADEQUACY_UNSUPPORTED_BOUNDED,
    ADEQUACY_UNSUPPORTED_CORRELATED,
    ADEQUACY_UNSUPPORTED_DISCRETE,
    ADEQUACY_ZERO_PERTURBATION,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID,
    POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION,
    ResponseFieldAccumulator,
    compute_position_noise_response_field,
    response_field_fingerprint,
)

from .noise_response_test_support import (
    BALL_SPEED,
    LAUNCH_ANGLE,
    ResponseFixtureConfig,
    build_response_inputs,
    default_fixture_config,
)


def test_affine_field_recovers_signed_response_and_explicit_metadata() -> None:
    config = default_fixture_config()

    field = compute_position_noise_response_field(build_response_inputs(config))

    assert field.schema_id == POSITION_NOISE_RESPONSE_FIELD_SCHEMA_ID
    assert field.schema_version == POSITION_NOISE_RESPONSE_FIELD_SCHEMA_VERSION
    assert field.resampling_policy_id == "swing-trace-time-linear-contiguous/v1"
    assert field.coordinate_kind == "time"
    assert field.coordinate_unit == "s"
    assert field.position_unit == "m"
    assert field.input_ids == ("input.ball-speed", "input.launch-angle")
    assert field.input_units == ("mph", "deg")
    np.testing.assert_allclose(field.input_declared_scales, [2.0, 4.0])
    np.testing.assert_allclose(field.input_normalization_scales, [2.0, 4.0])
    np.testing.assert_array_equal(field.availability_count, 4)
    np.testing.assert_array_equal(field.adequacy, ADEQUACY_ESTIMABLE)
    np.testing.assert_allclose(
        field.signed_response_m_per_declared_scale, config.coefficients
    )
    np.testing.assert_allclose(
        field.response_magnitude_m_per_declared_scale,
        np.linalg.norm(config.coefficients, axis=-1),
    )
    assert field.metric_ids == (
        "signed-cartesian-response",
        "response-magnitude",
        "matched-absolute-rms-scatter",
        "all-eligible-absolute-rms-scatter",
    )
    assert field.method_id == "paired-oat-linear-through-origin/v1"
    assert "not causal" in field.scientific_boundary.lower()


def test_field_is_immutable_and_fingerprint_binds_arrays_and_provenance() -> None:
    field = compute_position_noise_response_field(
        build_response_inputs(default_fixture_config())
    )

    fingerprint = response_field_fingerprint(field)

    assert len(fingerprint) == 64
    assert fingerprint == response_field_fingerprint(field)
    with pytest.raises(ValueError, match="read-only"):
        field.signed_response_m_per_declared_scale[0, 0, 0, 0] = 9.0
    assert field.source_sha256 == ("a" * 64, "a" * 64)
    assert len(set(field.plan_sha256)) == 1
    assert len(set(field.registry_sha256)) == 1


def test_zero_perturbation_is_non_estimable_not_zero_response() -> None:
    config = default_fixture_config()
    zero = ResponseFixtureConfig(
        deltas=np.zeros_like(config.deltas),
        coefficients=config.coefficients,
        baseline_positions_m=config.baseline_positions_m,
        baseline_valid=config.baseline_valid,
        perturbed_valid=config.perturbed_valid,
    )

    field = compute_position_noise_response_field(build_response_inputs(zero))

    np.testing.assert_array_equal(field.adequacy, ADEQUACY_ZERO_PERTURBATION)
    assert np.all(np.isnan(field.signed_response_m_per_declared_scale))
    assert np.all(np.isnan(field.response_magnitude_m_per_declared_scale))
    assert np.all(np.isfinite(field.matched_absolute_rms_scatter_m))


@pytest.mark.parametrize(
    ("kind", "bounded", "correlated", "expected"),
    [
        ("discrete", False, False, ADEQUACY_UNSUPPORTED_DISCRETE),
        ("continuous", True, False, ADEQUACY_UNSUPPORTED_BOUNDED),
        ("continuous", False, True, ADEQUACY_UNSUPPORTED_CORRELATED),
    ],
)
def test_unsupported_input_designs_are_explicit_nan_fields(
    kind: str, bounded: bool, correlated: bool, expected: str
) -> None:
    from shared.python.swing_sim.variation.group_spec import PerturbationGroup
    from shared.python.swing_sim.variation.spec import NoiseSpec

    config = default_fixture_config()
    specs = (
        NoiseSpec(
            BALL_SPEED,
            scale=2.0,
            lower=90.0 if bounded else None,
            upper=110.0 if bounded else None,
            spec_id="input.ball-speed",
        ),
        NoiseSpec(LAUNCH_ANGLE, scale=4.0, spec_id="input.launch-angle"),
    )
    groups = (
        (
            PerturbationGroup(
                group_id="joint-inputs",
                spec_ids=("input.ball-speed", "input.launch-angle"),
                matrix=((1.0, 0.5), (0.5, 1.0)),
            ),
        )
        if correlated
        else ()
    )
    altered = ResponseFixtureConfig(
        deltas=config.deltas,
        coefficients=config.coefficients,
        baseline_positions_m=config.baseline_positions_m,
        baseline_valid=config.baseline_valid,
        perturbed_valid=config.perturbed_valid,
        specs=specs,
        groups=groups,
        input_kinds=(kind, "continuous"),
    )

    field = compute_position_noise_response_field(build_response_inputs(altered))

    np.testing.assert_array_equal(field.adequacy[0], expected)
    assert np.all(np.isnan(field.signed_response_m_per_declared_scale[0]))
    assert np.all(np.isfinite(field.all_eligible_absolute_rms_scatter_m[0]))


def test_chunked_and_resumed_accumulation_is_fingerprint_identical() -> None:
    inputs = build_response_inputs(default_fixture_config())
    serial = compute_position_noise_response_field(inputs)
    chunked = compute_position_noise_response_field(inputs, chunk_size=2)
    accumulator = ResponseFieldAccumulator(inputs)
    accumulator.accept_trial_slice(0, 2)
    resumed = ResponseFieldAccumulator.from_snapshot(inputs, accumulator.snapshot())
    resumed.accept_trial_slice(2, 4)
    restored = resumed.freeze()

    assert response_field_fingerprint(serial) == response_field_fingerprint(chunked)
    assert response_field_fingerprint(serial) == response_field_fingerprint(restored)
