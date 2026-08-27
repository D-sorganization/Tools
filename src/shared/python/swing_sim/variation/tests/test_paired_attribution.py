"""Falsification tests for paired localized source-to-downstream attribution."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.variation.paired_attribution import (
    AVAILABILITY_AVAILABLE,
    AVAILABILITY_MISSING,
    AVAILABILITY_NO_IMPACT,
    AVAILABILITY_NONFINITE,
    AVAILABILITY_NUMERICAL_FAILURE,
    AVAILABILITY_UNSUPPORTED,
    INTERPRETATION_BOUNDARY,
    MAX_PAIRS,
    PAIRED_INTERVENTION_METHOD_ID,
    AttributionPair,
    AttributionRunContext,
    AttributionSource,
    AttributionTarget,
    PairedAttributionAccumulator,
    PairedAttributionInput,
    attribution_csv,
    attribution_record_fingerprint,
    attribution_rows,
    compute_paired_attribution,
    snapshot_from_json,
    snapshot_to_json,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_FRAME = "app_frame:x_target,y_up,z_right"


def _context(**changes: object) -> AttributionRunContext:
    values: dict[str, object] = {
        "model_id": "double-pendulum/v1",
        "adapter_id": "rate-double-pendulum/v1",
        "coordinate_frame": _FRAME,
        "trace_grid_sha256": _SHA_A,
        "plan_sha256": _SHA_B,
        "registry_sha256": _SHA_C,
        "execution_sha256": _SHA_D,
    }
    values.update(changes)
    return AttributionRunContext(**values)  # type: ignore[arg-type]


def _source() -> AttributionSource:
    return AttributionSource(
        source_id="wrist-window",
        variable_key="swing.wrist_torque_offset_nm",
        unit="N*m",
        point_id="joint.wrist",
        time_window_s=(0.2, 0.4),
    )


def _targets() -> tuple[AttributionTarget, ...]:
    return (
        AttributionTarget(
            target_id="clubhead-x-at-0.30",
            kind="state",
            unit="m",
            coordinate_frame=_FRAME,
            point_id="swing.clubhead",
            coordinate_value=0.3,
            coordinate_unit="s",
        ),
        AttributionTarget(
            target_id="clubhead-x-at-0.60",
            kind="state",
            unit="m",
            coordinate_frame=_FRAME,
            point_id="swing.clubhead",
            coordinate_value=0.6,
            coordinate_unit="s",
        ),
        AttributionTarget(target_id="impact-speed", kind="impact", unit="m/s"),
        AttributionTarget(target_id="carry", kind="shot", unit="m"),
    )


def _pair(
    pair_id: str,
    baseline_source: float,
    perturbed_source: float,
    baseline_values: tuple[float, ...],
    perturbed_values: tuple[float, ...],
    *,
    baseline_status: str = "evaluated_hit",
    perturbed_status: str = "evaluated_hit",
    baseline_states: tuple[str, ...] | None = None,
    perturbed_states: tuple[str, ...] | None = None,
) -> AttributionPair:
    count = len(baseline_values)
    available = (AVAILABILITY_AVAILABLE,) * count
    return AttributionPair(
        pair_id=pair_id,
        baseline_trial_id=f"{pair_id}-base",
        perturbed_trial_id=f"{pair_id}-perturbed",
        baseline_status=baseline_status,
        perturbed_status=perturbed_status,
        baseline_source_value=baseline_source,
        perturbed_source_value=perturbed_source,
        baseline_values=np.asarray(baseline_values, dtype=float),
        perturbed_values=np.asarray(perturbed_values, dtype=float),
        baseline_value_states=baseline_states or available,
        perturbed_value_states=perturbed_states or available,
    )


def _field(*pairs: AttributionPair) -> PairedAttributionInput:
    context = _context()
    return PairedAttributionInput(
        source=_source(),
        targets=_targets(),
        pairs=tuple(pairs),
        baseline_context=context,
        perturbed_context=context,
        source_sha256="e" * 64,
    )


def test_affine_fixture_recovers_signed_response_and_sign_reversal() -> None:
    positive = _pair(
        "positive",
        0.0,
        2.0,
        (1.0, 7.0, 10.0, 100.0),
        (7.0, 7.0, 14.0, 110.0),
    )
    negative = _pair(
        "negative",
        2.0,
        0.0,
        (7.0, 7.0, 14.0, 110.0),
        (1.0, 7.0, 10.0, 100.0),
    )

    record = compute_paired_attribution(_field(positive, negative))

    np.testing.assert_allclose(
        record.signed_response, ((6, 0, 4, 10), (-6, 0, -4, -10))
    )
    np.testing.assert_allclose(
        record.response_magnitude, np.abs(record.signed_response)
    )
    np.testing.assert_allclose(
        record.local_response_per_source_unit, ((3, 0, 2, 5), (3, 0, 2, 5))
    )
    assert record.method_id == PAIRED_INTERVENTION_METHOD_ID
    assert record.interpretation_boundary == INTERPRETATION_BOUNDARY
    assert "not" in record.interpretation_boundary.lower()


def test_nonlinear_fixture_exposes_baseline_and_locus_dependence() -> None:
    low = _pair("low", 0.0, 1.0, (0.0, 4.0, 0.0, 0.0), (1.0, 4.0, 0.0, 0.0))
    high = _pair("high", 2.0, 3.0, (4.0, 4.0, 0.0, 0.0), (9.0, 4.0, 0.0, 0.0))

    record = compute_paired_attribution(_field(low, high))

    assert record.local_response_per_source_unit[:, 0].tolist() == [1.0, 5.0]
    assert record.local_response_per_source_unit[:, 1].tolist() == [0.0, 0.0]
    assert "global" in record.interpretation_boundary.lower()


def test_typed_availability_never_fabricates_impact_or_shot_values() -> None:
    missing = np.nan
    no_impact = _pair(
        "no-impact",
        0.0,
        1.0,
        (1.0, 2.0, missing, missing),
        (2.0, 3.0, missing, missing),
        perturbed_status="evaluated_no_impact",
        baseline_states=(AVAILABILITY_AVAILABLE,) * 2 + (AVAILABILITY_MISSING,) * 2,
        perturbed_states=(AVAILABILITY_AVAILABLE,) * 2 + (AVAILABILITY_MISSING,) * 2,
    )
    failed = _pair(
        "failed",
        0.0,
        1.0,
        (1.0, 2.0, 3.0, 4.0),
        (missing,) * 4,
        perturbed_status="numerical_failure",
        perturbed_states=(AVAILABILITY_MISSING,) * 4,
    )
    unsupported = _pair(
        "unsupported",
        0.0,
        1.0,
        (missing, 2.0, 3.0, 4.0),
        (missing, 3.0, 4.0, 5.0),
        baseline_states=(AVAILABILITY_UNSUPPORTED,) + (AVAILABILITY_AVAILABLE,) * 3,
        perturbed_states=(AVAILABILITY_UNSUPPORTED,) + (AVAILABILITY_AVAILABLE,) * 3,
    )

    record = compute_paired_attribution(_field(no_impact, failed, unsupported))

    assert record.availability[0].tolist() == [
        AVAILABILITY_AVAILABLE,
        AVAILABILITY_AVAILABLE,
        AVAILABILITY_NO_IMPACT,
        AVAILABILITY_NO_IMPACT,
    ]
    assert record.availability[1].tolist() == [AVAILABILITY_NUMERICAL_FAILURE] * 4
    assert record.availability[2, 0] == AVAILABILITY_UNSUPPORTED
    assert np.isnan(record.signed_response[0, 2:]).all()
    assert np.isnan(record.signed_response[1]).all()
    assert record.available_count.tolist() == [1, 2, 1, 1]
    assert record.no_impact_count.tolist() == [0, 0, 1, 1]
    assert record.numerical_failure_count.tolist() == [1, 1, 1, 1]
    assert record.unsupported_count.tolist() == [1, 0, 0, 0]


@pytest.mark.parametrize(
    "changed",
    [
        {"model_id": "spatial/v2"},
        {"adapter_id": "other/v1"},
        {"coordinate_frame": "other-frame"},
        {"trace_grid_sha256": "f" * 64},
        {"plan_sha256": "f" * 64},
        {"registry_sha256": "f" * 64},
        {"execution_sha256": "f" * 64},
        {"source_adapter_id": "other-source-adapter/v1"},
    ],
)
def test_context_drift_fails_closed(changed: dict[str, object]) -> None:
    pair = _pair("pair", 0.0, 1.0, (0.0,) * 4, (1.0,) * 4)
    context = _context()

    with pytest.raises(ValueError, match="context mismatch"):
        PairedAttributionInput(
            _source(),
            _targets(),
            (pair,),
            context,
            replace(context, **changed),
            "e" * 64,
        )


def test_zero_delta_and_pair_identity_mismatch_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _pair("zero", 1.0, 1.0, (0.0,) * 4, (1.0,) * 4)

    pair = _pair("pair", 0.0, 1.0, (0.0,) * 4, (1.0,) * 4)
    with pytest.raises(ValueError, match="trial IDs"):
        replace(pair, perturbed_trial_id=pair.baseline_trial_id)


def test_chunk_resume_and_archive_replay_are_byte_equivalent() -> None:
    pairs = tuple(
        _pair(
            f"pair-{index}",
            float(index),
            float(index + 1),
            (float(index), 0.0, 1.0, 2.0),
            (float(index + 2), 0.0, 2.0, 4.0),
        )
        for index in range(6)
    )
    full_input = _field(*pairs)
    serial = compute_paired_attribution(full_input)
    accumulator = PairedAttributionAccumulator(full_input.contract_without_pairs())
    accumulator.accept(full_input.with_pairs(pairs[:2]))
    snapshot = snapshot_from_json(snapshot_to_json(accumulator.snapshot()))
    resumed = PairedAttributionAccumulator(
        full_input.contract_without_pairs(), snapshot
    )
    resumed.accept(full_input.with_pairs(pairs[2:]))
    chunked = resumed.finalize()

    assert attribution_record_fingerprint(serial) == attribution_record_fingerprint(
        chunked
    )
    assert attribution_rows(serial) == attribution_rows(chunked)
    assert snapshot.accepted_pairs == 2


def test_rows_csv_and_selector_preserve_pair_identity_and_precision() -> None:
    precise = 0.12345678901234566
    pair = _pair("precision", 0.0, 1.0, (0.0,) * 4, (precise,) * 4)
    record = compute_paired_attribution(_field(pair))

    rows = attribution_rows(record, target_id="carry")
    csv_text = attribution_csv(record, target_id="carry")

    assert len(rows) == 1
    assert rows[0]["pair_id"] == "precision"
    assert rows[0]["target_id"] == "carry"
    assert rows[0]["target_metric_id"] == "carry"
    assert rows[0]["signed_response"] == precise
    assert format(precise, ".17g") in csv_text


def test_observational_association_cannot_masquerade_as_intervention_response() -> None:
    pairs = tuple(
        _pair(
            f"confounded-{index}",
            float(index),
            float(index + 1),
            (10.0 * index, 0.0, 0.0, 0.0),
            (10.0 * index, 0.0, 0.0, 0.0),
        )
        for index in range(4)
    )

    record = compute_paired_attribution(_field(*pairs))

    assert record.signed_response[:, 0].tolist() == [0.0] * 4
    assert "rank" in record.interpretation_boundary.lower()


def test_nonfinite_cells_records_and_resource_caps_fail_closed() -> None:
    nonfinite = _pair(
        "nonfinite",
        0.0,
        1.0,
        (np.nan, 0.0, 0.0, 0.0),
        (np.nan, 1.0, 1.0, 1.0),
        baseline_states=(AVAILABILITY_NONFINITE,) + (AVAILABILITY_AVAILABLE,) * 3,
        perturbed_states=(AVAILABILITY_NONFINITE,) + (AVAILABILITY_AVAILABLE,) * 3,
    )
    record = compute_paired_attribution(_field(nonfinite))

    assert record.availability[0, 0] == AVAILABILITY_NONFINITE
    assert np.isnan(record.signed_response[0, 0])
    assert record.baseline_values.flags.writeable is False
    with pytest.raises(ValueError, match="resource cap"):
        _field(*(nonfinite for _ in range(MAX_PAIRS + 1)))
