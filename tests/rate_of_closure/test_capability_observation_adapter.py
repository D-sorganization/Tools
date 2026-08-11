"""Capability observation to scalar-ensemble adapter tests."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import cast

import pytest

from rate_of_closure.variation.canonical_numeric_json import canonical_numeric_json
from rate_of_closure.variation.capability_observation_adapter import (
    CapabilityObservationEnsembleBuilder,
    build_capability_observation_ensemble,
    capability_observation_ensemble_json,
)
from rate_of_closure.variation.scalar_ensemble_contract import (
    ScalarEnsembleDataset,
    ScalarEnsembleRow,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight.capability_contract import TargetDefinition
from shared.python.swing_sim.flight.capability_observation import (
    CapabilitySampleMetric,
    CapabilitySampleObservation,
    CapabilitySampleParameter,
    CapabilitySampleStatus,
)
from shared.python.swing_sim.flight.inverse_contract import EvaluationStatus
from shared.python.swing_sim.flight.result_contract import (
    FlightMetricId,
    SignRule,
    flight_metric_catalog,
)


def _observation(
    attempt_ordinal: int, effective_status: CapabilitySampleStatus
) -> CapabilitySampleObservation:
    complete = effective_status is CapabilitySampleStatus.COMPLETE
    club_id = "iron-7" if attempt_ordinal == 0 else "driver"
    reason = None if complete else "missed_ball"
    metrics = (
        (
            CapabilitySampleMetric(
                FlightMetricId.CARRY_DISTANCE, 105.0, "fixture.carry"
            ),
            CapabilitySampleMetric(
                FlightMetricId.CARRY_OFFLINE, 5.0, "fixture.offline"
            ),
            CapabilitySampleMetric(FlightMetricId.APEX_HEIGHT, 31.0, "fixture.apex"),
        )
        if complete
        else ()
    )
    return CapabilitySampleObservation(
        problem_id="observer-fixture",
        attempt_ordinal=attempt_ordinal,
        attempted_count=attempt_ordinal + 1,
        total_count=2,
        candidate_ordinal=attempt_ordinal,
        club_candidate_ordinal=attempt_ordinal,
        sample_ordinal=0,
        club_id=club_id,
        parameters=(
            CapabilitySampleParameter(
                "ball_speed", "m/s", 50.0, 51.0 + attempt_ordinal
            ),
            CapabilitySampleParameter(
                "launch_angle", "deg", 12.0, 13.0 + attempt_ordinal
            ),
        ),
        source_status=(
            EvaluationStatus.COMPLETE
            if complete
            else EvaluationStatus(effective_status.value)
        ),
        effective_status=effective_status,
        reason_code=reason,
        source_reason=reason,
        metrics=metrics,
    )


TARGET = TargetDefinition("green", 100.0, 2.0, 10.0, 15.0, 16.0)
ASCII_DIGEST = "df36f765afdf508d00a3d264911ce5b6f07e25da3744b187596d67487ea3be5f"  # noqa: E501  # pragma: allowlist secret
UNICODE_DIGEST = "18086b5e97d576598bbfa63407b6eda786a3a7ce20509654de282400bd32efd0"  # noqa: E501  # pragma: allowlist secret


def _build(
    observations: tuple[CapabilitySampleObservation, ...], max_rows: int = 4
) -> ScalarEnsembleDataset:
    return build_capability_observation_ensemble(
        observations, TARGET, max_rows, "fixture/evaluator-v1"
    )


def _assert_schema(dataset: ScalarEnsembleDataset) -> None:
    assert [row.row_id for row in dataset.rows] == [
        "series:candidate%3A0%2Fclub%3Airon-7/trial:0",
        "series:candidate%3A1%2Fclub%3Adriver/trial:0",
    ]
    variable_keys = [variable.key for variable in dataset.variables]
    assert variable_keys[:4] == [
        "nominal.ball_speed",
        "perturbed.ball_speed",
        "nominal.launch_angle",
        "perturbed.launch_angle",
    ]
    assert [key for key in variable_keys if key.startswith("metric.")] == [
        f"metric.{definition.metric_id.value}"
        for definition in flight_metric_catalog().definitions
        if definition.sign_rule is not SignRule.VECTOR_COMPONENTS
    ]
    assert variable_keys[-6:] == [
        "target_downrange_residual",
        "target_lateral_residual",
        "target_residual",
        "target_signed_distance",
        "target_solver_residual",
        "target_contains",
    ]


def _assert_complete_row(complete: ScalarEnsembleRow) -> None:
    assert complete.values["nominal.ball_speed"] == 50.0
    assert complete.values["perturbed.ball_speed"] == 51.0
    assert complete.values["metric.carry_distance"] == 105.0
    assert complete.values["target_downrange_residual"] == 5.0
    assert complete.values["target_lateral_residual"] == 3.0
    assert complete.values["target_residual"] == pytest.approx(math.sqrt(34.0))
    assert complete.values["target_signed_distance"] == pytest.approx(
        math.sqrt(34.0) - 10.0
    )
    assert complete.values["target_solver_residual"] == pytest.approx(
        0.05 * math.sqrt(34.0)
    )
    assert complete.values["target_contains"] == 1.0
    assert complete.values["metric.total_distance"] is None
    assert complete.attributes is not None
    assert complete.attributes["metric.carry_distance.provenance"] == "fixture.carry"


def _assert_no_impact_row(no_impact: ScalarEnsembleRow) -> None:
    assert no_impact.cohort == "no_impact"
    assert all(
        value is None
        for key, value in no_impact.values.items()
        if key.startswith(("metric.", "target_"))
    )
    assert no_impact.attributes is not None
    assert no_impact.attributes["metric.carry_distance.provenance"] is None


def test_declares_all_values_and_emits_deterministic_nullable_rows() -> None:
    dataset = _build(
        (
            _observation(1, CapabilitySampleStatus.NO_IMPACT),
            _observation(0, CapabilitySampleStatus.COMPLETE),
        )
    )
    _assert_schema(dataset)
    complete, no_impact = dataset.rows
    _assert_complete_row(complete)
    _assert_no_impact_row(no_impact)


def test_rejects_overflow_instead_of_truncation_and_bounds_max_rows() -> None:
    observations = (
        _observation(0, CapabilitySampleStatus.COMPLETE),
        _observation(1, CapabilitySampleStatus.FAILED),
    )
    with pytest.raises(ContractViolationError, match="exceeds max_rows"):
        _build(observations, 1)
    for max_rows in (0, -1, 100_001, 1.5, True):
        with pytest.raises(ContractViolationError, match="max_rows"):
            _build(observations[:1], max_rows)  # type: ignore[arg-type]
    builder = CapabilityObservationEnsembleBuilder(TARGET, 1, "fixture/evaluator-v1")
    builder.accept(observations[0])
    with pytest.raises(ContractViolationError, match="max_rows"):
        builder.accept(observations[1])
    assert builder.retained_count == 1
    assert len(builder.build().rows) == 1


def test_builder_rejects_non_contract_observation_without_retaining_it() -> None:
    builder = CapabilityObservationEnsembleBuilder(TARGET, 2, "fixture/evaluator-v1")
    with pytest.raises(ContractViolationError, match="CapabilitySampleObservation"):
        builder.accept(cast(CapabilitySampleObservation, object()))
    assert builder.retained_count == 0


def test_rejects_ambiguous_identity_and_parameter_declaration_drift() -> None:
    first = _observation(0, CapabilitySampleStatus.COMPLETE)
    duplicate = _observation(0, CapabilitySampleStatus.COMPLETE)
    with pytest.raises(ContractViolationError, match="attempt_ordinal"):
        _build((first, duplicate))
    prefix = replace(first, total_count=3)
    gap = replace(
        _observation(1, CapabilitySampleStatus.COMPLETE),
        attempt_ordinal=2,
        attempted_count=3,
        total_count=3,
    )
    with pytest.raises(ContractViolationError, match="contiguous prefix"):
        _build((prefix, gap))
    drift = CapabilitySampleObservation(
        **{
            **_observation(1, CapabilitySampleStatus.COMPLETE).__dict__,
            "parameters": (
                CapabilitySampleParameter("launch_angle", "deg", 12.0, 13.0),
                CapabilitySampleParameter("ball_speed", "mph", 50.0, 51.0),
            ),
        }
    )
    with pytest.raises(ContractViolationError, match="parameter declarations"):
        _build((first, drift))


def test_stable_wire_matches_typescript_digest() -> None:
    dataset = _build(
        (
            _observation(1, CapabilitySampleStatus.NO_IMPACT),
            _observation(0, CapabilitySampleStatus.COMPLETE),
        )
    )
    digest = hashlib.sha256(
        capability_observation_ensemble_json(dataset).encode()
    ).hexdigest()
    assert digest == ASCII_DIGEST


def test_stable_wire_matches_typescript_unicode_digest() -> None:
    base = _observation(0, CapabilitySampleStatus.COMPLETE)
    unicode_observation = replace(
        base,
        problem_id="観測-ß",
        club_id="ドライバー",
        total_count=1,
        parameters=(
            CapabilitySampleParameter("zeta", "m/s", 1.0, 2.0),
            CapabilitySampleParameter("Ω", "deg", 3.0, 4.0),
        ),
        metrics=(
            replace(base.metrics[0], provenance="測定.é"),
            *base.metrics[1:],
        ),
    )
    payload = capability_observation_ensemble_json(_build((unicode_observation,)))
    assert "観測-ß" in payload
    assert hashlib.sha256(payload.encode()).hexdigest() == UNICODE_DIGEST


def test_stable_wire_uses_canonical_numeric_tokens_for_every_float() -> None:
    base = _observation(0, CapabilitySampleStatus.COMPLETE)
    adversarial = replace(
        base,
        total_count=1,
        parameters=(
            CapabilitySampleParameter("positive_half", "1", 1.234567890125, 1e-12),
            CapabilitySampleParameter("negative_half", "1", -1.234567890125, -0.0),
            CapabilitySampleParameter("threshold", "1", 1e-11, 9_007_199_254_740_991.0),
            CapabilitySampleParameter(
                "half_away", "1", 1.000000000005, -1.000000000005
            ),
        ),
    )

    payload = capability_observation_ensemble_json(_build((adversarial,)))
    wire = json.loads(payload)
    row_values = wire["rows"][0]["values"]

    assert row_values["nominal.positive_half"] == 1.23456789012
    assert row_values["perturbed.positive_half"] == 0
    assert row_values["nominal.negative_half"] == -1.23456789012
    assert row_values["perturbed.negative_half"] == 0
    assert row_values["nominal.threshold"] == 0.00000000001
    assert row_values["perturbed.threshold"] == 9_007_199_254_740_991
    assert row_values["nominal.half_away"] == 1.00000000001
    assert row_values["perturbed.half_away"] == -1.00000000001
    assert all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in row_values.values()
        if value is not None
    )
    assert '"nominal.threshold":0.00000000001' in payload
    assert '"perturbed.threshold":9007199254740991' in payload


def test_parameter_labels_only_uppercase_initial_ascii_letters() -> None:
    base = _observation(0, CapabilitySampleStatus.COMPLETE)
    adversarial = replace(
        base,
        total_count=1,
        parameters=(
            CapabilitySampleParameter("ball_speed", "m/s", 1.0, 2.0),
            CapabilitySampleParameter("ß_value", "1", 3.0, 4.0),
            CapabilitySampleParameter("éCLAIR_rate", "1/s", 5.0, 6.0),
            CapabilitySampleParameter("ALPHA_value", "1", 7.0, 8.0),
        ),
    )

    dataset = _build((adversarial,))
    labels = {
        variable.key: variable.label
        for variable in dataset.variables
        if variable.key.startswith("nominal.")
    }

    assert labels == {
        "nominal.ball_speed": "Nominal Ball Speed",
        "nominal.ß_value": "Nominal ß Value",
        "nominal.éCLAIR_rate": "Nominal éCLAIR Rate",
        "nominal.ALPHA_value": "Nominal ALPHA Value",
    }


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_numeric_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite floats"):
        canonical_numeric_json({"value": value})
