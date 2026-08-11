"""Versioned capability-workflow authoring and persistence contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.application.capability_workflow import (
    CAPABILITY_WORKFLOW_SCHEMA_VERSION,
    CapabilityWorkflowInputs,
    build_capability_workflow,
    capability_workflow_from_json,
    capability_workflow_json,
)
from rate_of_closure.variation.scalar_ensemble_contract import (
    SCALAR_ENSEMBLE_SCHEMA_VERSION,
    ScalarCohortDefinition,
    ScalarEnsembleDataset,
    ScalarEnsembleProvenance,
    ScalarEnsembleRow,
    ScalarEnsembleStage,
    ScalarVariableCategory,
    ScalarVariableDefinition,
    scalar_ensemble_row_id,
)
from rate_of_closure.variation.scalar_ensemble_io import non_complete_reason_summary

pytestmark = pytest.mark.unit

_PARSER_FIXTURE = json.loads(
    (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "capability_workflow_parser_cases_v1.json"
    ).read_text(encoding="utf-8")
)
_PARSER_CASES = _PARSER_FIXTURE["cases"]
_HOSTILE_NUMBERS = _PARSER_FIXTURE["hostile_numbers"]


def _mutated_workflow(case: dict[str, object]) -> str:
    payload = json.loads(
        capability_workflow_json(build_capability_workflow(CapabilityWorkflowInputs()))
    )
    path = case["path"]
    assert isinstance(path, list) and path
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = case["value"]
    return json.dumps(payload)


def test_default_driver_workflow_is_model_ready_and_auditable() -> None:
    document = build_capability_workflow(CapabilityWorkflowInputs())

    assert document.schema_version == CAPABILITY_WORKFLOW_SCHEMA_VERSION
    assert document.request.club_ids == ("driver",)
    assert document.profile.club("driver").parameters[0].parameter_id == "ball_speed"
    assert document.request.target.distance_m == pytest.approx(230.0)
    assert document.evaluator_config.spin_defaults[0].club_id == "driver"
    assert "user-authored" in document.evaluator_config.spin_defaults[0].provenance


def test_shared_parser_fixture_schema_is_supported() -> None:
    assert _PARSER_FIXTURE["schema_version"] == "capability-workflow-parser-cases/v1"


@pytest.mark.parametrize("case", _HOSTILE_NUMBERS, ids=lambda case: case["id"])
def test_shared_parser_rejects_oversized_raw_json_numbers(
    case: dict[str, object],
) -> None:
    source = capability_workflow_json(
        build_capability_workflow(CapabilityWorkflowInputs())
    )
    raw_number = str(case["digit"]) * int(case["digits"])
    source = source.replace('"candidate_budget":8', f'"candidate_budget":{raw_number}')

    with pytest.raises(ValueError, match="magnitude|finite"):
        capability_workflow_from_json(source)


def test_workflow_round_trip_preserves_strict_nested_contracts() -> None:
    source = build_capability_workflow(
        CapabilityWorkflowInputs(
            club_id="driver-fit-a",
            target_distance_m=245.0,
            target_lateral_m=-4.0,
            total_spin_rpm=2250.0,
            spin_axis_tilt_deg=-3.5,
            candidate_budget=4,
            ensemble_size=6,
        )
    )

    encoded = capability_workflow_json(source)
    restored = capability_workflow_from_json(encoded)

    assert restored == source
    assert json.loads(encoded)["schema_version"] == CAPABILITY_WORKFLOW_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ball_speed_mps": 0.0}, "ball_speed_mps"),
        ({"candidate_budget": 501, "ensemble_size": 201}, "100000"),
        ({"alternatives_count": 3, "candidate_budget": 2}, "alternatives_count"),
        ({"spin_axis_tilt_deg": 91.0}, "spin_axis_tilt_deg"),
        ({"total_spin_rpm": 20_001.0}, "total_spin_rpm"),
        ({"max_time_s": 121.0}, "max_time_s"),
        ({"seed": 2**31}, "seed"),
    ],
)
def test_workflow_rejects_unsafe_or_unrenderable_inputs(
    changes: dict[str, object], message: str
) -> None:
    values = CapabilityWorkflowInputs().__dict__ | changes
    with pytest.raises(ValueError, match=message):
        build_capability_workflow(CapabilityWorkflowInputs(**values))


def test_workflow_json_rejects_extra_fields() -> None:
    payload = json.loads(
        capability_workflow_json(build_capability_workflow(CapabilityWorkflowInputs()))
    )
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="fields"):
        capability_workflow_from_json(json.dumps(payload))


def test_workflow_json_rejects_spin_default_for_a_different_club() -> None:
    payload = json.loads(
        capability_workflow_json(build_capability_workflow(CapabilityWorkflowInputs()))
    )
    payload["evaluator_config"]["spin_defaults"][0]["club_id"] = "other-club"

    with pytest.raises(ValueError, match="spin default club_ids"):
        capability_workflow_from_json(json.dumps(payload))


@pytest.mark.parametrize("case", _PARSER_CASES, ids=lambda case: case["id"])
def test_shared_workflow_parser_cases_have_strict_types(
    case: dict[str, object],
) -> None:
    source = _mutated_workflow(case)

    if case["accepted"]:
        capability_workflow_from_json(source)
    else:
        with pytest.raises((TypeError, ValueError)):
            capability_workflow_from_json(source)


def _reason_dataset(
    rows: tuple[tuple[str, str | None], ...],
) -> ScalarEnsembleDataset:
    return ScalarEnsembleDataset(
        SCALAR_ENSEMBLE_SCHEMA_VERSION,
        "reason-example",
        ScalarEnsembleProvenance("reason-test/v1", "source/v1", "fixture"),
        (ScalarEnsembleStage("input", "Inputs"),),
        (ScalarVariableCategory("launch", "Launch"),),
        (ScalarVariableDefinition("speed", "Speed", "m/s", "input", "launch"),),
        (
            ScalarCohortDefinition("complete", "Complete"),
            ScalarCohortDefinition("failed", "Failed"),
        ),
        tuple(
            ScalarEnsembleRow(
                scalar_ensemble_row_id(index, "run"),
                index,
                cohort,
                {"speed": 1.0},
                "run",
                {"reason_code": reason},
            )
            for index, (cohort, reason) in enumerate(rows)
        ),
    )


def test_reason_summary_names_a_horizon_timeout_instead_of_implying_breakage() -> None:
    dataset = _reason_dataset(
        (
            ("complete", None),
            ("failed", "no_ground_crossing_before_max_time"),
            ("failed", "no_ground_crossing_before_max_time"),
        )
    )

    assert non_complete_reason_summary(dataset) == (
        " Non-complete reasons: no_ground_crossing_before_max_time x2."
    )


def test_reason_summary_orders_by_count_then_reason_and_labels_missing() -> None:
    dataset = _reason_dataset(
        (("failed", None), ("failed", "overflow"), ("failed", "overflow"))
    )

    assert non_complete_reason_summary(dataset) == (
        " Non-complete reasons: overflow x2; unspecified x1."
    )


def test_reason_summary_is_empty_when_every_retained_row_completed() -> None:
    assert non_complete_reason_summary(_reason_dataset((("complete", None),))) == ""
