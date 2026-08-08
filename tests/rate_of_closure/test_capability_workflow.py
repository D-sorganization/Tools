"""Versioned capability-workflow authoring and persistence contracts."""

from __future__ import annotations

import json

import pytest

from rate_of_closure.application.capability_workflow import (
    CAPABILITY_WORKFLOW_SCHEMA_VERSION,
    CapabilityWorkflowInputs,
    build_capability_workflow,
    capability_workflow_from_json,
    capability_workflow_json,
)

pytestmark = pytest.mark.unit


def test_default_driver_workflow_is_model_ready_and_auditable() -> None:
    document = build_capability_workflow(CapabilityWorkflowInputs())

    assert document.schema_version == CAPABILITY_WORKFLOW_SCHEMA_VERSION
    assert document.request.club_ids == ("driver",)
    assert document.profile.club("driver").parameters[0].parameter_id == "ball_speed"
    assert document.request.target.distance_m == pytest.approx(230.0)
    assert document.evaluator_config.spin_defaults[0].club_id == "driver"
    assert "user-authored" in document.evaluator_config.spin_defaults[0].provenance


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
