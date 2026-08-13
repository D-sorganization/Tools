"""Execution binding between regional plans and frozen ground-result v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground import (
    GroundEventType,
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialRegion,
    GroundSimulationRequest,
    GroundSimulationResult,
    RegionalGroundExecutionFailureReason,
    RegionalGroundExecutionOptions,
    RegionalGroundExecutionResult,
    RegionalGroundExecutionStatus,
    RepeatedBounceResult,
    SkidRollSettings,
    execute_regional_ground,
    regional_ground_execution_result_from_json,
)

from ._support import _settled_prefix, _surface_run_request

FIXTURES = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plan(
    request: GroundSimulationRequest,
    *,
    lower_coordinate_m: float = 0.25,
    rolling_resistance: float = 0.1,
) -> GroundRegionalMaterialPlanRequest:
    surface = request.surface
    region_surface = replace(
        surface,
        surface_id="regional-rough",
        rolling_resistance=rolling_resistance,
    )
    return GroundRegionalMaterialPlanRequest(
        request_id="regional-execution-plan-001",
        base_surface=surface,
        axis_origin_m=(0.0, surface.height_m, 0.0),
        axis_unit=(1.0, 0.0, 0.0),
        lower_coordinate_m=-1.0,
        upper_coordinate_m=10.0,
        regions=(
            GroundRegionalMaterialRegion(
                "rough-band",
                10,
                lower_coordinate_m,
                lower_coordinate_m + 1.0,
                region_surface,
            ),
        ),
        provenance=GroundProvenance(
            "tools.rate_of_closure",
            "1.0.0",
            "regional-execution-test-plan",
            "a" * 64,
        ),
    )


def _execution(*, transition: bool = True) -> tuple[
    GroundSimulationRequest,
    RepeatedBounceResult,
    GroundRegionalMaterialPlanRequest,
]:
    request = _surface_run_request(max_time_s=0.4)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    plan = _plan(request, lower_coordinate_m=0.25 if transition else 5.0)
    return request, prefix, plan


def _fixture_execution() -> tuple[
    GroundSimulationRequest,
    RepeatedBounceResult,
    GroundRegionalMaterialPlanRequest,
]:
    request = _surface_run_request(max_time_s=0.15)
    prefix = _settled_prefix(
        request,
        velocity_m_s=(2.0, 0.0, 0.0),
        angular_velocity_rad_s=(0.0, 0.0, -2.0 / request.ball_radius_m),
    )
    return request, prefix, _plan(request, lower_coordinate_m=0.1)


def test_executor_recomputes_digests_and_preserves_plan_executor_provenance() -> None:
    request, prefix, plan = _execution()

    result = execute_regional_ground(request, prefix, plan)

    assert result.ground_request_sha256 == _digest(request.to_json())
    assert result.regional_plan_sha256 == _digest(plan.to_json())
    assert result.request_id == request.request_id
    assert result.surface_id == request.surface.surface_id
    assert result.plan_id == plan.request_id
    assert result.regional_plan == plan
    assert result.plan_provenance == plan.provenance
    assert result.executor_provenance.input_sha256 == result.execution_input_sha256
    assert result.ground_result is not None
    assert result.model_id == result.ground_result.model_id
    assert result.model_version == result.ground_result.model_version


def test_executor_rejects_base_surface_and_phase_identity_mismatch() -> None:
    request, prefix, plan = _execution()
    changed_surface = replace(request.surface, surface_id="different-base")
    mismatched_plan = replace(plan, base_surface=changed_surface)

    with pytest.raises(ValueError, match="base_surface"):
        execute_regional_ground(request, prefix, mismatched_plan)
    with pytest.raises(ValueError, match="identities"):
        execute_regional_ground(request, replace(prefix, request_id="wrong"), plan)


def test_execution_options_make_resolver_substitution_impossible() -> None:
    assert "resolver" not in RegionalGroundExecutionOptions.__dataclass_fields__
    with pytest.raises(TypeError, match="resolver"):
        RegionalGroundExecutionOptions(resolver=object())  # type: ignore[call-arg]


def test_transition_ledger_matches_embedded_ground_events_one_to_one() -> None:
    request, prefix, plan = _execution()
    result = execute_regional_ground(request, prefix, plan)
    ground = result.ground_result

    assert ground is not None
    events = tuple(
        event
        for event in ground.events
        if event.event_type is GroundEventType.SURFACE_TRANSITION
    )
    assert len(events) == len(result.transitions) == 1
    assert tuple(event.sequence for event in events) == tuple(
        transition.event_sequence for transition in result.transitions
    )
    assert result.transitions[0].from_region_id is None
    assert result.transitions[0].to_region_id == "rough-band"
    assert result.transitions[0].from_surface_id == request.surface.surface_id
    assert result.transitions[0].to_surface_id == plan.regions[0].surface.surface_id

    changed = result.to_dict()
    changed["transitions"][0]["event_sequence"] += 1
    with pytest.raises(ValueError, match="transition ledger"):
        RegionalGroundExecutionResult.from_dict(changed)

    fabricated = result.to_dict()
    fabricated["transitions"][0]["from_region_id"] = "rough-band"
    fabricated["transitions"][0]["to_region_id"] = None
    with pytest.raises(ValueError, match="regional plan"):
        RegionalGroundExecutionResult.from_dict(fabricated)

    reversed_order = result.to_dict()
    item = reversed_order["transitions"][0]
    item["from_region_id"], item["to_region_id"] = (
        item["to_region_id"],
        item["from_region_id"],
    )
    item["from_surface_id"], item["to_surface_id"] = (
        item["to_surface_id"],
        item["from_surface_id"],
    )
    with pytest.raises(ValueError, match="regional plan"):
        RegionalGroundExecutionResult.from_dict(reversed_order)


def test_empty_transition_run_retains_complete_plan_provenance() -> None:
    request, prefix, plan = _execution(transition=False)

    result = execute_regional_ground(request, prefix, plan)

    assert result.transitions == ()
    assert result.plan_id == plan.request_id
    assert result.plan_provenance == plan.provenance
    assert result.regional_plan_sha256 == _digest(plan.to_json())


def test_cancellation_and_internal_bound_return_typed_null_result_status() -> None:
    request, prefix, plan = _execution()
    cancelled = execute_regional_ground(
        request,
        prefix,
        plan,
        RegionalGroundExecutionOptions(is_cancelled=lambda: True),
    )
    failed = execute_regional_ground(
        request,
        prefix,
        plan,
        RegionalGroundExecutionOptions(settings=SkidRollSettings(max_steps=1)),
    )

    assert cancelled.status is RegionalGroundExecutionStatus.CANCELLED
    assert cancelled.failure_reason is RegionalGroundExecutionFailureReason.CANCELLED
    assert cancelled.ground_result is None
    assert failed.status is RegionalGroundExecutionStatus.FAILED
    assert failed.failure_reason is RegionalGroundExecutionFailureReason.STEP_LIMIT
    assert failed.ground_result is None


def test_untraversed_material_changes_plan_digest_not_physics() -> None:
    request, prefix, first_plan = _execution(transition=False)
    second_plan = _plan(request, lower_coordinate_m=5.0, rolling_resistance=0.2)

    first = execute_regional_ground(request, prefix, first_plan)
    second = execute_regional_ground(request, prefix, second_plan)

    assert first.regional_plan_sha256 != second.regional_plan_sha256
    assert first.ground_request_sha256 == second.ground_request_sha256
    assert first.ground_result is not None and second.ground_result is not None
    assert first.ground_result.to_json() == second.ground_result.to_json()


def test_wire_parser_rejects_extra_duplicate_and_malformed_evidence() -> None:
    request, prefix, plan = _execution()
    result = execute_regional_ground(request, prefix, plan)
    extra = {**result.to_dict(), "unexpected": True}
    malformed = {**result.to_dict(), "ground_request_sha256": "not-a-digest"}
    uppercase = {
        **result.to_dict(),
        "ground_request_sha256": result.ground_request_sha256.upper(),
    }

    with pytest.raises(ValueError, match="fields"):
        RegionalGroundExecutionResult.from_dict(extra)
    with pytest.raises(ValueError, match="ground_request_sha256"):
        RegionalGroundExecutionResult.from_dict(malformed)
    with pytest.raises(ValueError, match="ground_request_sha256"):
        RegionalGroundExecutionResult.from_dict(uppercase)
    with pytest.raises(ValueError, match="duplicate"):
        regional_ground_execution_result_from_json(
            '{"schema_version":"ground-regional-execution-result/v1",'
            '"schema_version":"ground-regional-execution-result/v1"}'
        )

    wrong_producer = result.to_dict()
    wrong_producer["executor_provenance"]["producer"] = "lookalike"
    with pytest.raises(ValueError, match="executor producer"):
        RegionalGroundExecutionResult.from_dict(wrong_producer)
    wrong_version = result.to_dict()
    wrong_version["executor_provenance"]["producer_version"] = "9.9.9"
    with pytest.raises(ValueError, match="executor version"):
        RegionalGroundExecutionResult.from_dict(wrong_version)
    revised_source = result.to_dict()
    revised_source["executor_provenance"]["source_revision"] = "verified-build-2"
    assert (
        RegionalGroundExecutionResult.from_dict(
            revised_source
        ).executor_provenance.source_revision
        == "verified-build-2"
    )


def test_shared_adversarial_transition_wire_parity() -> None:
    fixture = json.loads(
        (FIXTURES / "ground_regional_execution_adversarial_v1.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = execute_regional_ground(*_fixture_execution()).to_dict()

    for case in fixture["cases"]:
        changed = json.loads(json.dumps(baseline))
        changed["transitions"][0].update(case["overrides"])
        if case["accepted"]:
            parsed = RegionalGroundExecutionResult.from_dict(changed)
            assert parsed.transitions[0].event_sequence == 3
        else:
            with pytest.raises(ValueError, match=case["error"]):
                RegionalGroundExecutionResult.from_dict(changed)


def test_shared_golden_results_are_executor_produced_and_round_trip() -> None:
    fixture = json.loads(
        (FIXTURES / "ground_regional_execution_golden_v1.json").read_text(
            encoding="utf-8"
        )
    )
    request, prefix, plan = _fixture_execution()
    produced = {
        "representable": execute_regional_ground(request, prefix, plan),
        "cancelled": execute_regional_ground(
            request,
            prefix,
            plan,
            RegionalGroundExecutionOptions(is_cancelled=lambda: True),
        ),
        "failed": execute_regional_ground(
            request,
            prefix,
            plan,
            RegionalGroundExecutionOptions(settings=SkidRollSettings(max_steps=1)),
        ),
    }
    for name, expected in produced.items():
        result = RegionalGroundExecutionResult.from_dict(fixture[name]["result"])
        assert result.to_dict() == expected.to_dict() == fixture[name]["result"]
        assert _digest(result.to_json()) == fixture[name]["result_sha256"]

    invalid_cancelled = json.loads(json.dumps(fixture["cancelled"]["result"]))
    invalid_cancelled["failure_reason"] = "step_limit"
    with pytest.raises(ValueError, match="cancelled failure_reason"):
        RegionalGroundExecutionResult.from_dict(invalid_cancelled)
    invalid_failed = json.loads(json.dumps(fixture["failed"]["result"]))
    invalid_failed["failure_reason"] = "cancelled"
    with pytest.raises(ValueError, match="cancelled status"):
        RegionalGroundExecutionResult.from_dict(invalid_failed)

    fabricated_cancelled = json.loads(json.dumps(fixture["cancelled"]["result"]))
    fabricated_cancelled["transitions"] = fixture["representable"]["result"][
        "transitions"
    ]
    with pytest.raises(
        ValueError, match="null ground_result cannot declare transitions"
    ):
        RegionalGroundExecutionResult.from_dict(fabricated_cancelled)


def test_frozen_base_result_fixture_bytes_remain_unchanged() -> None:
    base_fixture = json.loads(
        (FIXTURES / "flight_to_ground_golden_v1.json").read_text(encoding="utf-8")
    )
    base = GroundSimulationResult.from_dict(base_fixture["result"])

    assert base.to_json() == canonical_numeric_json(base_fixture["result"])
