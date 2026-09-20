"""Contract tests for launch-monitor convention provenance and comparison."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shared.python.swing_sim.conventions import (
    ComparabilityReason,
    ConventionId,
    ConventionRegistry,
    EventTime,
    ParameterDefinition,
    ParameterId,
    ReferencePoint,
    SignRule,
    compare_definitions,
    convention_registry,
    shift_point_velocity,
    transform_vector,
)


def test_catalog_is_complete_for_the_foundation_parameters() -> None:
    registry = convention_registry()

    assert registry.schema_version == "launch-monitor-conventions/v1"
    for convention in ConventionId:
        assert {
            definition.parameter_id
            for definition in registry.for_convention(convention)
        } == set(ParameterId)


def test_catalog_matches_the_cross_client_golden_cases() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[6]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "launch_monitor_registry_golden_v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    registry = convention_registry()

    assert fixture["schema_version"] == registry.schema_version
    assert fixture["definition_count"] == len(registry.definitions)
    assert (
        hashlib.sha256(registry.to_json().encode()).hexdigest()
        == fixture["canonical_json_sha256"]
    )
    for case in fixture["cases"]:
        definition = registry.definition(
            ConventionId(case["convention_id"]), ParameterId(case["parameter_id"])
        )
        for field in ("reference_point", "event_time", "quantity_status"):
            assert getattr(definition, field).value == case[field]


def test_contracts_are_immutable_and_strictly_validated() -> None:
    definition = convention_registry().definition(
        ConventionId.TRACKMAN_COMPARABLE, ParameterId.CLUB_SPEED
    )

    assert dataclasses.is_dataclass(definition)
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.unit = "mph"  # type: ignore[misc]
    with pytest.raises(ValueError, match="source_url"):
        ParameterDefinition(
            **{
                **dataclasses.asdict(definition),
                "source_url": "not-a-url",
            }
        )


def test_json_is_deterministic_and_v0_migrates_without_guessing_semantics() -> None:
    registry = convention_registry()
    first = registry.to_json()

    assert first == registry.to_json()
    assert ConventionRegistry.from_json(json.loads(first)).to_json() == first
    legacy = json.loads(first)
    legacy["schema_version"] = "launch-monitor-conventions/v0"
    legacy["definitions"][0]["vendor"] = legacy["definitions"][0].pop("convention_id")
    migrated = ConventionRegistry.from_json(legacy)

    assert migrated.schema_version == "launch-monitor-conventions/v1"
    assert migrated.definitions[0] == registry.definitions[0]


def test_comparison_reports_reference_point_and_event_time_mismatch() -> None:
    registry = convention_registry()
    trackman = registry.definition(
        ConventionId.TRACKMAN_COMPARABLE, ParameterId.CLUB_PATH
    )
    foresight = registry.definition(
        ConventionId.FORESIGHT_COMPARABLE, ParameterId.CLUB_PATH
    )

    result = compare_definitions(trackman, foresight)

    assert not result.comparable
    assert result.reasons == (
        ComparabilityReason.REFERENCE_POINT,
        ComparabilityReason.EVENT_TIME,
    )


def test_comparison_reports_geometry_contract_mismatch() -> None:
    definition = convention_registry().definition(
        ConventionId.APP_NATIVE, ParameterId.SPIN_LOFT
    )

    result = compare_definitions(
        definition,
        dataclasses.replace(definition, geometry_contract="planar_loft_difference"),
    )

    assert result.reasons == (ComparabilityReason.GEOMETRY,)


def test_comparison_reports_sign_rule_mismatch() -> None:
    definition = convention_registry().definition(
        ConventionId.APP_NATIVE, ParameterId.LAUNCH_DIRECTION
    )

    result = compare_definitions(
        definition,
        dataclasses.replace(definition, sign_rule=SignRule.UNSPECIFIED),
    )

    assert result.reasons == (ComparabilityReason.SIGN_RULE,)


def test_foresight_launch_direction_does_not_invent_an_absolute_sign() -> None:
    definition = convention_registry().definition(
        ConventionId.FORESIGHT_COMPARABLE, ParameterId.LAUNCH_DIRECTION
    )

    assert definition.sign_rule is SignRule.UNSPECIFIED


def test_point_shift_uses_exact_rigid_body_velocity_identity() -> None:
    shifted = shift_point_velocity(
        reference_velocity=(50.0, 1.0, 0.0),
        angular_velocity=(0.0, 20.0, 0.0),
        point_offset=(0.04, 0.0, 0.02),
    )

    assert shifted == pytest.approx((50.4, 1.0, -0.8))
    with pytest.raises(ValueError, match="finite"):
        shift_point_velocity((0.0, 0.0, 0.0), (0.0, np.nan, 0.0), (0.0, 0.0, 0.0))


def test_frame_transform_requires_a_proper_orthonormal_rotation() -> None:
    rotation = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))

    assert transform_vector((1.0, 2.0, 3.0), rotation) == pytest.approx(
        (3.0, 2.0, -1.0)
    )
    with pytest.raises(ValueError, match="proper orthonormal"):
        transform_vector((1.0, 0.0, 0.0), ((2.0, 0.0, 0.0),) * 3)


def test_trackman_reference_and_time_policies_are_source_explicit() -> None:
    registry = convention_registry()
    speed = registry.definition(
        ConventionId.TRACKMAN_COMPARABLE, ParameterId.CLUB_SPEED
    )
    face = registry.definition(ConventionId.TRACKMAN_COMPARABLE, ParameterId.FACE_ANGLE)

    assert speed.reference_point is ReferencePoint.GEOMETRIC_CENTER
    assert speed.event_time is EventTime.JUST_BEFORE_FIRST_CONTACT
    assert face.reference_point is ReferencePoint.IMPACT_LOCATION
    assert face.event_time is EventTime.MAXIMUM_COMPRESSION
