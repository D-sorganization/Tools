"""Whole-workspace document contract tests for Tools #4220."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from rate_of_closure.application.workspace_document import (
    WORKSPACE_SCHEMA,
    WORKSPACE_SCHEMA_VERSION,
    VersionedPayload,
    WorkspaceDocument,
    WorkspaceLayout,
    WorkspaceMetadata,
    workspace_from_json,
    workspace_to_json,
)
from shared.python.swing_sim.torque_profiles import (
    JointTorqueAssignment,
    PrescribedTorqueProfile,
    TorquePolynomial,
    TorqueProfileSource,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan
from shared.python.swing_sim.variation.execution_metadata import (
    LEGACY_CURRENT_REGISTRY_WARNING,
)
from shared.python.swing_sim.variation.persisted_plan_io import (
    persisted_plan_dumps,
    persisted_plan_loads,
)


def _torque_profile() -> PrescribedTorqueProfile:
    return PrescribedTorqueProfile(
        profile_id="profile.manual.1",
        model_id="double_pendulum",
        name="Manual test profile",
        description="Deterministic profile for the workspace contract.",
        source=TorqueProfileSource.DIRECT,
        source_metadata={"author": "test"},
        created_at_utc="2026-08-07T12:00:00Z",
        modified_at_utc="2026-08-07T12:00:00Z",
        time_domain_s=(0.0, 0.3),
        assignments=(
            JointTorqueAssignment(
                "joint.shoulder", TorquePolynomial((1.0, -0.5, 0.25))
            ),
        ),
    )


def _variation_plan() -> VariationPlan:
    return VariationPlan(
        mode="delivery",
        noise=(NoiseSpec("swing_sim.impact.delivery.face_angle_deg", scale=1.0),),
        n_runs=16,
        seed=42,
    )


def _document() -> WorkspaceDocument:
    plan = _variation_plan()
    return WorkspaceDocument(
        metadata=WorkspaceMetadata(
            document_id="workspace.demo.1",
            title="Showcase workspace",
            created_at_utc="2026-08-07T12:00:00Z",
            modified_at_utc="2026-08-07T12:05:00Z",
            app_version="1.0.0",
            provenance={"origin": "unit-test"},
        ),
        model_session=VersionedPayload(
            schema="rate_of_closure.model_session",
            schema_version=1,
            data={"source_kind": "manual", "impact_time_s": 0.03},
        ),
        prescribed_torque_profiles=(_torque_profile(),),
        club_configuration=VersionedPayload(
            schema="rate_of_closure.club_configuration",
            schema_version=1,
            data={"club_id": "driver.10_5", "head_mass_kg": 0.2},
        ),
        variation_plan=plan,
        layout=WorkspaceLayout(
            module_order=("simulation", "flight", "variation"),
            visible_module_ids=("simulation", "flight"),
            active_module_id="simulation",
            view_workspace=VersionedPayload(
                schema="rate_of_closure.view_workspace",
                schema_version=1,
                data={"mode": "single", "selected_view": "swing"},
            ),
        ),
        variation_plan_evidence=persisted_plan_loads(persisted_plan_dumps(plan)),
    )


def test_current_document_round_trip_is_deterministic_and_lossless() -> None:
    document = _document()

    encoded = workspace_to_json(document)
    decoded = workspace_from_json(encoded)

    assert decoded == document
    assert workspace_to_json(decoded) == encoded
    assert json.loads(encoded)["schema"] == WORKSPACE_SCHEMA
    assert json.loads(encoded)["schema_version"] == WORKSPACE_SCHEMA_VERSION
    binding = json.loads(encoded)["variation_plan"]
    assert binding["state"] == "canonical"
    assert binding["document"]["plan"] == _variation_plan().to_json_dict()


def test_v1_document_migrates_torque_name_and_versioned_layout() -> None:
    current = _document().to_json_dict()
    legacy = deepcopy(current)
    legacy["schema_version"] = 1
    legacy["torque_profiles"] = legacy.pop("prescribed_torque_profiles")
    legacy["layout"].pop("view_workspace")
    legacy["variation_plan"] = legacy["variation_plan"]["document"]["plan"]

    migrated = WorkspaceDocument.from_json_dict(legacy)

    assert migrated.prescribed_torque_profiles == (_torque_profile(),)
    assert migrated.layout.view_workspace is None
    assert migrated.to_json_dict()["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert migrated.variation_plan_evidence is not None
    assert migrated.variation_plan_evidence.warning == LEGACY_CURRENT_REGISTRY_WARNING


def test_v2_workspace_migrates_raw_plan_as_explicit_legacy_evidence() -> None:
    legacy = deepcopy(_document().to_json_dict())
    legacy["schema_version"] = 2
    legacy["variation_plan"] = legacy["variation_plan"]["document"]["plan"]

    migrated = WorkspaceDocument.from_json_dict(legacy)

    assert migrated.variation_plan == _variation_plan()
    assert migrated.variation_plan_evidence is not None
    assert migrated.variation_plan_evidence.metadata is None
    assert migrated.variation_plan_evidence.provenance is None
    assert migrated.variation_plan_evidence.warning == LEGACY_CURRENT_REGISTRY_WARNING
    assert migrated.to_json_dict()["variation_plan"]["state"] == "legacy"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra="forbidden"),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(schema_version=999),
        lambda value: value.update(schema="another.application"),
        lambda value: value["layout"].update(extra="forbidden"),
        lambda value: value["metadata"].update(extra="forbidden"),
    ],
)
def test_document_rejects_unknown_fields_and_unsupported_identity(mutate) -> None:  # type: ignore[no-untyped-def]
    payload = _document().to_json_dict()
    mutate(payload)

    with pytest.raises((TypeError, ValueError)):
        WorkspaceDocument.from_json_dict(payload)


def test_embedded_payload_rejects_non_json_and_non_finite_values() -> None:
    with pytest.raises((TypeError, ValueError)):
        VersionedPayload("example", 1, {"bad": object()})
    with pytest.raises((TypeError, ValueError)):
        VersionedPayload("example", 1, {"bad": float("nan")})
    with pytest.raises((TypeError, ValueError)):
        VersionedPayload("example", 1, {"bad": (1, 2)})


def test_payload_is_defensively_copied_and_immutable() -> None:
    source = {"nested": {"values": [1, 2]}}
    payload = VersionedPayload("example", 1, source)
    source["nested"]["values"].append(3)  # type: ignore[index,union-attr]

    assert payload.to_json_dict()["data"] == {"nested": {"values": [1, 2]}}
    with pytest.raises(TypeError):
        payload.data["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "layout",
    [
        WorkspaceLayout,
        lambda: WorkspaceLayout(
            ("simulation", "simulation"), ("simulation",), "simulation"
        ),
        lambda: WorkspaceLayout(("simulation",), (), "simulation"),
        lambda: WorkspaceLayout(("simulation",), ("unknown",), "unknown"),
        lambda: WorkspaceLayout(("simulation", "flight"), ("simulation",), "flight"),
    ],
)
def test_layout_rejects_unsafe_module_states(layout) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((TypeError, ValueError)):
        layout()


def test_duplicate_json_keys_are_rejected() -> None:
    valid = workspace_to_json(_document())
    duplicated = valid.replace(
        '"schema": "rate_of_closure.workspace",',
        '"schema": "rate_of_closure.workspace",\n  "schema": "duplicate",',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON field"):
        workspace_from_json(duplicated)


def test_non_finite_json_numbers_are_rejected_during_parse() -> None:
    valid = workspace_to_json(_document())
    non_finite = valid.replace('"impact_time_s": 0.03', '"impact_time_s": NaN')

    with pytest.raises(ValueError, match="non-finite JSON number"):
        workspace_from_json(non_finite)


def test_metadata_compares_timestamp_instants_not_timestamp_text() -> None:
    with pytest.raises(ValueError, match="must not precede creation"):
        WorkspaceMetadata(
            document_id="workspace.time.order",
            title="Invalid time order",
            created_at_utc="2026-08-07T12:00:00.1Z",
            modified_at_utc="2026-08-07T12:00:00Z",
            app_version="1.0.0",
            provenance={},
        )
