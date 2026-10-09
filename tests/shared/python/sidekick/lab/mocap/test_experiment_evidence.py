from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from sidekick.lab.mocap import (
    ActuationInputKind,
    CapabilityAvailability,
    CapabilitySupport,
    ComparisonEvidenceRow,
    ComparisonLevel,
    ComparisonRowRequirement,
    DriveMode,
    EvidenceArtifactKind,
    EvidenceArtifactReference,
    ImplementationEvidence,
    ImplementationEvidenceKind,
    ReplayMode,
    build_comparison_evidence_receipt,
    build_experiment_replay_bundle,
    dumps_comparison_evidence_receipt,
    load_comparison_evidence_receipt,
)


def _replay_bundle(
    replay_mode: ReplayMode = ReplayMode.NATIVE_OWN_CONTACT,
    input_values: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,)),
    input_kind: ActuationInputKind = ActuationInputKind.ACTUATOR_TORQUE,
):
    from sidekick.lab.mocap import (
        CapabilityDeclaration,
        InitialStateSchema,
        InputChannel,
        InputInterpolation,
        ModelIdentity,
        ReplayExecutionPolicy,
        StateComponentRole,
        StateComponentSpec,
    )

    channel = (
        InputChannel("muscle", "muscle:synthetic", "1")
        if input_kind
        in {
            ActuationInputKind.MUSCLE_EXCITATION,
            ActuationInputKind.MUSCLE_ACTIVATION,
        }
        else InputChannel(
            "joint",
            "joint",
            "N" if input_kind is ActuationInputKind.ACTUATOR_FORCE else "N*m",
            frame_id=(
                "world" if input_kind is ActuationInputKind.ACTUATOR_FORCE else None
            ),
        )
    )

    state_components = [
        StateComponentSpec("q", StateComponentRole.POSITION, 1, "rad", "scalar"),
        StateComponentSpec("v", StateComponentRole.VELOCITY, 1, "rad/s", "scalar"),
    ]
    if input_kind in {
        ActuationInputKind.MUSCLE_EXCITATION,
        ActuationInputKind.MUSCLE_ACTIVATION,
    }:
        state_components.append(
            StateComponentSpec(
                "activation",
                StateComponentRole.MUSCLE_ACTIVATION,
                1,
                "1",
                "scalar",
            )
        )
    state_schema = InitialStateSchema(
        "synthetic-state",
        "1.0.0",
        tuple(state_components),
    )
    initial_state_values = [("q", (0.0,)), ("v", (0.0,))]
    if input_kind in {
        ActuationInputKind.MUSCLE_EXCITATION,
        ActuationInputKind.MUSCLE_ACTIVATION,
    }:
        initial_state_values.append(("activation", (0.0,)))
    model = ModelIdentity(
        engine_id="synthetic-engine",
        model_id="synthetic-model",
        variant_id="synthetic-variant",
        model_version="1.0.0",
        source_model_sha256="a" * 64,
        provider_id="synthetic-provider",
        provider_version="1.0.0",
        provider_sha256="b" * 64,
        state_schema=state_schema,
        ordered_input_channel_ids=(channel.channel_id,),
    )
    return build_experiment_replay_bundle(
        experiment_id="synthetic-evidence-run",
        model=model,
        capabilities=(
            CapabilityDeclaration(
                "forward_dynamics",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        initial_state_values=tuple(initial_state_values),
        channels=(channel,),
        input_kind=input_kind,
        interpolation=InputInterpolation.ZERO_ORDER_HOLD,
        time_seconds=(0.0, 0.01),
        input_values=input_values,
        policy=ReplayExecutionPolicy(
            replay_mode=replay_mode,
            solver_id="synthetic-solver",
            solver_version="1.0.0",
            integration_method="rk4",
            step_policy="fixed",
            step_size_seconds=0.01,
            initialization_policy_id="synthetic-init",
            initialization_policy_version="1.0.0",
            contact_policy_id="synthetic-contact",
            contact_policy_version="1.0.0",
            contact_policy_sha256="c" * 64,
            external_loads_sha256=(
                "f" * 64 if replay_mode is ReplayMode.EXTERNALLY_FORCED else None
            ),
            input_player_id="time-only-input-player",
            input_player_version="1.0.0",
            observation_access=False,
            state_feedback_access=False,
            state_reset_allowed=False,
        ),
    )


def _record(
    *,
    replay_mode: ReplayMode = ReplayMode.NATIVE_OWN_CONTACT,
    row_id: str = "synthetic-engine/model/variant/torque",
    input_values: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,)),
    input_kind: ActuationInputKind = ActuationInputKind.ACTUATOR_TORQUE,
    drive_mode: DriveMode = DriveMode.TORQUE,
):
    bundle = _replay_bundle(replay_mode, input_values, input_kind)
    return ComparisonEvidenceRow(
        row_id=row_id,
        package_id="synthetic-package",
        variant_id="synthetic-variant",
        drive_mode=drive_mode,
        replay_bundle=bundle,
        support=CapabilitySupport.SUPPORTED,
        availability=CapabilityAvailability.AVAILABLE,
        implementation_evidence=(
            ImplementationEvidence(
                kind=ImplementationEvidenceKind.DYNAMICS,
                implementation_id="synthetic-dynamics",
                version="1.0.0",
                sha256="d" * 64,
                required=True,
                support=CapabilitySupport.SUPPORTED,
                availability=CapabilityAvailability.AVAILABLE,
            ),
        ),
        artifacts=(
            EvidenceArtifactReference(
                kind=EvidenceArtifactKind.DYNAMICS,
                reference_id="opaque:synthetic-dynamics-001",
                sha256="e" * 64,
            ),
        ),
    )


def test_comparison_evidence_is_structural_data_not_qualification() -> None:
    requirement = ComparisonRowRequirement(
        row_id="synthetic-engine/model/variant/torque",
        required=True,
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        required_implementation_kinds=(ImplementationEvidenceKind.DYNAMICS,),
        required_artifact_kinds=(EvidenceArtifactKind.DYNAMICS,),
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-comparison",
        comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
        requirements=(requirement,),
        rows=(_record(),),
    )
    assert receipt.missing_required_rows == ()
    assert not hasattr(receipt, "qualified")
    assert not hasattr(receipt, "qualification_status")


def test_missing_required_model_engine_row_is_reported() -> None:
    requirement = ComparisonRowRequirement(
        "synthetic-engine/model/variant/torque",
        required=True,
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-missing-row",
        comparison_level=ComparisonLevel.SAME_INPUT,
        requirements=(requirement,),
        rows=(),
    )
    assert receipt.missing_required_rows == (requirement.row_id,)


def test_required_unavailable_row_is_retained_as_data_not_dropped() -> None:
    requirement = ComparisonRowRequirement(
        row_id="unavailable-engine/model/variant/torque",
        required=True,
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
    )
    unavailable = ComparisonEvidenceRow(
        row_id=requirement.row_id,
        package_id="unavailable-package",
        variant_id="synthetic-variant",
        drive_mode=DriveMode.TORQUE,
        replay_bundle=None,
        support=CapabilitySupport.UNKNOWN,
        availability=CapabilityAvailability.UNAVAILABLE,
        reason="synthetic provider is not installed",
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-unavailable-row",
        comparison_level=ComparisonLevel.SAME_INPUT,
        requirements=(requirement,),
        rows=(unavailable,),
    )
    assert receipt.rows == (unavailable,)
    assert receipt.missing_required_rows == (requirement.row_id,)


def test_weak_evidence_cannot_satisfy_stronger_required_replay_mode() -> None:
    requirement = ComparisonRowRequirement(
        "synthetic-engine/model/variant/torque",
        required=True,
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
    )
    with pytest.raises(ValueError, match="replay_mode"):
        build_comparison_evidence_receipt(
            receipt_id="synthetic-promoted-evidence",
            comparison_level=ComparisonLevel.SAME_INPUT,
            requirements=(requirement,),
            rows=(_record(replay_mode=ReplayMode.EXTERNALLY_FORCED),),
        )


def test_comparison_rejects_mixed_forced_and_native_replay_modes() -> None:
    first = ComparisonRowRequirement("engine-a/model/variant/torque", True)
    second = ComparisonRowRequirement("engine-b/model/variant/torque", True)
    native = _record(row_id=first.row_id)
    with pytest.raises(ValueError, match="cannot mix replay modes"):
        build_comparison_evidence_receipt(
            receipt_id="synthetic-mixed-modes",
            comparison_level=ComparisonLevel.SAME_INPUT,
            requirements=(first, second),
            rows=(
                native,
                _record(replay_mode=ReplayMode.EXTERNALLY_FORCED, row_id=second.row_id),
            ),
        )


def test_comparison_rejects_torque_and_excitation_drive_mode_mixture() -> None:
    torque_id = "engine-a/model/variant/torque"
    excitation_id = "engine-b/model/variant/muscle_excitation"
    with pytest.raises(ValueError, match="cannot mix drive modes"):
        build_comparison_evidence_receipt(
            receipt_id="synthetic-mixed-drive-modes",
            comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
            requirements=(
                ComparisonRowRequirement(torque_id, True),
                ComparisonRowRequirement(excitation_id, True),
            ),
            rows=(
                _record(row_id=torque_id),
                _record(
                    row_id=excitation_id,
                    drive_mode=DriveMode.MUSCLE_EXCITATION,
                    input_kind=ActuationInputKind.MUSCLE_EXCITATION,
                ),
            ),
        )


def test_same_input_level_requires_matching_payload_identity() -> None:
    first_id = "engine-a/model/variant/torque"
    second_id = "engine-b/model/variant/torque"
    with pytest.raises(ValueError, match="same_input comparison requires matching"):
        build_comparison_evidence_receipt(
            receipt_id="synthetic-different-inputs",
            comparison_level=ComparisonLevel.SAME_INPUT,
            requirements=(
                ComparisonRowRequirement(first_id, True),
                ComparisonRowRequirement(second_id, True),
            ),
            rows=(
                _record(row_id=first_id),
                _record(row_id=second_id, input_values=((0.0,), (2.0,))),
            ),
        )


def test_unknown_required_implementation_remains_a_structural_blocker() -> None:
    requirement = ComparisonRowRequirement(
        row_id="synthetic-engine/model/variant/torque",
        required=True,
        required_implementation_kinds=(),
    )
    row = ComparisonEvidenceRow(
        row_id=requirement.row_id,
        package_id="synthetic-package",
        variant_id="synthetic-variant",
        drive_mode=DriveMode.TORQUE,
        replay_bundle=_replay_bundle(),
        support=CapabilitySupport.SUPPORTED,
        availability=CapabilityAvailability.AVAILABLE,
        implementation_evidence=(
            ImplementationEvidence(
                kind=ImplementationEvidenceKind.CONTACT,
                implementation_id=None,
                version=None,
                sha256=None,
                required=True,
                support=CapabilitySupport.UNKNOWN,
                availability=CapabilityAvailability.UNKNOWN,
                reason="synthetic fixture has no contact implementation identity",
            ),
        ),
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-unknown-contact",
        comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
        requirements=(requirement,),
        rows=(row,),
    )
    assert receipt.missing_required_evidence == (
        f"{requirement.row_id}:implementation:contact",
    )


def test_absent_required_implementation_and_artifact_remain_blockers() -> None:
    requirement = ComparisonRowRequirement(
        row_id="synthetic-engine/model/variant/torque",
        required=True,
        required_implementation_kinds=(ImplementationEvidenceKind.CONTACT,),
        required_artifact_kinds=(EvidenceArtifactKind.FORCE,),
    )
    row = ComparisonEvidenceRow(
        row_id=requirement.row_id,
        package_id="synthetic-package",
        variant_id="synthetic-variant",
        drive_mode=DriveMode.TORQUE,
        replay_bundle=_replay_bundle(),
        support=CapabilitySupport.SUPPORTED,
        availability=CapabilityAvailability.AVAILABLE,
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-absent-required-evidence",
        comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
        requirements=(requirement,),
        rows=(row,),
    )
    assert receipt.missing_required_evidence == (
        f"{requirement.row_id}:artifact:force",
        f"{requirement.row_id}:implementation:contact",
    )


def test_comparison_receipt_round_trips_and_matches_strict_schema() -> None:
    requirement = ComparisonRowRequirement(
        row_id="synthetic-engine/model/variant/torque",
        required=True,
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        required_implementation_kinds=(ImplementationEvidenceKind.DYNAMICS,),
        required_artifact_kinds=(EvidenceArtifactKind.DYNAMICS,),
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-round-trip",
        comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
        requirements=(requirement,),
        rows=(_record(),),
    )
    encoded = dumps_comparison_evidence_receipt(receipt)
    assert load_comparison_evidence_receipt(encoded) == receipt
    assert encoded.endswith("\n")
    root = Path(__file__).resolve().parents[6]
    schema_path = root / "schemas" / "mocap" / "comparison-evidence-v1.schema.json"
    replay_schema_path = root / "schemas" / "mocap" / "experiment-replay-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    replay_schema = json.loads(replay_schema_path.read_text(encoding="utf-8"))
    replay_schema_uri = replay_schema["$id"]
    registry = Registry().with_resource(
        replay_schema_uri, Resource.from_contents(replay_schema)
    )
    validator = Draft202012Validator(schema, registry=registry)
    validator.validate(json.loads(encoded))
    payload = json.loads(encoded)
    payload["unexpected"] = "field"
    assert list(validator.iter_errors(payload))


def test_nested_replay_policy_tampering_is_rejected() -> None:
    requirement = ComparisonRowRequirement(
        row_id="synthetic-engine/model/variant/torque",
        required=True,
    )
    receipt = build_comparison_evidence_receipt(
        receipt_id="synthetic-tampered-policy",
        comparison_level=ComparisonLevel.WITHIN_ENGINE_REPLAY,
        requirements=(requirement,),
        rows=(_record(),),
    )
    payload = json.loads(dumps_comparison_evidence_receipt(receipt))
    payload["rows"][0]["replay_bundle"]["policy"]["step_size_seconds"] = 0.02
    with pytest.raises(ValueError, match="execution_policy_sha256"):
        load_comparison_evidence_receipt(json.dumps(payload))


def test_private_evidence_reference_rejects_paths() -> None:
    with pytest.raises(ValueError, match="opaque"):
        EvidenceArtifactReference(
            kind=EvidenceArtifactKind.OBSERVATION,
            reference_id="C:\\private\\subject.c3d",
            sha256="f" * 64,
        )


def test_drive_mode_preserves_coarse_class_separate_from_input_kind() -> None:
    row = _record()
    assert row.drive_mode is DriveMode.TORQUE
    assert row.replay_bundle is not None
    assert (
        row.replay_bundle.input_history.input_kind is ActuationInputKind.ACTUATOR_TORQUE
    )
    with pytest.raises(ValueError, match="drive_mode.*incompatible.*input_kind"):
        ComparisonEvidenceRow(
            row_id=row.row_id,
            package_id=row.package_id,
            variant_id=row.variant_id,
            drive_mode=DriveMode.MUSCLE_EXCITATION,
            replay_bundle=row.replay_bundle,
            support=row.support,
            availability=row.availability,
        )


@pytest.mark.parametrize(
    "input_kind",
    (
        ActuationInputKind.ACTUATOR_COMMAND,
        ActuationInputKind.ACTUATOR_FORCE,
        ActuationInputKind.GENERALIZED_EFFORT,
    ),
)
def test_f01_torque_drive_mode_accepts_only_declared_direct_effort_kinds(
    input_kind: ActuationInputKind,
) -> None:
    updated_bundle = _replay_bundle(input_kind=input_kind)
    assert ComparisonEvidenceRow(
        row_id="synthetic-effort-row",
        package_id="synthetic-package",
        variant_id="synthetic-variant",
        drive_mode=DriveMode.TORQUE,
        replay_bundle=updated_bundle,
        support=CapabilitySupport.SUPPORTED,
        availability=CapabilityAvailability.AVAILABLE,
    )


def test_torque_drive_mode_rejects_activation_input() -> None:
    updated_bundle = _replay_bundle(input_kind=ActuationInputKind.MUSCLE_ACTIVATION)
    with pytest.raises(ValueError, match="drive_mode.*incompatible.*input_kind"):
        ComparisonEvidenceRow(
            row_id="synthetic-activation-row",
            package_id="synthetic-package",
            variant_id="synthetic-variant",
            drive_mode=DriveMode.TORQUE,
            replay_bundle=updated_bundle,
            support=CapabilitySupport.SUPPORTED,
            availability=CapabilityAvailability.AVAILABLE,
        )
