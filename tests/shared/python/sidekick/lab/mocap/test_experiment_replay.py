from __future__ import annotations

import json
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator
from sidekick.lab.mocap import (
    ActuationInputKind,
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    InitialStateSchema,
    InputChannel,
    InputInterpolation,
    ModelIdentity,
    ReplayExecutionPolicy,
    ReplayMode,
    StateComponentRole,
    StateComponentSpec,
    build_experiment_replay_bundle,
    dumps_experiment_replay_bundle,
    load_experiment_replay_bundle,
)


def _bundle(*, input_kind: ActuationInputKind = ActuationInputKind.ACTUATOR_TORQUE):
    state_schema = InitialStateSchema(
        schema_id="synthetic-state",
        version="1.0.0",
        components=(
            StateComponentSpec(
                component_id="q",
                role=StateComponentRole.POSITION,
                dimension=2,
                unit="rad",
                representation="model-native",
            ),
            StateComponentSpec(
                component_id="v",
                role=StateComponentRole.VELOCITY,
                dimension=1,
                unit="rad/s",
                representation="tangent-vector",
            ),
            StateComponentSpec(
                component_id="motor_state",
                role=StateComponentRole.ACTUATOR_INTERNAL_STATE,
                dimension=1,
                unit="1",
                representation="activation-state",
            ),
        ),
    )
    model = ModelIdentity(
        engine_id="synthetic-engine",
        model_id="two-link",
        variant_id="elastic-motor",
        model_version="1.0.0",
        source_model_sha256="a" * 64,
        provider_id="synthetic-provider",
        provider_version="1.0.0",
        provider_sha256="b" * 64,
        state_schema=state_schema,
        ordered_input_channel_ids=("shoulder", "elbow"),
    )
    channels = (
        InputChannel(
            "shoulder",
            "shoulder-motor",
            "1" if input_kind is ActuationInputKind.MUSCLE_EXCITATION else "N*m",
        ),
        InputChannel(
            "elbow",
            "elbow-motor",
            "1" if input_kind is ActuationInputKind.MUSCLE_EXCITATION else "N*m",
        ),
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="synthetic-rk",
        solver_version="1.0.0",
        integration_method="rk4",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="synthetic-init",
        initialization_policy_version="1.0.0",
        contact_policy_id="synthetic-contact",
        contact_policy_version="1.0.0",
        contact_policy_sha256="c" * 64,
        input_player_id="time-only-player",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
    )
    return build_experiment_replay_bundle(
        experiment_id="synthetic-run-001",
        model=model,
        capabilities=(
            CapabilityDeclaration(
                capability_id="native_contact",
                required=True,
                support=CapabilitySupport.SUPPORTED,
                availability=CapabilityAvailability.UNAVAILABLE,
                reason="synthetic engine exposes no contact backend",
            ),
        ),
        initial_state_values=(
            ("q", (0.0, 0.2)),
            ("v", (0.0,)),
            ("motor_state", (0.1,)),
        ),
        channels=channels,
        input_kind=input_kind,
        interpolation=InputInterpolation.ZERO_ORDER_HOLD,
        time_seconds=(0.0, 0.01, 0.02),
        input_values=(
            ((0.0, 0.0), (0.2, 0.1), (0.3, 0.2))
            if input_kind is ActuationInputKind.MUSCLE_EXCITATION
            else ((0.0, 0.0), (0.2, -0.1), (0.3, -0.2))
        ),
        policy=policy,
    )


def test_experiment_replay_bundle_is_strict_canonical_and_round_trips() -> None:
    bundle = _bundle()
    encoded = dumps_experiment_replay_bundle(bundle)
    assert encoded == dumps_experiment_replay_bundle(bundle)
    assert encoded.endswith("\n")
    assert load_experiment_replay_bundle(encoded) == bundle
    assert bundle.blocking_capabilities == ("native_contact",)
    assert not hasattr(bundle, "qualification_status")


def test_input_order_must_match_model_mapping() -> None:
    with pytest.raises(ValueError, match="ordered_input_channel_ids"):
        _bundle_with_reordered_channels()


def test_required_unsupported_capability_blocks_even_when_available() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
    kwargs["capabilities"] = (
        CapabilityDeclaration(
            capability_id="native_contact",
            required=True,
            support=CapabilitySupport.UNSUPPORTED,
            availability=CapabilityAvailability.AVAILABLE,
            reason="synthetic provider has no native contact implementation",
        ),
    )
    bundle = build_experiment_replay_bundle(**kwargs)
    assert bundle.blocking_capabilities == ("native_contact",)


def _bundle_with_reordered_channels():
    bundle = _bundle()
    return build_experiment_replay_bundle(
        experiment_id=bundle.experiment_id,
        model=bundle.model,
        capabilities=bundle.capabilities,
        initial_state_values=tuple(
            (component.component_id, component.values)
            for component in bundle.initial_state
        ),
        channels=tuple(reversed(bundle.input_history.channels)),
        input_kind=bundle.input_history.input_kind,
        interpolation=bundle.input_history.interpolation,
        time_seconds=bundle.input_history.time_seconds,
        input_values=tuple(tuple(reversed(row)) for row in bundle.input_history.values),
        policy=bundle.policy,
    )


def test_excitation_cannot_be_encoded_as_torque_and_requires_muscle_state() -> None:
    with pytest.raises(ValueError, match="muscle_activation"):
        _bundle(input_kind=ActuationInputKind.MUSCLE_EXCITATION)


def test_normalized_excitation_channels_are_not_torque_channels() -> None:
    bundle = _bundle()
    with pytest.raises(ValueError, match=r"N\*m"):
        build_experiment_replay_bundle(
            **(
                _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
                | {
                    "channels": tuple(
                        InputChannel(channel.channel_id, channel.target_id, "1")
                        for channel in bundle.input_history.channels
                    )
                }
            )
        )


def test_integrity_loader_rejects_policy_tampering() -> None:
    payload = json.loads(dumps_experiment_replay_bundle(_bundle()))
    payload["policy"]["step_size_seconds"] = 0.02
    with pytest.raises(ValueError, match="execution_policy_sha256"):
        load_experiment_replay_bundle(json.dumps(payload))


@pytest.mark.parametrize(
    ("section", "field", "value", "digest_name"),
    [
        ("model", "source_model_sha256", "e" * 64, "model_identity_sha256"),
        ("initial_state", "values", [0.0, 0.3], "initial_state_sha256"),
        ("input_history", "time_seconds", [0.0, 0.01, 0.03], "time_grid_sha256"),
        (
            "input_history",
            "values",
            [[0.0, 0.0], [0.2, -0.2], [0.3, -0.2]],
            "applied_input_sha256",
        ),
    ],
)
def test_integrity_loader_rejects_model_state_input_and_time_tampering(
    section: str, field: str, value: object, digest_name: str
) -> None:
    payload = json.loads(dumps_experiment_replay_bundle(_bundle()))
    target = payload[section]
    if section == "initial_state":
        target = target[0]
    target[field] = value
    with pytest.raises(ValueError, match=digest_name):
        load_experiment_replay_bundle(json.dumps(payload))


def test_integrity_exposes_time_grid_and_state_schema_digests() -> None:
    bundle = _bundle()
    assert bundle.time_grid_sha256 == bundle.integrity.time_grid_sha256
    assert bundle.state_schema_sha256 == bundle.integrity.state_schema_sha256
    assert bundle.policy_sha256 == bundle.integrity.execution_policy_sha256


def test_actuator_force_rejects_torque_units() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_FORCE)
    with pytest.raises(ValueError, match="require N units"):
        build_experiment_replay_bundle(**kwargs)


def test_force_history_requires_frame_and_binds_channel_schema() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_FORCE)
    kwargs["channels"] = tuple(
        InputChannel(channel.channel_id, channel.target_id, "N", frame_id="world")
        for channel in kwargs["channels"]
    )
    bundle = build_experiment_replay_bundle(**kwargs)
    assert bundle.input_history.timebase_id == "simulation_relative"
    assert (
        bundle.input_channel_schema_sha256
        == bundle.integrity.input_channel_schema_sha256
    )


def test_force_history_rejects_missing_frame() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_FORCE)
    kwargs["channels"] = tuple(
        InputChannel(channel.channel_id, channel.target_id, "N")
        for channel in kwargs["channels"]
    )
    with pytest.raises(ValueError, match="require frame_id"):
        build_experiment_replay_bundle(**kwargs)


@pytest.mark.parametrize(
    ("time_seconds", "message"),
    [((0.0, 0.01, 0.01), "strictly increasing"), ((0.0, float("nan"), 0.02), "finite")],
)
def test_input_time_grid_rejects_duplicate_and_non_finite_times(
    time_seconds: tuple[float, ...], message: str
) -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
    kwargs["time_seconds"] = time_seconds
    with pytest.raises(ValueError, match=message):
        build_experiment_replay_bundle(**kwargs)


def test_input_timebase_rejects_unmapped_external_clock() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
    kwargs["timebase_id"] = "camera_clock"
    with pytest.raises(ValueError, match="simulation_relative"):
        build_experiment_replay_bundle(**kwargs)


def test_bundle_copies_mutable_input_sequences_before_hashing() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
    mutable_channels = list(kwargs["channels"])
    mutable_capabilities = list(kwargs["capabilities"])
    kwargs["channels"] = mutable_channels
    kwargs["capabilities"] = mutable_capabilities
    bundle = build_experiment_replay_bundle(**kwargs)
    mutable_channels.clear()
    mutable_capabilities.clear()
    assert len(bundle.input_history.channels) == 2
    assert len(bundle.capabilities) == 1
    assert (
        load_experiment_replay_bundle(dumps_experiment_replay_bundle(bundle)) == bundle
    )


def test_integrity_loader_rejects_frame_mapping_tampering() -> None:
    payload = json.loads(dumps_experiment_replay_bundle(_bundle()))
    payload["input_history"]["channels"][0]["frame_id"] = "unexpected-frame"
    with pytest.raises(ValueError, match="input_channel_schema_sha256"):
        load_experiment_replay_bundle(json.dumps(payload))


def test_integrity_loader_rejects_capability_declaration_tampering() -> None:
    payload = json.loads(dumps_experiment_replay_bundle(_bundle()))
    payload["capabilities"][0]["availability"] = "available"
    with pytest.raises(ValueError, match="capability_declarations_sha256"):
        load_experiment_replay_bundle(json.dumps(payload))


def test_replay_policy_rejects_feedback_reset_and_missing_external_load_identity() -> (
    None
):
    policy = _bundle().policy
    with pytest.raises(ValueError, match="forbids observation"):
        replace(policy, state_feedback_access=True)
    with pytest.raises(ValueError, match="fixed step policy requires"):
        replace(policy, step_size_seconds=None)
    with pytest.raises(ValueError, match="requires external-load identity"):
        replace(policy, replay_mode=ReplayMode.EXTERNALLY_FORCED)


def test_torque_history_requires_zoh_and_finite_increasing_samples() -> None:
    kwargs = _bundle_kwargs(input_kind=ActuationInputKind.ACTUATOR_TORQUE)
    kwargs["interpolation"] = InputInterpolation.LINEAR
    with pytest.raises(ValueError, match="zero_order_hold"):
        build_experiment_replay_bundle(**kwargs)


def _bundle_kwargs(*, input_kind: ActuationInputKind):
    bundle = _bundle()
    normalized_input = input_kind in {
        ActuationInputKind.MUSCLE_EXCITATION,
        ActuationInputKind.MUSCLE_ACTIVATION,
    }
    return {
        "experiment_id": bundle.experiment_id,
        "model": bundle.model,
        "capabilities": bundle.capabilities,
        "initial_state_values": tuple(
            (component.component_id, component.values)
            for component in bundle.initial_state
        ),
        "channels": tuple(
            InputChannel(
                channel.channel_id,
                channel.target_id,
                "1" if normalized_input else channel.unit,
            )
            for channel in bundle.input_history.channels
        ),
        "input_kind": input_kind,
        "interpolation": InputInterpolation.ZERO_ORDER_HOLD,
        "time_seconds": bundle.input_history.time_seconds,
        "input_values": (
            ((0.0, 0.0), (0.2, 0.1), (0.3, 0.2))
            if normalized_input
            else bundle.input_history.values
        ),
        "policy": bundle.policy,
    }


def test_json_schema_accepts_synthetic_bundle_and_rejects_unknown_fields() -> None:
    from pathlib import Path

    schema_path = (
        Path(__file__).resolve().parents[6]
        / "schemas"
        / "mocap"
        / "experiment-replay-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    payload = json.loads(dumps_experiment_replay_bundle(_bundle()))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    payload["private_subject_path"] = "must be rejected"
    assert list(Draft202012Validator(schema).iter_errors(payload))
