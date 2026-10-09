"""Opaque restart bytes are not numeric observable-state substitutes."""

import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest
from sidekick.lab.mocap.native_state_artifact import (
    NativeStateArtifact,
    NativeStateClock,
    NativeStateEncoding,
    NativeStateExecution,
    NativeStateIdentity,
    NativeStateRole,
    freeze_native_state_artifact,
)


def _descriptor(payload: bytes = b"synthetic-opaque-restart") -> NativeStateArtifact:
    return NativeStateArtifact(
        encoding=NativeStateEncoding("synthetic-native", "1.0.0", "NativeRestart"),
        identity=NativeStateIdentity(
            "synthetic", "runtime-build-123", "a" * 64, "b" * 64, "c" * 64
        ),
        clock=NativeStateClock(0.0, 0.25, "native-simulation-seconds"),
        execution=NativeStateExecution(
            "native-solver", "1.0.0", "exact-restart", "1.0.0", "d" * 64, "e" * 64
        ),
        payload_sha256=sha256(payload).hexdigest(),
        byte_size=len(payload),
        role=NativeStateRole.COMPLETE_NATIVE_RESTART,
    )


def test_owned_restart_freezes_mutable_source_bytes() -> None:
    source = bytearray(b"synthetic-opaque-restart")
    owned = freeze_native_state_artifact(_descriptor(bytes(source)), source)
    source[:] = b"x" * len(source)
    assert owned.payload == b"synthetic-opaque-restart"
    assert owned.descriptor.clock.snapshot_time_seconds == 0.25
    with pytest.raises(FrozenInstanceError):
        owned.payload = b"changed"


@pytest.mark.parametrize(
    "payload", [b"", b"tampered", b"synthetic-opaque-restarts", b"x" * 24]
)
def test_missing_or_tampered_payload_is_rejected(payload: bytes) -> None:
    with pytest.raises(ValueError, match="size|digest|payload"):
        freeze_native_state_artifact(_descriptor(), payload)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_declared_native_artifact_size_is_positive_integer(size: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_descriptor(), byte_size=size)


@pytest.mark.parametrize(
    "start,snapshot",
    [(1.0, 0.25), (float("nan"), 0.25), (0.0, float("inf")), (True, 1.0)],
)
def test_native_clock_preserves_finite_snapshot_order(
    start: float, snapshot: float
) -> None:
    with pytest.raises((TypeError, ValueError)):
        NativeStateClock(start, snapshot, "native-simulation-seconds")


def test_observables_only_artifact_cannot_be_loaded_as_native_restart() -> None:
    descriptor = replace(_descriptor(), role=NativeStateRole.OBSERVABLES_ONLY)
    with pytest.raises(ValueError, match="complete|observable"):
        freeze_native_state_artifact(descriptor, b"synthetic-opaque-restart")


@pytest.mark.parametrize(
    "field", ["provider_sha256", "source_model_sha256", "loaded_model_sha256"]
)
def test_invalid_native_identity_digests_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        replace(_descriptor().identity, **{field: "not-a-sha256"})


@pytest.mark.parametrize(
    "field", ["effective_configuration_sha256", "compatibility_sha256"]
)
def test_invalid_effective_execution_identity_is_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        replace(_descriptor().execution, **{field: "not-a-sha256"})


def test_opaque_bytes_are_never_deserialized_by_shared_contract() -> None:
    payload = b"not-a-mat-file; shared-contract-does-not-execute"
    assert (
        freeze_native_state_artifact(_descriptor(payload), payload).payload == payload
    )


@pytest.mark.parametrize("payload", [3, True, [0, 0, 0]])
def test_non_byte_inputs_cannot_be_coerced_into_native_restart(payload: object) -> None:
    with pytest.raises(TypeError, match="bytes"):
        freeze_native_state_artifact(_descriptor(b"\x00\x00\x00"), payload)


def test_native_artifact_public_facade_uses_the_same_contract() -> None:
    import sidekick.lab.mocap as mocap

    assert mocap.NativeStateArtifact is NativeStateArtifact
    assert mocap.freeze_native_state_artifact is freeze_native_state_artifact


def _envelope():
    from sidekick.lab.mocap import (
        ActuationInputKind,
        CapabilityAvailability,
        CapabilityDeclaration,
        CapabilitySupport,
        InputChannel,
        InputHistory,
        InputInterpolation,
        ReplayExecutionPolicy,
        ReplayMode,
    )
    from sidekick.lab.mocap.native_state_replay import (
        NativeReplayModel,
        build_native_state_replay_envelope,
    )

    history = InputHistory(
        ActuationInputKind.ACTUATOR_TORQUE,
        "simulation_relative",
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.1),
        (InputChannel("motor", "motor", "N*m"),),
        ((0.2,), (0.2,)),
    )
    policy = ReplayExecutionPolicy(
        replay_mode=ReplayMode.NATIVE_OWN_CONTACT,
        solver_id="native-solver",
        solver_version="1.0.0",
        integration_method="native",
        step_policy="fixed",
        step_size_seconds=0.01,
        initialization_policy_id="exact-restart",
        initialization_policy_version="1.0.0",
        input_player_id="saved-torque",
        input_player_version="1.0.0",
        observation_access=False,
        state_feedback_access=False,
        state_reset_allowed=False,
        contact_policy_id="synthetic-no-contact",
        contact_policy_version="1.0.0",
        contact_policy_sha256="f" * 64,
    )
    return build_native_state_replay_envelope(
        "native-replay",
        NativeReplayModel("model", "variant", "1.0.0", ("motor",)),
        _descriptor(),
        (
            CapabilityDeclaration(
                "native-restart",
                True,
                CapabilitySupport.SUPPORTED,
                CapabilityAvailability.AVAILABLE,
            ),
        ),
        history,
        policy,
    )


def test_native_envelope_roundtrip_preserves_nonzero_snapshot_clock() -> None:
    from sidekick.lab.mocap.native_state_replay_serialization import (
        dumps_native_state_replay_envelope,
        load_native_state_replay_envelope,
    )

    envelope = _envelope()
    encoded = dumps_native_state_replay_envelope(envelope)
    restored = load_native_state_replay_envelope(encoded)
    assert restored == envelope
    assert restored.native_time_seconds == (0.25, 0.35)
    assert restored.artifact.clock.start_time_seconds == 0.0
    assert restored.artifact.clock.snapshot_time_seconds == 0.25
    assert not restored.blocking_capabilities


def test_numeric_reader_rejects_native_envelope_instead_of_dropping_hidden_state() -> (
    None
):
    from sidekick.lab.mocap import load_experiment_replay_bundle
    from sidekick.lab.mocap.native_state_replay_serialization import (
        dumps_native_state_replay_envelope,
    )

    encoded = dumps_native_state_replay_envelope(_envelope())
    with pytest.raises(ValueError):
        load_experiment_replay_bundle(encoded)


def test_native_envelope_rejects_observable_only_downgrade() -> None:
    envelope = _envelope()
    with pytest.raises(ValueError, match="complete|observable"):
        replace(
            envelope,
            artifact=replace(envelope.artifact, role=NativeStateRole.OBSERVABLES_ONLY),
        )


def test_native_envelope_integrity_binds_snapshot_time() -> None:
    envelope = _envelope()
    clock = replace(envelope.artifact.clock, snapshot_time_seconds=0.3)
    with pytest.raises(ValueError, match="hash|integrity"):
        replace(envelope, artifact=replace(envelope.artifact, clock=clock))


def test_native_envelope_rejects_changed_solver_identity() -> None:
    envelope = _envelope()
    with pytest.raises(ValueError, match="solver|policy|hash|integrity"):
        replace(envelope, policy=replace(envelope.policy, solver_id="other-solver"))


@pytest.mark.parametrize(
    "group,field,value",
    [
        ("identity", "runtime_id", "different-runtime"),
        ("identity", "source_model_sha256", "1" * 64),
        ("identity", "loaded_model_sha256", "2" * 64),
        ("encoding", "native_class", "ObservablesOnly"),
        ("execution", "effective_configuration_sha256", "3" * 64),
    ],
)
def test_native_envelope_integrity_binds_declared_native_metadata(
    group: str, field: str, value: str
) -> None:
    from sidekick.lab.mocap.native_state_replay_serialization import (
        dumps_native_state_replay_envelope,
        load_native_state_replay_envelope,
    )

    data = json.loads(dumps_native_state_replay_envelope(_envelope()))
    data["artifact"][group][field] = value
    with pytest.raises(ValueError, match="integrity|hash"):
        load_native_state_replay_envelope(json.dumps(data))


def test_native_envelope_rejects_duplicate_json_keys() -> None:
    from sidekick.lab.mocap.native_state_replay_serialization import (
        dumps_native_state_replay_envelope,
        load_native_state_replay_envelope,
    )

    text = dumps_native_state_replay_envelope(_envelope())
    text = text.replace(
        '"experiment_id":', '"experiment_id":"first","experiment_id":', 1
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_native_state_replay_envelope(text)


def test_native_envelope_rejects_collapsed_absolute_native_clock() -> None:
    envelope = _envelope()
    clock = replace(envelope.artifact.clock, snapshot_time_seconds=1e16)
    with pytest.raises(ValueError, match="precision|time grid"):
        replace(envelope, artifact=replace(envelope.artifact, clock=clock))


def test_native_envelope_integrity_binds_experiment_identity() -> None:
    envelope = _envelope()
    with pytest.raises(ValueError, match="integrity|hash"):
        replace(envelope, experiment_id="different-experiment")


def test_native_model_rejects_string_channel_sequence() -> None:
    from sidekick.lab.mocap.native_state_replay import NativeReplayModel

    with pytest.raises(TypeError, match="channels|array|sequence"):
        NativeReplayModel("model", "variant", "1.0.0", "m")


def test_native_loader_rejects_string_instead_of_channel_array() -> None:
    from sidekick.lab.mocap.native_state_replay import (
        build_native_state_replay_envelope,
    )
    from sidekick.lab.mocap.native_state_replay_serialization import (
        dumps_native_state_replay_envelope,
        load_native_state_replay_envelope,
    )

    original = _envelope()
    history = replace(
        original.input_history,
        channels=(replace(original.input_history.channels[0], channel_id="m"),),
    )
    model = replace(original.model, ordered_input_channel_ids=("m",))
    envelope = build_native_state_replay_envelope(
        "native-replay",
        model,
        original.artifact,
        original.capabilities,
        history,
        original.policy,
    )
    data = json.loads(dumps_native_state_replay_envelope(envelope))
    data["model"]["ordered_input_channel_ids"] = "m"
    with pytest.raises((TypeError, ValueError), match="array|channels|sequence"):
        load_native_state_replay_envelope(json.dumps(data))


def test_native_clock_rejects_distorted_intervals_before_hash_validation() -> None:
    from sidekick.lab.mocap.native_state_replay import (
        build_native_state_replay_envelope,
    )

    envelope = _envelope()
    artifact = replace(
        envelope.artifact,
        clock=replace(envelope.artifact.clock, snapshot_time_seconds=1e15),
    )
    with pytest.raises(ValueError, match="precision|interval|clock"):
        build_native_state_replay_envelope(
            "native-replay",
            envelope.model,
            artifact,
            envelope.capabilities,
            envelope.input_history,
            envelope.policy,
        )


def test_native_clock_rejects_unrepresentable_solver_step() -> None:
    from sidekick.lab.mocap.native_state_replay import (
        build_native_state_replay_envelope,
    )

    envelope = _envelope()
    artifact = replace(
        envelope.artifact,
        clock=replace(envelope.artifact.clock, snapshot_time_seconds=1e15),
    )
    history = replace(envelope.input_history, time_seconds=(0.0, 1.0))
    with pytest.raises(ValueError, match="precision|interval|clock"):
        build_native_state_replay_envelope(
            "native-replay",
            envelope.model,
            artifact,
            envelope.capabilities,
            history,
            envelope.policy,
        )


def test_required_unavailable_native_provider_remains_a_blocker() -> None:
    from sidekick.lab.mocap import CapabilityAvailability
    from sidekick.lab.mocap.native_state_replay import (
        build_native_state_replay_envelope,
    )

    envelope = _envelope()
    unavailable = replace(
        envelope.capabilities[0],
        availability=CapabilityAvailability.UNAVAILABLE,
        reason="provider absent on worker",
    )
    blocked = build_native_state_replay_envelope(
        "native-replay",
        envelope.model,
        envelope.artifact,
        (unavailable,),
        envelope.input_history,
        envelope.policy,
    )
    assert blocked.blocking_capabilities == ("native-restart",)


def test_native_envelope_public_facade_preserves_contract_authority() -> None:
    import sidekick.lab.mocap as mocap
    from sidekick.lab.mocap.native_state_replay import NativeStateReplayEnvelope

    assert mocap.NativeStateReplayEnvelope is NativeStateReplayEnvelope
    assert (
        mocap.load_native_state_replay_envelope(
            mocap.dumps_native_state_replay_envelope(_envelope())
        )
        == _envelope()
    )
