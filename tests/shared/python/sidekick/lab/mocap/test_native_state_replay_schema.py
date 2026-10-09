"""Schema validation against a serialized native restart envelope."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
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
from sidekick.lab.mocap.native_state_artifact import (
    NativeStateArtifact,
    NativeStateClock,
    NativeStateEncoding,
    NativeStateExecution,
    NativeStateIdentity,
    NativeStateRole,
)
from sidekick.lab.mocap.native_state_replay import (
    NativeReplayModel,
    build_native_state_replay_envelope,
)
from sidekick.lab.mocap.native_state_replay_serialization import (
    dumps_native_state_replay_envelope,
)

SCHEMA = (
    Path(__file__).resolve().parents[6]
    / "schemas/mocap/native-state-replay-v1.1.schema.json"
)


def _serialized_native_envelope() -> dict[str, Any]:
    artifact = NativeStateArtifact(
        encoding=NativeStateEncoding("synthetic-native", "1.0.0", "RestartState"),
        identity=NativeStateIdentity(
            "synthetic", "runtime-build", "a" * 64, "b" * 64, "c" * 64
        ),
        clock=NativeStateClock(0.0, 0.25, "native-simulation-seconds"),
        execution=NativeStateExecution(
            "native-solver", "1.0.0", "exact-restart", "1.0.0", "d" * 64, "e" * 64
        ),
        payload_sha256="f" * 64,
        byte_size=12,
        role=NativeStateRole.COMPLETE_NATIVE_RESTART,
    )
    history = InputHistory(
        ActuationInputKind.ACTUATOR_TORQUE,
        "simulation_relative",
        InputInterpolation.ZERO_ORDER_HOLD,
        (0.0, 0.01),
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
        contact_policy_sha256="1" * 64,
    )
    envelope = build_native_state_replay_envelope(
        "native-test",
        NativeReplayModel("model", "variant", "1.0.0", ("motor",)),
        artifact,
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
    return json.loads(dumps_native_state_replay_envelope(envelope))


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    shared = json.loads(
        (SCHEMA.parent / "experiment-replay-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        uri=shared["$id"], resource=Resource.from_contents(shared)
    )
    return Draft202012Validator(schema, registry=registry)


def test_serialized_native_envelope_matches_standalone_schema() -> None:
    assert not list(_validator().iter_errors(_serialized_native_envelope()))


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), "native-state-replay/1.0.0"),
        (("unexpected",), True),
        (("artifact", "encoding", "unknown"), "covert-state"),
        (("input_history", "unknown"), "undocumented-channel"),
        (("policy", "unexpected_feedback"), True),
        (("artifact", "identity", "provider_sha256"), "truncated"),
        (("artifact", "encoding", "native_class"), ""),
        (("artifact", "byte_size"), 0),
        (("artifact", "role"), "observables_only"),
        (("artifact", "clock", "snapshot_time_seconds"), "0.25"),
    ],
)
def test_native_schema_rejects_invalid_wire_metadata(
    path: tuple[str, ...], value: object
) -> None:
    payload = deepcopy(_serialized_native_envelope())
    parent = payload
    for name in path[:-1]:
        parent = parent[name]
    parent[path[-1]] = value
    assert list(_validator().iter_errors(payload))
