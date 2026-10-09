"""Strict native-state envelope JSON; payload bytes are resolved separately."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any, TypeVar

from .experiment_replay import _canonical_json
from .native_state_artifact import (
    NativeStateArtifact,
    NativeStateClock,
    NativeStateEncoding,
    NativeStateExecution,
    NativeStateIdentity,
    NativeStateRole,
)
from .native_state_replay import (
    NATIVE_STATE_REPLAY_SCHEMA_VERSION,
    NativeReplayIntegrity,
    NativeReplayModel,
    NativeStateReplayEnvelope,
)
from .replay_serialization import (
    _load_capability,
    _load_input_history,
    _load_integrity,
    _load_policy,
    _require_fields,
)

_NativeGroup = TypeVar(
    "_NativeGroup",
    NativeStateEncoding,
    NativeStateIdentity,
    NativeStateClock,
    NativeStateExecution,
)


def dumps_native_state_replay_envelope(envelope: NativeStateReplayEnvelope) -> str:
    """Serialize validated metadata; no native artifact deserialization occurs."""
    if not isinstance(envelope, NativeStateReplayEnvelope):
        raise TypeError("envelope must be NativeStateReplayEnvelope")
    encoded: str = _canonical_json(envelope)
    return encoded + "\n"


def _load_group(value: Any, kind: type[_NativeGroup]) -> _NativeGroup:
    data = _require_fields(value, {item.name for item in fields(kind)}, kind.__name__)
    return kind(**data)


def _load_artifact(value: Any) -> NativeStateArtifact:
    data = _require_fields(
        value, {item.name for item in fields(NativeStateArtifact)}, "artifact"
    )
    return NativeStateArtifact(
        encoding=_load_group(data["encoding"], NativeStateEncoding),
        identity=_load_group(data["identity"], NativeStateIdentity),
        clock=_load_group(data["clock"], NativeStateClock),
        execution=_load_group(data["execution"], NativeStateExecution),
        payload_sha256=data["payload_sha256"],
        byte_size=data["byte_size"],
        role=NativeStateRole(data["role"]),
    )


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate native replay JSON field: {name}")
        result[name] = value
    return result


def _load_native_integrity(value: Any) -> NativeReplayIntegrity:
    data = _require_fields(value, {"components", "envelope_sha256"}, "integrity")
    return NativeReplayIntegrity(
        _load_integrity(data["components"]), data["envelope_sha256"]
    )


def load_native_state_replay_envelope(text: str) -> NativeStateReplayEnvelope:
    """Reject unknown fields, unsupported versions and stale state/input identity."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = json.loads(text, object_pairs_hook=_unique_fields)
    except json.JSONDecodeError as error:
        raise ValueError("native replay envelope must contain valid JSON") from error
    data = _require_fields(
        payload, {item.name for item in fields(NativeStateReplayEnvelope)}, "envelope"
    )
    if data["schema_version"] != NATIVE_STATE_REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported native-state replay schema_version")
    model = _require_fields(
        data["model"], {item.name for item in fields(NativeReplayModel)}, "model"
    )
    if not isinstance(model["ordered_input_channel_ids"], list):
        raise TypeError("ordered input channels must be a JSON array")
    return NativeStateReplayEnvelope(
        experiment_id=data["experiment_id"],
        model=NativeReplayModel(**model),
        artifact=_load_artifact(data["artifact"]),
        capabilities=tuple(_load_capability(item) for item in data["capabilities"]),
        input_history=_load_input_history(data["input_history"]),
        policy=_load_policy(data["policy"]),
        integrity=_load_native_integrity(data["integrity"]),
        schema_version=data["schema_version"],
    )
