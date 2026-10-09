"""Strict canonical JSON serialization for experiment replay bundles."""

from __future__ import annotations

import json
from typing import Any

from .experiment_contracts import (
    EXPERIMENT_REPLAY_SCHEMA_VERSION,
    ActuationInputKind,
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    InitialStateSchema,
    InitialStateValue,
    InputChannel,
    ModelIdentity,
    ReplayMode,
    StateComponentRole,
    StateComponentSpec,
)
from .experiment_execution import (
    InputHistory,
    InputInterpolation,
    IntegrityHashes,
    ReplayExecutionPolicy,
)
from .experiment_replay import ExperimentReplayBundle, _canonical_json


def dumps_experiment_replay_bundle(bundle: ExperimentReplayBundle) -> str:
    """Serialize a validated replay bundle as canonical JSON with a final newline."""
    if not isinstance(bundle, ExperimentReplayBundle):
        raise TypeError("bundle must be an ExperimentReplayBundle")
    return _canonical_json(bundle) + "\n"


def load_experiment_replay_bundle(text: str) -> ExperimentReplayBundle:
    """Load strict replay JSON and reject stale hashes or unsupported versions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("replay bundle must contain valid JSON") from exc
    _require_fields(
        payload,
        {
            "schema_version",
            "experiment_id",
            "model",
            "capabilities",
            "initial_state",
            "input_history",
            "policy",
            "integrity",
        },
        "bundle",
    )
    if payload["schema_version"] != EXPERIMENT_REPLAY_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {EXPERIMENT_REPLAY_SCHEMA_VERSION!r}")
    model = _load_model(payload["model"])
    capabilities = tuple(_load_capability(item) for item in payload["capabilities"])
    initial_state = tuple(_load_state_value(item) for item in payload["initial_state"])
    input_history = _load_input_history(payload["input_history"])
    policy = _load_policy(payload["policy"])
    integrity = _load_integrity(payload["integrity"])
    return ExperimentReplayBundle(
        payload["experiment_id"],
        model,
        capabilities,
        initial_state,
        input_history,
        policy,
        integrity,
        payload["schema_version"],
    )


def _require_fields(value: Any, expected: set[str], field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{field_name} fields differ; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return value


def _load_state_schema(value: Any) -> InitialStateSchema:
    data = _require_fields(
        value, {"schema_id", "version", "components"}, "state_schema"
    )
    specs = tuple(
        StateComponentSpec(
            **_require_fields(
                item,
                {"component_id", "role", "dimension", "unit", "representation"},
                "state_component_spec",
            )
            | {"role": StateComponentRole(item["role"])}
        )
        for item in data["components"]
    )
    return InitialStateSchema(data["schema_id"], data["version"], specs)


def _load_model(value: Any) -> ModelIdentity:
    expected = {
        "engine_id",
        "model_id",
        "variant_id",
        "model_version",
        "source_model_sha256",
        "provider_id",
        "provider_version",
        "provider_sha256",
        "state_schema",
        "ordered_input_channel_ids",
        "loaded_native_model_sha256",
    }
    data = _require_fields(value, expected, "model")
    return ModelIdentity(
        **(
            data
            | {
                "state_schema": _load_state_schema(data["state_schema"]),
                "ordered_input_channel_ids": tuple(data["ordered_input_channel_ids"]),
            }
        )
    )


def _load_capability(value: Any) -> CapabilityDeclaration:
    data = _require_fields(
        value,
        {"capability_id", "required", "support", "availability", "reason"},
        "capability",
    )
    return CapabilityDeclaration(
        **(
            data
            | {
                "support": CapabilitySupport(data["support"]),
                "availability": CapabilityAvailability(data["availability"]),
            }
        )
    )


def _load_state_value(value: Any) -> InitialStateValue:
    data = _require_fields(value, {"component_id", "values"}, "initial_state_component")
    return InitialStateValue(data["component_id"], tuple(data["values"]))


def _load_input_history(value: Any) -> InputHistory:
    data = _require_fields(
        value,
        {
            "input_kind",
            "timebase_id",
            "interpolation",
            "time_seconds",
            "channels",
            "values",
        },
        "input_history",
    )
    channels = tuple(
        InputChannel(
            **_require_fields(
                item,
                {"channel_id", "target_id", "unit", "coordinate_id", "frame_id"},
                "input_channel",
            )
        )
        for item in data["channels"]
    )
    return InputHistory(
        ActuationInputKind(data["input_kind"]),
        data["timebase_id"],
        InputInterpolation(data["interpolation"]),
        tuple(data["time_seconds"]),
        channels,
        tuple(tuple(row) for row in data["values"]),
    )


def _load_policy(value: Any) -> ReplayExecutionPolicy:
    expected = {
        "replay_mode",
        "solver_id",
        "solver_version",
        "integration_method",
        "step_policy",
        "step_size_seconds",
        "initialization_policy_id",
        "initialization_policy_version",
        "input_player_id",
        "input_player_version",
        "observation_access",
        "state_feedback_access",
        "state_reset_allowed",
        "contact_policy_id",
        "contact_policy_version",
        "contact_policy_sha256",
        "external_loads_sha256",
    }
    data = _require_fields(value, expected, "policy")
    return ReplayExecutionPolicy(
        **(data | {"replay_mode": ReplayMode(data["replay_mode"])})
    )


def _load_integrity(value: Any) -> IntegrityHashes:
    data = _require_fields(
        value,
        {
            "model_identity_sha256",
            "state_schema_sha256",
            "capability_declarations_sha256",
            "input_channel_schema_sha256",
            "time_grid_sha256",
            "initial_state_sha256",
            "applied_input_sha256",
            "execution_policy_sha256",
        },
        "integrity",
    )
    return IntegrityHashes(**data)


__all__ = ["dumps_experiment_replay_bundle", "load_experiment_replay_bundle"]
