"""Validated experiment replay bundle assembly and integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Any

from ._validation import require_text, require_unique_text
from .experiment_contracts import (
    EXPERIMENT_REPLAY_SCHEMA_VERSION,
    ActuationInputKind,
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    InitialStateValue,
    InputChannel,
    ModelIdentity,
    StateComponentRole,
)
from .experiment_execution import (
    InputHistory,
    InputInterpolation,
    IntegrityHashes,
    ReplayExecutionPolicy,
)


@dataclass(frozen=True, slots=True)
class ExperimentReplayBundle:
    """Immutable experiment/replay data; it carries no physics qualification."""

    experiment_id: str
    model: ModelIdentity
    capabilities: tuple[CapabilityDeclaration, ...]
    initial_state: tuple[InitialStateValue, ...]
    input_history: InputHistory
    policy: ReplayExecutionPolicy
    integrity: IntegrityHashes
    schema_version: str = EXPERIMENT_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", require_text(self.experiment_id, "experiment_id")
        )
        if self.schema_version != EXPERIMENT_REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {EXPERIMENT_REPLAY_SCHEMA_VERSION!r}"
            )
        if not isinstance(self.model, ModelIdentity) or not isinstance(
            self.input_history, InputHistory
        ):
            raise TypeError(
                "model and input_history must use public replay contract types"
            )
        if not isinstance(self.policy, ReplayExecutionPolicy) or not isinstance(
            self.integrity, IntegrityHashes
        ):
            raise TypeError(
                "policy and integrity must use public replay contract types"
            )
        capabilities = tuple(self.capabilities)
        initial_state = tuple(self.initial_state)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "initial_state", initial_state)
        if not capabilities or any(
            not isinstance(item, CapabilityDeclaration) for item in capabilities
        ):
            raise ValueError("capabilities must contain CapabilityDeclaration values")
        require_unique_text(
            tuple(item.capability_id for item in capabilities), "capabilities"
        )
        self._validate_initial_state()
        self._validate_input_mapping()
        self._validate_integrity()

    @property
    def blocking_capabilities(self) -> tuple[str, ...]:
        """Return required capabilities that are not currently available."""
        return tuple(
            item.capability_id
            for item in self.capabilities
            if item.required
            and (
                item.support is not CapabilitySupport.SUPPORTED
                or item.availability is not CapabilityAvailability.AVAILABLE
            )
        )

    @property
    def state_schema_sha256(self) -> str:
        """Digest of the named state representation used by this bundle."""
        return self.integrity.state_schema_sha256

    @property
    def input_channel_schema_sha256(self) -> str:
        """Digest of the ordered channel target, unit, coordinate, and frame mapping."""
        return self.integrity.input_channel_schema_sha256

    @property
    def applied_input_sha256(self) -> str:
        """Digest of the exact ordered input history and time grid."""
        return self.integrity.applied_input_sha256

    @property
    def time_grid_sha256(self) -> str:
        """Digest of the exact simulation-relative input sample times."""
        return self.integrity.time_grid_sha256

    @property
    def policy_sha256(self) -> str:
        """Digest of the integration, initialization, contact, and player policy."""
        return self.integrity.execution_policy_sha256

    def _validate_initial_state(self) -> None:
        schema = self.model.state_schema
        if not self.initial_state or any(
            not isinstance(value, InitialStateValue) for value in self.initial_state
        ):
            raise ValueError("initial_state must contain named state values")
        actual_ids = tuple(value.component_id for value in self.initial_state)
        expected_ids = tuple(component.component_id for component in schema.components)
        if actual_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(actual_ids))
            extra = sorted(set(actual_ids) - set(expected_ids))
            raise ValueError(
                "initial_state must match the complete ordered state schema; "
                f"missing={missing}, extra={extra}"
            )
        component_by_id = {
            component.component_id: component for component in schema.components
        }
        for value in self.initial_state:
            expected = component_by_id[value.component_id]
            if len(value.values) != expected.dimension:
                raise ValueError(
                    f"initial state dimension differs for {value.component_id!r}"
                )
        if self.input_history.input_kind in {
            ActuationInputKind.MUSCLE_EXCITATION,
            ActuationInputKind.MUSCLE_ACTIVATION,
        } and StateComponentRole.MUSCLE_ACTIVATION not in {
            component.role for component in schema.components
        }:
            raise ValueError(
                "muscle input requires a complete state schema with muscle_activation"
            )

    def _validate_input_mapping(self) -> None:
        actual_ids = tuple(
            channel.channel_id for channel in self.input_history.channels
        )
        if actual_ids != self.model.ordered_input_channel_ids:
            raise ValueError("ordered_input_channel_ids do not match the model mapping")

    def _validate_integrity(self) -> None:
        expected = _expected_hashes(
            self.model,
            self.capabilities,
            self.initial_state,
            self.input_history,
            self.policy,
        )
        for field_name in (
            "model_identity_sha256",
            "state_schema_sha256",
            "capability_declarations_sha256",
            "input_channel_schema_sha256",
            "time_grid_sha256",
            "initial_state_sha256",
            "applied_input_sha256",
            "execution_policy_sha256",
        ):
            if getattr(expected, field_name) != getattr(self.integrity, field_name):
                raise ValueError(f"replay payload hash mismatch: {field_name}")


def build_experiment_replay_bundle(
    experiment_id: str,
    model: ModelIdentity,
    capabilities: tuple[CapabilityDeclaration, ...],
    initial_state_values: tuple[tuple[str, tuple[float, ...]], ...],
    channels: tuple[InputChannel, ...],
    input_kind: ActuationInputKind,
    interpolation: InputInterpolation,
    time_seconds: tuple[float, ...],
    input_values: tuple[tuple[float, ...], ...],
    policy: ReplayExecutionPolicy,
    timebase_id: str = "simulation_relative",
) -> ExperimentReplayBundle:
    """Build a replay bundle and compute digests over its complete payload."""
    capabilities = tuple(capabilities)
    channels = tuple(channels)
    state = tuple(
        InitialStateValue(component_id, values)
        for component_id, values in initial_state_values
    )
    history = InputHistory(
        input_kind, timebase_id, interpolation, time_seconds, channels, input_values
    )
    hashes = _expected_hashes(model, capabilities, state, history, policy)
    return ExperimentReplayBundle(
        experiment_id, model, capabilities, state, history, policy, hashes
    )


def _expected_hashes(
    model: ModelIdentity,
    capabilities: tuple[CapabilityDeclaration, ...],
    initial_state: tuple[InitialStateValue, ...],
    input_history: InputHistory,
    policy: ReplayExecutionPolicy,
) -> IntegrityHashes:
    return IntegrityHashes(
        model_identity_sha256=_sha256(model),
        state_schema_sha256=_sha256(model.state_schema),
        capability_declarations_sha256=_sha256(capabilities),
        input_channel_schema_sha256=_sha256(input_history.channels),
        time_grid_sha256=_sha256(
            {
                "timebase_id": input_history.timebase_id,
                "time_seconds": input_history.time_seconds,
            }
        ),
        initial_state_sha256=_sha256(initial_state),
        applied_input_sha256=_sha256(input_history),
        execution_policy_sha256=_sha256(policy),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _to_primitive(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _to_primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    return value


__all__ = ["ExperimentReplayBundle", "build_experiment_replay_bundle"]
