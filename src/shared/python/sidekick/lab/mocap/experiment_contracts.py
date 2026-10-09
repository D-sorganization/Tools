"""Shared value objects for versioned experiment/replay contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ._validation import (
    require_finite,
    require_semver,
    require_text,
    require_unique_text,
)

EXPERIMENT_REPLAY_SCHEMA_VERSION = "experiment-replay/1.0.0"
_DIGEST_LENGTH = 64


class CapabilityAvailability(StrEnum):
    """Runtime availability; this is not support or qualification evidence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CapabilitySupport(StrEnum):
    """Declared support kept independent from runtime availability."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ActuationInputKind(StrEnum):
    """Meaning of the time-indexed input values in a replay bundle."""

    ACTUATOR_COMMAND = "actuator_command"
    ACTUATOR_TORQUE = "actuator_torque"
    ACTUATOR_FORCE = "actuator_force"
    GENERALIZED_EFFORT = "generalized_effort"
    MUSCLE_EXCITATION = "muscle_excitation"
    MUSCLE_ACTIVATION = "muscle_activation"
    EXTERNAL_LOAD = "external_load"


class InputInterpolation(StrEnum):
    """Interpolation applied to saved inputs between their sample times."""

    ZERO_ORDER_HOLD = "zero_order_hold"
    LINEAR = "linear"


class ReplayMode(StrEnum):
    """Physical interpretation of the replay plant and contact inputs."""

    NATIVE_OWN_CONTACT = "native_own_contact"
    EXTERNALLY_FORCED = "externally_forced"
    SHARED_RIGID_BODY_EMULATION = "shared_rigid_body_emulation"


class StateComponentRole(StrEnum):
    """Role of one named component in a model's complete initial state."""

    POSITION = "position"
    VELOCITY = "velocity"
    MUSCLE_ACTIVATION = "muscle_activation"
    MUSCLE_FIBER_STATE = "muscle_fiber_state"
    MUSCLE_TENDON_STATE = "muscle_tendon_state"
    ACTUATOR_INTERNAL_STATE = "actuator_internal_state"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True, slots=True)
class CapabilityDeclaration:
    """A required capability and its separate availability evidence."""

    capability_id: str
    required: bool
    support: CapabilitySupport
    availability: CapabilityAvailability
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capability_id", require_text(self.capability_id, "capability_id")
        )
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        if not isinstance(self.support, CapabilitySupport):
            raise TypeError("support must be a CapabilitySupport")
        if not isinstance(self.availability, CapabilityAvailability):
            raise TypeError("availability must be a CapabilityAvailability")
        if self.reason is not None:
            object.__setattr__(self, "reason", require_text(self.reason, "reason"))
        if (
            self.support is not CapabilitySupport.SUPPORTED
            or self.availability is not CapabilityAvailability.AVAILABLE
        ) and self.reason is None:
            raise ValueError(
                "unsupported or unavailable capability declarations require a reason"
            )


@dataclass(frozen=True, slots=True)
class StateComponentSpec:
    """Versioned shape and meaning of one named model-native state component."""

    component_id: str
    role: StateComponentRole
    dimension: int
    unit: str
    representation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", require_text(self.component_id, "component_id")
        )
        if not isinstance(self.role, StateComponentRole):
            raise TypeError("role must be a StateComponentRole")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        object.__setattr__(self, "unit", require_text(self.unit, "unit"))
        object.__setattr__(
            self, "representation", require_text(self.representation, "representation")
        )


@dataclass(frozen=True, slots=True)
class InitialStateSchema:
    """Model-authored schema for the complete physical initial state."""

    schema_id: str
    version: str
    components: tuple[StateComponentSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", require_text(self.schema_id, "schema_id"))
        object.__setattr__(self, "version", require_semver(self.version, "version"))
        components = tuple(self.components)
        if not components or any(
            not isinstance(item, StateComponentSpec) for item in components
        ):
            raise ValueError("components must contain StateComponentSpec values")
        object.__setattr__(self, "components", components)
        ids = tuple(item.component_id for item in components)
        require_unique_text(ids, "components")
        roles = {item.role for item in self.components}
        if (
            StateComponentRole.POSITION not in roles
            or StateComponentRole.VELOCITY not in roles
        ):
            raise ValueError("state schema requires position and velocity components")


@dataclass(frozen=True, slots=True)
class InitialStateValue:
    """Finite values for one component in a model's initial state."""

    component_id: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", require_text(self.component_id, "component_id")
        )
        if not self.values:
            raise ValueError("values must be non-empty")
        object.__setattr__(
            self,
            "values",
            tuple(
                require_finite(value, "initial_state_value") for value in self.values
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable engine, model, provider, state-schema, and input-order identity."""

    engine_id: str
    model_id: str
    variant_id: str
    model_version: str
    source_model_sha256: str
    provider_id: str
    provider_version: str
    provider_sha256: str
    state_schema: InitialStateSchema
    ordered_input_channel_ids: tuple[str, ...]
    loaded_native_model_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("engine_id", "model_id", "variant_id", "provider_id"):
            object.__setattr__(
                self, field_name, require_text(getattr(self, field_name), field_name)
            )
        for field_name in ("model_version", "provider_version"):
            object.__setattr__(
                self, field_name, require_semver(getattr(self, field_name), field_name)
            )
        for field_name in ("source_model_sha256", "provider_sha256"):
            _require_sha256(getattr(self, field_name), field_name)
        if self.loaded_native_model_sha256 is not None:
            _require_sha256(
                self.loaded_native_model_sha256, "loaded_native_model_sha256"
            )
        if not isinstance(self.state_schema, InitialStateSchema):
            raise TypeError("state_schema must be an InitialStateSchema")
        channel_ids = require_unique_text(
            self.ordered_input_channel_ids, "ordered_input_channel_ids"
        )
        if not channel_ids:
            raise ValueError("ordered_input_channel_ids must be non-empty")
        object.__setattr__(self, "ordered_input_channel_ids", channel_ids)


@dataclass(frozen=True, slots=True)
class InputChannel:
    """One ordered input target with an explicit unit and optional coordinate."""

    channel_id: str
    target_id: str
    unit: str
    coordinate_id: str | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("channel_id", "target_id", "unit"):
            object.__setattr__(
                self, field_name, require_text(getattr(self, field_name), field_name)
            )
        if self.coordinate_id is not None:
            object.__setattr__(
                self, "coordinate_id", require_text(self.coordinate_id, "coordinate_id")
            )
        if self.frame_id is not None:
            object.__setattr__(
                self, "frame_id", require_text(self.frame_id, "frame_id")
            )


def _require_sha256(value: str, field_name: str) -> str:
    normalized = require_text(value, field_name)
    if len(normalized) != _DIGEST_LENGTH or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return normalized


__all__ = [
    "EXPERIMENT_REPLAY_SCHEMA_VERSION",
    "ActuationInputKind",
    "CapabilityAvailability",
    "CapabilityDeclaration",
    "CapabilitySupport",
    "InitialStateSchema",
    "InitialStateValue",
    "InputChannel",
    "InputInterpolation",
    "ModelIdentity",
    "ReplayMode",
    "StateComponentRole",
    "StateComponentSpec",
]
