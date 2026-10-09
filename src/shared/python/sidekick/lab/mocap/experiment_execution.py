"""Replay time-grid, input, and execution-policy contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import (
    require_finite,
    require_semver,
    require_text,
    require_unique_text,
)
from .experiment_contracts import (
    ActuationInputKind,
    InputChannel,
    InputInterpolation,
    ReplayMode,
    _require_sha256,
)

_TORQUE_UNITS = frozenset({"N*m", "N m"})
_FORCE_UNITS = frozenset({"N"})
_NORMALIZED_UNIT = "1"
_SUPPORTED_TIMEBASE = "simulation_relative"


@dataclass(frozen=True, slots=True)
class ReplayExecutionPolicy:
    """Pinned numerical/contact setup and access limits for controller-off replay."""

    replay_mode: ReplayMode
    solver_id: str
    solver_version: str
    integration_method: str
    step_policy: str
    step_size_seconds: float | None
    initialization_policy_id: str
    initialization_policy_version: str
    input_player_id: str
    input_player_version: str
    observation_access: bool
    state_feedback_access: bool
    state_reset_allowed: bool
    contact_policy_id: str | None = None
    contact_policy_version: str | None = None
    contact_policy_sha256: str | None = None
    external_loads_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.replay_mode, ReplayMode):
            raise TypeError("replay_mode must be a ReplayMode")
        for field_name in (
            "solver_id",
            "integration_method",
            "step_policy",
            "initialization_policy_id",
            "input_player_id",
        ):
            object.__setattr__(
                self, field_name, require_text(getattr(self, field_name), field_name)
            )
        for field_name in (
            "solver_version",
            "initialization_policy_version",
            "input_player_version",
        ):
            object.__setattr__(
                self, field_name, require_semver(getattr(self, field_name), field_name)
            )
        if self.step_policy not in {"fixed", "adaptive"}:
            raise ValueError("step_policy must be 'fixed' or 'adaptive'")
        if self.step_policy == "fixed":
            if self.step_size_seconds is None:
                raise ValueError("fixed step policy requires step_size_seconds")
            step_size = require_finite(self.step_size_seconds, "step_size_seconds")
            if step_size <= 0.0:
                raise ValueError("step_size_seconds must be positive")
            object.__setattr__(self, "step_size_seconds", step_size)
        elif self.step_size_seconds is not None:
            step_size = require_finite(self.step_size_seconds, "step_size_seconds")
            if step_size <= 0.0:
                raise ValueError(
                    "adaptive step_size_seconds must be absent or positive"
                )
            object.__setattr__(self, "step_size_seconds", step_size)
        for field_name in (
            "observation_access",
            "state_feedback_access",
            "state_reset_allowed",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a boolean")
        if (
            self.observation_access
            or self.state_feedback_access
            or self.state_reset_allowed
        ):
            raise ValueError(
                "controller-off replay forbids observation, feedback, and "
                "state-reset access"
            )
        contact_values = (
            self.contact_policy_id,
            self.contact_policy_version,
            self.contact_policy_sha256,
        )
        if any(value is not None for value in contact_values):
            contact_id, contact_version, contact_digest = contact_values
            if contact_id is None or contact_version is None or contact_digest is None:
                raise ValueError(
                    "contact policy identity requires id, version, and sha256"
                )
            object.__setattr__(
                self,
                "contact_policy_id",
                require_text(contact_id, "contact_policy_id"),
            )
            object.__setattr__(
                self,
                "contact_policy_version",
                require_semver(contact_version, "contact_policy_version"),
            )
            _require_sha256(contact_digest, "contact_policy_sha256")
        if (
            self.replay_mode
            in {ReplayMode.NATIVE_OWN_CONTACT, ReplayMode.SHARED_RIGID_BODY_EMULATION}
            and self.contact_policy_sha256 is None
        ):
            raise ValueError("this replay mode requires a contact policy identity")
        if self.replay_mode is ReplayMode.EXTERNALLY_FORCED:
            if self.external_loads_sha256 is None:
                raise ValueError(
                    "externally_forced replay requires external-load identity"
                )
            _require_sha256(self.external_loads_sha256, "external_loads_sha256")
        elif self.external_loads_sha256 is not None:
            _require_sha256(self.external_loads_sha256, "external_loads_sha256")


@dataclass(frozen=True, slots=True)
class InputHistory:
    """Ordered values at the declared injection boundary on relative seconds.

    Interpolation applies to the saved channel value. For actuator commands,
    it does not imply that state-dependent downstream force is held constant.
    """

    input_kind: ActuationInputKind
    timebase_id: str
    interpolation: InputInterpolation
    time_seconds: tuple[float, ...]
    channels: tuple[InputChannel, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_kind, ActuationInputKind):
            raise TypeError("input_kind must be an ActuationInputKind")
        timebase_id = require_text(self.timebase_id, "timebase_id")
        if timebase_id != _SUPPORTED_TIMEBASE:
            raise ValueError(f"timebase_id must be {_SUPPORTED_TIMEBASE!r} in v1")
        object.__setattr__(self, "timebase_id", timebase_id)
        if not isinstance(self.interpolation, InputInterpolation):
            raise TypeError("interpolation must be an InputInterpolation")
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        times = tuple(
            require_finite(value, "time_seconds") for value in self.time_seconds
        )
        if len(times) < 2 or times[0] != 0.0:
            raise ValueError(
                "time_seconds must contain at least two samples and start at 0"
            )
        if any(right <= left for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("time_seconds must be strictly increasing")
        object.__setattr__(self, "time_seconds", times)
        if (
            self.input_kind
            in {
                ActuationInputKind.ACTUATOR_TORQUE,
                ActuationInputKind.ACTUATOR_FORCE,
                ActuationInputKind.GENERALIZED_EFFORT,
                ActuationInputKind.MUSCLE_EXCITATION,
                ActuationInputKind.MUSCLE_ACTIVATION,
            }
            and self.interpolation is not InputInterpolation.ZERO_ORDER_HOLD
        ):
            if (
                self.input_kind
                not in {
                    ActuationInputKind.MUSCLE_EXCITATION,
                    ActuationInputKind.MUSCLE_ACTIVATION,
                }
                or self.interpolation is not InputInterpolation.LINEAR
            ):
                raise ValueError(
                    "torque and effort histories require zero_order_hold; only "
                    "muscle inputs may use linear"
                )
        if not channels or any(
            not isinstance(channel, InputChannel) for channel in channels
        ):
            raise ValueError("channels must contain InputChannel values")
        channel_ids = tuple(channel.channel_id for channel in channels)
        require_unique_text(channel_ids, "channels")
        matrix = tuple(
            tuple(require_finite(value, "input_value") for value in row)
            for row in self.values
        )
        if len(matrix) != len(times) or any(
            len(row) != len(channels) for row in matrix
        ):
            raise ValueError("values must align with time samples and ordered channels")
        if self.input_kind in {
            ActuationInputKind.MUSCLE_EXCITATION,
            ActuationInputKind.MUSCLE_ACTIVATION,
        }:
            if any(channel.unit != _NORMALIZED_UNIT for channel in self.channels):
                raise ValueError(
                    "muscle excitation and activation channels require unit '1'"
                )
            if any(value < 0.0 or value > 1.0 for row in matrix for value in row):
                raise ValueError(
                    "muscle excitation and activation values must be within [0, 1]"
                )
        elif self.input_kind is ActuationInputKind.ACTUATOR_TORQUE:
            if any(channel.unit not in _TORQUE_UNITS for channel in self.channels):
                raise ValueError("actuator torque channels require N*m units")
        elif self.input_kind is ActuationInputKind.ACTUATOR_FORCE:
            if any(channel.unit not in _FORCE_UNITS for channel in self.channels):
                raise ValueError("actuator force channels require N units")
        elif self.input_kind is ActuationInputKind.EXTERNAL_LOAD:
            if any(
                channel.unit not in _TORQUE_UNITS | _FORCE_UNITS
                for channel in self.channels
            ):
                raise ValueError("force/load channels require N or N*m units")
        if self.input_kind in {
            ActuationInputKind.ACTUATOR_FORCE,
            ActuationInputKind.EXTERNAL_LOAD,
        } and any(channel.frame_id is None for channel in self.channels):
            raise ValueError(
                "actuator force and external-load channels require frame_id"
            )
        for channel in self.channels:
            if (
                self.input_kind is ActuationInputKind.MUSCLE_EXCITATION
                and channel.target_id.startswith("coordinate:")
            ):
                raise ValueError(
                    "muscle_excitation cannot target a generalized coordinate"
                )
            if (
                self.input_kind is ActuationInputKind.GENERALIZED_EFFORT
                and channel.unit == _NORMALIZED_UNIT
            ):
                raise ValueError("generalized_effort requires physical effort units")
        object.__setattr__(self, "values", matrix)


@dataclass(frozen=True, slots=True)
class IntegrityHashes:
    """Digests binding the declared model, state, inputs, and executed policy."""

    model_identity_sha256: str
    state_schema_sha256: str
    capability_declarations_sha256: str
    input_channel_schema_sha256: str
    time_grid_sha256: str
    initial_state_sha256: str
    applied_input_sha256: str
    execution_policy_sha256: str

    def __post_init__(self) -> None:
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
            _require_sha256(getattr(self, field_name), field_name)


__all__ = ["InputHistory", "IntegrityHashes", "ReplayExecutionPolicy"]
