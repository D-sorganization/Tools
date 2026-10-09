"""Native restart envelope with explicit opaque-state and physical-time authority."""

from __future__ import annotations

from dataclasses import dataclass
from math import ulp

from ._validation import (
    require_finite,
    require_semver,
    require_text,
    require_unique_text,
)
from .experiment_contracts import (
    CapabilityAvailability,
    CapabilityDeclaration,
    CapabilitySupport,
    _require_sha256,
)
from .experiment_execution import InputHistory, IntegrityHashes, ReplayExecutionPolicy
from .experiment_replay import _sha256
from .native_state_artifact import NativeStateArtifact, NativeStateRole

NATIVE_STATE_REPLAY_SCHEMA_VERSION = "native-state-replay/1.1.0"


@dataclass(frozen=True, slots=True)
class NativeReplayIntegrity:
    """Shared component digests plus a digest of the entire native declaration."""

    components: IntegrityHashes
    envelope_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.components, IntegrityHashes):
            raise TypeError("components must be IntegrityHashes")
        object.__setattr__(
            self,
            "envelope_sha256",
            _require_sha256(self.envelope_sha256, "envelope_sha256"),
        )


@dataclass(frozen=True, slots=True)
class NativeReplayModel:
    """Inventory model identity; native source/runtime bindings live in the artifact."""

    model_id: str
    variant_id: str
    model_version: str
    ordered_input_channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("model_id", "variant_id"):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        object.__setattr__(
            self, "model_version", require_semver(self.model_version, "model_version")
        )
        supplied_channels: object = self.ordered_input_channel_ids
        if isinstance(supplied_channels, (str, bytes)):
            raise TypeError("ordered input channels must be a sequence, not text")
        channels = require_unique_text(
            self.ordered_input_channel_ids, "ordered_input_channel_ids"
        )
        if not channels:
            raise ValueError("ordered_input_channel_ids must be non-empty")
        object.__setattr__(self, "ordered_input_channel_ids", channels)


@dataclass(frozen=True, slots=True)
class NativeStateReplayEnvelope:
    """Replay declaration whose complete initial state is an opaque native artifact.

    Numeric observations are deliberately absent from the restart authority.
    Input times remain interval-relative; the native clock starts at the saved
    snapshot without retiming the native operating point. No physics is qualified.
    """

    experiment_id: str
    model: NativeReplayModel
    artifact: NativeStateArtifact
    capabilities: tuple[CapabilityDeclaration, ...]
    input_history: InputHistory
    policy: ReplayExecutionPolicy
    integrity: NativeReplayIntegrity
    schema_version: str = NATIVE_STATE_REPLAY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", require_text(self.experiment_id, "experiment_id")
        )
        if self.schema_version != NATIVE_STATE_REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported native-state replay schema_version")
        for name, expected in (
            ("model", NativeReplayModel),
            ("artifact", NativeStateArtifact),
            ("input_history", InputHistory),
            ("policy", ReplayExecutionPolicy),
            ("integrity", NativeReplayIntegrity),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if self.artifact.role is not NativeStateRole.COMPLETE_NATIVE_RESTART:
            raise ValueError(
                "complete native restart required; observables are insufficient"
            )
        capabilities = tuple(self.capabilities)
        if not capabilities or any(
            not isinstance(item, CapabilityDeclaration) for item in capabilities
        ):
            raise ValueError("capabilities must contain CapabilityDeclaration values")
        require_unique_text(
            (item.capability_id for item in capabilities), "capabilities"
        )
        object.__setattr__(self, "capabilities", capabilities)
        self._validate_execution()
        expected = _native_state_hashes(
            self.experiment_id,
            self.model,
            self.artifact,
            capabilities,
            self.input_history,
            self.policy,
        )
        if self.integrity != expected:
            raise ValueError("native replay integrity hash mismatch")

    def _validate_execution(self) -> None:
        channels = tuple(item.channel_id for item in self.input_history.channels)
        if channels != self.model.ordered_input_channel_ids:
            raise ValueError(
                "ordered input channels differ from the native model mapping"
            )
        execution = self.artifact.execution
        if (self.policy.solver_id, self.policy.solver_version) != (
            execution.solver_id,
            execution.solver_version,
        ):
            raise ValueError("solver identity differs from the native restart artifact")
        if (
            self.policy.initialization_policy_id,
            self.policy.initialization_policy_version,
        ) != (execution.policy_id, execution.policy_version):
            raise ValueError("initialization policy differs from the native artifact")
        times = self.native_time_seconds
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("native time grid collapses at this snapshot precision")
        relative = self.input_history.time_seconds
        intervals = tuple(b - a for a, b in zip(relative, relative[1:], strict=False))
        actual = tuple(b - a for a, b in zip(times, times[1:], strict=False))
        for requested, represented in zip(intervals, actual, strict=True):
            _require_clock_interval(requested, represented)
        step = self.policy.step_size_seconds
        if step is not None:
            snapshot = self.artifact.clock.snapshot_time_seconds
            _require_clock_interval(step, (snapshot + step) - snapshot)

    @property
    def native_time_seconds(self) -> tuple[float, ...]:
        """Absolute native times; never overwrite the saved snapshot time."""
        return tuple(
            require_finite(
                self.artifact.clock.snapshot_time_seconds + value, "native_time_seconds"
            )
            for value in self.input_history.time_seconds
        )

    @property
    def blocking_capabilities(self) -> tuple[str, ...]:
        """Keep required unavailable/unsupported providers visible."""
        return tuple(
            item.capability_id
            for item in self.capabilities
            if item.required
            and (
                item.support is not CapabilitySupport.SUPPORTED
                or item.availability is not CapabilityAvailability.AVAILABLE
            )
        )


def _require_clock_interval(requested: float, represented: float) -> None:
    """Allow 1e-9 relative clock error or 64 interval ULPs, never epoch ULPs.

    Large epoch spacing must not relax the physical replay interval. Consumers
    must additionally verify their effective native solver/time representation.
    """
    tolerance = max(1e-9 * requested, 64 * ulp(requested))
    if abs(represented - requested) > tolerance:
        raise ValueError("native clock precision distorts the requested interval")


def _native_state_hashes(
    experiment_id: str,
    model: NativeReplayModel,
    artifact: NativeStateArtifact,
    capabilities: tuple[CapabilityDeclaration, ...],
    history: InputHistory,
    policy: ReplayExecutionPolicy,
) -> NativeReplayIntegrity:
    components = IntegrityHashes(
        model_identity_sha256=_sha256((model, artifact.identity)),
        state_schema_sha256=_sha256((artifact.encoding, artifact.role)),
        capability_declarations_sha256=_sha256(capabilities),
        input_channel_schema_sha256=_sha256(history.channels),
        time_grid_sha256=_sha256(
            (artifact.clock, history.timebase_id, history.time_seconds)
        ),
        initial_state_sha256=_sha256(artifact),
        applied_input_sha256=_sha256(history),
        execution_policy_sha256=_sha256((policy, artifact.execution)),
    )
    return NativeReplayIntegrity(
        components,
        _sha256(
            (
                NATIVE_STATE_REPLAY_SCHEMA_VERSION,
                experiment_id,
                model,
                artifact,
                capabilities,
                history,
                policy,
            )
        ),
    )


def build_native_state_replay_envelope(
    experiment_id: str,
    model: NativeReplayModel,
    artifact: NativeStateArtifact,
    capabilities: tuple[CapabilityDeclaration, ...],
    input_history: InputHistory,
    policy: ReplayExecutionPolicy,
) -> NativeStateReplayEnvelope:
    """Bind native state metadata and saved inputs without loading opaque bytes."""
    frozen_capabilities = tuple(capabilities)
    integrity = _native_state_hashes(
        require_text(experiment_id, "experiment_id"),
        model,
        artifact,
        frozen_capabilities,
        input_history,
        policy,
    )
    return NativeStateReplayEnvelope(
        experiment_id,
        model,
        artifact,
        frozen_capabilities,
        input_history,
        policy,
        integrity,
    )
