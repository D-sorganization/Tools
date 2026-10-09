"""Immutable opaque restart artifacts; metadata is not native qualification.

Native consumers must additionally verify decoded class, clock and model
compatibility. This module never deserializes or executes artifact contents.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_semver,
    require_text,
)
from .experiment_contracts import _require_sha256


def _normalize_fields(
    instance: object, names: tuple[str, ...], validator: Callable[[str, str], str]
) -> None:
    for name in names:
        object.__setattr__(instance, name, validator(getattr(instance, name), name))


class NativeStateRole(StrEnum):
    """Declared artifact scope, never inferred from a numeric sidecar."""

    COMPLETE_NATIVE_RESTART = "complete_native_restart"
    OBSERVABLES_ONLY = "observables_only"


@dataclass(frozen=True, slots=True)
class NativeStateEncoding:
    """Versioned native representation and expected decoded class."""

    format_id: str
    format_version: str
    native_class: str

    def __post_init__(self) -> None:
        _normalize_fields(self, ("format_id", "native_class"), require_text)
        _normalize_fields(self, ("format_version",), require_semver)


@dataclass(frozen=True, slots=True)
class NativeStateIdentity:
    """Exact producing runtime, provider and source/loaded-model identities."""

    engine_id: str
    runtime_id: str
    provider_sha256: str
    source_model_sha256: str
    loaded_model_sha256: str

    def __post_init__(self) -> None:
        _normalize_fields(self, ("engine_id", "runtime_id"), require_text)
        _normalize_fields(
            self,
            ("provider_sha256", "source_model_sha256", "loaded_model_sha256"),
            _require_sha256,
        )


@dataclass(frozen=True, slots=True)
class NativeStateClock:
    """Unmodified native start/snapshot times in an explicit simulation clock."""

    start_time_seconds: float
    snapshot_time_seconds: float
    clock_id: str

    def __post_init__(self) -> None:
        for name in ("start_time_seconds", "snapshot_time_seconds"):
            object.__setattr__(self, name, require_finite(getattr(self, name), name))
        if self.snapshot_time_seconds < self.start_time_seconds:
            raise ValueError("snapshot time must not precede native start time")
        _normalize_fields(self, ("clock_id",), require_text)


@dataclass(frozen=True, slots=True)
class NativeStateExecution:
    """Solver and effective configuration/checksum policy, including hidden state."""

    solver_id: str
    solver_version: str
    policy_id: str
    policy_version: str
    effective_configuration_sha256: str
    compatibility_sha256: str

    def __post_init__(self) -> None:
        _normalize_fields(self, ("solver_id", "policy_id"), require_text)
        _normalize_fields(self, ("solver_version", "policy_version"), require_semver)
        _normalize_fields(
            self,
            ("effective_configuration_sha256", "compatibility_sha256"),
            _require_sha256,
        )


@dataclass(frozen=True, slots=True)
class NativeStateArtifact:
    """Content-addressed opaque state declaration, separate from observables.

    Artifact byte identity does not assert canonical semantic-state encoding or
    source-to-binary equivalence. Effective configuration must include actual
    solver settings and external dependency/checksum policy, not UI defaults.
    """

    encoding: NativeStateEncoding
    identity: NativeStateIdentity
    clock: NativeStateClock
    execution: NativeStateExecution
    payload_sha256: str
    byte_size: int
    role: NativeStateRole

    def __post_init__(self) -> None:
        for name, expected in (
            ("encoding", NativeStateEncoding),
            ("identity", NativeStateIdentity),
            ("clock", NativeStateClock),
            ("execution", NativeStateExecution),
            ("role", NativeStateRole),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        _normalize_fields(self, ("payload_sha256",), _require_sha256)
        size = require_nonnegative_integer(self.byte_size, "byte_size")
        if size == 0:
            raise ValueError("native artifact byte_size must be positive")


@dataclass(frozen=True, slots=True)
class OwnedNativeStateArtifact:
    """Verified owned bytes; consumers load these bytes rather than reopen a path."""

    descriptor: NativeStateArtifact
    payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, NativeStateArtifact):
            raise TypeError("descriptor must be NativeStateArtifact")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must contain bytes")
        frozen = bytes(self.payload)
        if self.descriptor.role is not NativeStateRole.COMPLETE_NATIVE_RESTART:
            raise ValueError(
                "complete native restart required; observables are insufficient"
            )
        if len(frozen) != self.descriptor.byte_size:
            raise ValueError("native payload size mismatch")
        if sha256(frozen).hexdigest() != self.descriptor.payload_sha256:
            raise ValueError("native payload digest mismatch")
        object.__setattr__(self, "payload", frozen)


def freeze_native_state_artifact(
    descriptor: NativeStateArtifact, payload: bytes | bytearray | memoryview
) -> OwnedNativeStateArtifact:
    """Verify an owned immutable snapshot before any native decoding boundary."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must contain bytes")
    return OwnedNativeStateArtifact(descriptor, bytes(payload))
