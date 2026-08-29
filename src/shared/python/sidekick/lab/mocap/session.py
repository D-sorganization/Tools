"""Versioned markerless-mocap session and provenance records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ._validation import (
    require_nonnegative_integer,
    require_semver,
    require_text,
    require_unique_text,
)
from .devices import CameraIdentity
from .enums import SessionState
from .geometry import CoordinateFrame
from .timebase import ClockDomain

MOCAP_SESSION_SCHEMA_VERSION = "mocap-session/1.0.0"


def _require_utc_timestamp(value: str) -> str:
    normalized = require_text(value, "created_at_utc")
    try:
        timestamp = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at_utc must be an ISO 8601 timestamp") from exc
    utc_offset = timestamp.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0.0:
        raise ValueError("created_at_utc must be UTC")
    return normalized


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    """Version and license provenance for a capture or processing method."""

    method_id: str
    version: str
    implementation: str
    license_spdx: str
    artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", require_text(self.method_id, "method_id"))
        object.__setattr__(self, "version", require_semver(self.version, "version"))
        object.__setattr__(
            self, "implementation", require_text(self.implementation, "implementation")
        )
        object.__setattr__(
            self, "license_spdx", require_text(self.license_spdx, "license_spdx")
        )
        if self.artifact_sha256 is not None:
            digest = require_text(self.artifact_sha256, "artifact_sha256")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "artifact_sha256 must be 64 lowercase hexadecimal characters"
                )
            object.__setattr__(self, "artifact_sha256", digest)


@dataclass(frozen=True, slots=True)
class RecordingPolicy:
    """Consent, retention, and no-store policy captured with a session."""

    consent_recorded: bool
    raw_video_retained: bool
    retention_days: int
    no_store: bool

    def __post_init__(self) -> None:
        for name in ("consent_recorded", "raw_video_retained", "no_store"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "retention_days",
            require_nonnegative_integer(self.retention_days, "retention_days"),
        )
        if self.no_store and (self.raw_video_retained or self.retention_days != 0):
            raise ValueError(
                "no_store requires no raw video retention and zero retention_days"
            )
        if self.raw_video_retained and not self.consent_recorded:
            raise ValueError("raw video retention requires recorded consent")


@dataclass(frozen=True, slots=True)
class MocapSessionManifest:
    """Strict, deterministic session-level identity and provenance manifest."""

    session_id: str
    created_at_utc: str
    state: SessionState
    world_frame: CoordinateFrame
    cameras: tuple[CameraIdentity, ...]
    clocks: tuple[ClockDomain, ...]
    methods: tuple[MethodDescriptor, ...]
    recording_policy: RecordingPolicy
    calibration_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = field(init=False, default=MOCAP_SESSION_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_id", require_text(self.session_id, "session_id")
        )
        object.__setattr__(
            self, "created_at_utc", _require_utc_timestamp(self.created_at_utc)
        )
        if not isinstance(self.state, SessionState):
            raise TypeError("state must be a SessionState")
        if not isinstance(self.world_frame, CoordinateFrame):
            raise TypeError("world_frame must be a CoordinateFrame")
        self._require_unique_records()
        calibrations = require_unique_text(self.calibration_ids, "calibration_ids")
        warnings = require_unique_text(self.warnings, "warnings")
        object.__setattr__(self, "calibration_ids", calibrations)
        object.__setattr__(self, "warnings", warnings)
        if self.state is SessionState.FINALIZED:
            self._validate_finalized()

    def _require_unique_records(self) -> None:
        camera_keys = tuple(camera.stable_key for camera in self.cameras)
        clock_ids = tuple(clock.clock_id for clock in self.clocks)
        method_ids = tuple(method.method_id for method in self.methods)
        for values, field_name in (
            (camera_keys, "cameras"),
            (clock_ids, "clocks"),
            (method_ids, "methods"),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must have unique identities")

    def _validate_finalized(self) -> None:
        if not self.recording_policy.consent_recorded:
            raise ValueError("finalized sessions require recorded consent")
        for values, field_name in (
            (self.cameras, "cameras"),
            (self.clocks, "clocks"),
            (self.methods, "methods"),
            (self.calibration_ids, "calibration_ids"),
        ):
            if not values:
                raise ValueError(f"finalized sessions require {field_name}")


__all__: list[str] = []
