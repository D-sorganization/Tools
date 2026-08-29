"""Deterministic serialization for markerless-mocap contracts."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from .devices import CameraIdentity
from .enums import ClockKind, SessionState
from .geometry import CoordinateFrame
from .session import (
    MOCAP_SESSION_SCHEMA_VERSION,
    MethodDescriptor,
    MocapSessionManifest,
    RecordingPolicy,
)
from .timebase import ClockDomain


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    return value


def dumps_canonical(manifest: MocapSessionManifest) -> str:
    """Serialize a session as sorted, stable UTF-8 JSON with a final newline."""
    if not isinstance(manifest, MocapSessionManifest):
        raise TypeError("manifest must be a MocapSessionManifest")
    return (
        json.dumps(
            _to_primitive(manifest),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _require_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_exact_fields(
    value: dict[str, Any], expected: set[str], field_name: str
) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise ValueError(f"{field_name} contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{field_name} is missing fields: {sorted(missing)}")


def _camera(value: Any) -> CameraIdentity:
    payload = _require_object(value, "camera")
    expected = {
        "provider_id",
        "device_id",
        "transport",
        "vendor",
        "model",
        "serial_number",
        "firmware_version",
    }
    _require_exact_fields(payload, expected, "camera")
    return CameraIdentity(**payload)


def _clock(value: Any) -> ClockDomain:
    payload = _require_object(value, "clock")
    expected = {"clock_id", "kind", "tick_period_seconds", "monotonic"}
    _require_exact_fields(payload, expected, "clock")
    return ClockDomain(
        clock_id=payload["clock_id"],
        kind=ClockKind(payload["kind"]),
        tick_period_seconds=payload["tick_period_seconds"],
        monotonic=payload["monotonic"],
    )


def _coordinate_frame(value: Any) -> CoordinateFrame:
    payload = _require_object(value, "world_frame")
    expected = {"frame_id", "handedness", "x_axis", "y_axis", "z_axis", "length_unit"}
    _require_exact_fields(payload, expected, "world_frame")
    return CoordinateFrame(**payload)


def _method(value: Any) -> MethodDescriptor:
    payload = _require_object(value, "method")
    expected = {
        "method_id",
        "version",
        "implementation",
        "license_spdx",
        "artifact_sha256",
    }
    _require_exact_fields(payload, expected, "method")
    return MethodDescriptor(**payload)


def _recording_policy(value: Any) -> RecordingPolicy:
    payload = _require_object(value, "recording_policy")
    expected = {"consent_recorded", "raw_video_retained", "retention_days", "no_store"}
    _require_exact_fields(payload, expected, "recording_policy")
    return RecordingPolicy(**payload)


def load_session_manifest(text: str) -> MocapSessionManifest:
    """Parse strict session JSON and reject unknown or incompatible fields."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        payload = _require_object(json.loads(text), "manifest")
    except json.JSONDecodeError as exc:
        raise ValueError("manifest must contain valid JSON") from exc
    expected = {
        "schema_version",
        "session_id",
        "created_at_utc",
        "state",
        "world_frame",
        "cameras",
        "clocks",
        "methods",
        "recording_policy",
        "calibration_ids",
        "warnings",
    }
    _require_exact_fields(payload, expected, "manifest")
    if payload["schema_version"] != MOCAP_SESSION_SCHEMA_VERSION:
        received_version = payload["schema_version"]
        raise ValueError(
            f"schema_version must be {MOCAP_SESSION_SCHEMA_VERSION!r}; "
            f"got {received_version!r}"
        )
    return MocapSessionManifest(
        session_id=payload["session_id"],
        created_at_utc=payload["created_at_utc"],
        state=SessionState(payload["state"]),
        world_frame=_coordinate_frame(payload["world_frame"]),
        cameras=tuple(_camera(value) for value in payload["cameras"]),
        clocks=tuple(_clock(value) for value in payload["clocks"]),
        methods=tuple(_method(value) for value in payload["methods"]),
        recording_policy=_recording_policy(payload["recording_policy"]),
        calibration_ids=tuple(payload["calibration_ids"]),
        warnings=tuple(payload["warnings"]),
    )


__all__: list[str] = []
