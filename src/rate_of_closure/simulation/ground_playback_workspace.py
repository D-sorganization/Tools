"""Strict persistence and evidence exports for ground-result playback."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, cast

from rate_of_closure.simulation.ground_playback import (
    DEFAULT_IMPORT_MAX_BYTES,
    DEFAULT_IMPORT_MAX_POINTS,
    GroundPlaybackTimeline,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.ground import GroundSimulationResult
from shared.python.swing_sim.ground.contract_wire import record_from_dict
from shared.python.swing_sim.ground.strict_json import strict_json_object

GROUND_PLAYBACK_WORKSPACE_SCHEMA = "rate-of-closure-ground-playback-workspace/v1"
SUPPORTED_PLAYBACK_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
MIN_CAMERA_ZOOM = 0.4
MAX_CAMERA_ZOOM = 4.0


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True)
class GroundPlaybackState:
    """Portable paused playback state; active timers are never persisted."""

    time_s: float
    speed: float
    loop: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite_number(self.time_s, "time_s"))
        normalized_speed = _finite_number(self.speed, "speed")
        if normalized_speed not in SUPPORTED_PLAYBACK_SPEEDS:
            raise ValueError("speed must be a supported playback speed")
        object.__setattr__(self, "speed", normalized_speed)
        if type(self.loop) is not bool:
            raise TypeError("loop must be a boolean")


@dataclass(frozen=True)
class GroundPlaybackViewState:
    """UI-neutral orbit camera orientation and physical-scale zoom."""

    yaw_deg: float
    pitch_deg: float
    zoom: float

    def __post_init__(self) -> None:
        yaw = _finite_number(self.yaw_deg, "yaw_deg")
        pitch = _finite_number(self.pitch_deg, "pitch_deg")
        zoom = _finite_number(self.zoom, "zoom")
        if not -180.0 <= yaw <= 180.0:
            raise ValueError("yaw_deg must lie within [-180, 180]")
        if not -90.0 <= pitch <= 90.0:
            raise ValueError("pitch_deg must lie within [-90, 90]")
        if not MIN_CAMERA_ZOOM <= zoom <= MAX_CAMERA_ZOOM:
            raise ValueError("zoom must lie within [0.4, 4.0]")
        object.__setattr__(self, "yaw_deg", yaw)
        object.__setattr__(self, "pitch_deg", pitch)
        object.__setattr__(self, "zoom", zoom)


@dataclass(frozen=True)
class GroundPlaybackWorkspace:
    """One validated result plus portable paused playback and camera state."""

    result: GroundSimulationResult
    playback: GroundPlaybackState
    view: GroundPlaybackViewState
    schema_version: str = GROUND_PLAYBACK_WORKSPACE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.result) is not GroundSimulationResult:
            raise TypeError("result must use the exact GroundSimulationResult type")
        if type(self.playback) is not GroundPlaybackState:
            raise TypeError("playback must use the exact GroundPlaybackState type")
        if type(self.view) is not GroundPlaybackViewState:
            raise TypeError("view must use the exact GroundPlaybackViewState type")
        if self.schema_version != GROUND_PLAYBACK_WORKSPACE_SCHEMA:
            raise ValueError("unsupported ground playback workspace schema_version")
        timeline = GroundPlaybackTimeline(self.result)
        if not timeline.start_time_s <= self.playback.time_s <= timeline.end_time_s:
            raise ValueError("playback time_s must lie within the result timeline")


def _exact_fields(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields do not match v1 schema")


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _workspace_dict(workspace: GroundPlaybackWorkspace) -> dict[str, Any]:
    return {
        "playback": {
            "loop": workspace.playback.loop,
            "speed": workspace.playback.speed,
            "time_s": workspace.playback.time_s,
        },
        "result": workspace.result.to_dict(),
        "schema_version": workspace.schema_version,
        "view": {
            "pitch_deg": workspace.view.pitch_deg,
            "yaw_deg": workspace.view.yaw_deg,
            "zoom": workspace.view.zoom,
        },
    }


def ground_workspace_to_json(workspace: GroundPlaybackWorkspace) -> str:
    """Return deterministic compact canonical JSON for one exact workspace."""
    if type(workspace) is not GroundPlaybackWorkspace:
        raise TypeError("workspace must use the exact GroundPlaybackWorkspace type")
    return str(canonical_numeric_json(_workspace_dict(workspace)))


def ground_workspace_from_json(
    text: str,
    *,
    max_bytes: int = DEFAULT_IMPORT_MAX_BYTES,
    max_points: int = DEFAULT_IMPORT_MAX_POINTS,
) -> GroundPlaybackWorkspace:
    """Parse one strict v1 workspace, rejecting duplicate and unknown fields."""
    if type(text) is not str:
        raise TypeError("ground playback workspace JSON must be text")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_points) is not int or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("ground playback workspace JSON exceeds the import size limit")
    payload = strict_json_object(text)
    _exact_fields(
        payload, {"schema_version", "result", "playback", "view"}, "workspace"
    )
    playback = _object(payload["playback"], "playback")
    view = _object(payload["view"], "view")
    result_payload = _object(payload["result"], "result")
    _exact_fields(playback, {"time_s", "speed", "loop"}, "playback")
    _exact_fields(view, {"yaw_deg", "pitch_deg", "zoom"}, "view")
    result = cast(
        GroundSimulationResult,
        record_from_dict(GroundSimulationResult, result_payload),
    )
    if len(result.trajectory) > max_points:
        raise ValueError("ground result trajectory exceeds the import point limit")
    return GroundPlaybackWorkspace(
        result=result,
        playback=GroundPlaybackState(
            playback["time_s"], playback["speed"], playback["loop"]
        ),
        view=GroundPlaybackViewState(view["yaw_deg"], view["pitch_deg"], view["zoom"]),
        schema_version=payload["schema_version"],
    )


def ground_result_json(result: GroundSimulationResult) -> str:
    """Return the lossless canonical strict result document."""
    if type(result) is not GroundSimulationResult:
        raise TypeError("result must use the exact GroundSimulationResult type")
    return cast(str, result.to_json())


def _number(value: int | float) -> str:
    return str(canonical_numeric_json(value))


def _vector_values(vector: tuple[float, float, float]) -> list[str]:
    return [_number(component) for component in vector]


def _csv_text(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def ground_trajectory_csv(result: GroundSimulationResult) -> str:
    """Return every trajectory field as deterministic UTF-8-ready CSV text."""
    if type(result) is not GroundSimulationResult:
        raise TypeError("result must use the exact GroundSimulationResult type")
    headers = (
        "sample_index",
        "time_s",
        "phase",
        "frame",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "angular_velocity_x_rad_s",
        "angular_velocity_y_rad_s",
        "angular_velocity_z_rad_s",
    )
    rows = [
        [
            str(index),
            _number(point.time_s),
            point.phase.value,
            point.frame,
            *_vector_values(point.position_m),
            *_vector_values(point.velocity_m_s),
            *_vector_values(point.angular_velocity_rad_s),
        ]
        for index, point in enumerate(result.trajectory)
    ]
    return _csv_text(headers, rows)


def ground_event_csv(result: GroundSimulationResult) -> str:
    """Return every event field as deterministic UTF-8-ready CSV text."""
    if type(result) is not GroundSimulationResult:
        raise TypeError("result must use the exact GroundSimulationResult type")
    headers = (
        "sequence",
        "event_type",
        "time_s",
        "frame",
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_before_x_m_s",
        "velocity_before_y_m_s",
        "velocity_before_z_m_s",
        "velocity_after_x_m_s",
        "velocity_after_y_m_s",
        "velocity_after_z_m_s",
        "angular_velocity_before_x_rad_s",
        "angular_velocity_before_y_rad_s",
        "angular_velocity_before_z_rad_s",
        "angular_velocity_after_x_rad_s",
        "angular_velocity_after_y_rad_s",
        "angular_velocity_after_z_rad_s",
    )
    rows = [
        [
            str(event.sequence),
            event.event_type.value,
            _number(event.time_s),
            event.frame,
            *_vector_values(event.position_m),
            *_vector_values(event.velocity_before_m_s),
            *_vector_values(event.velocity_after_m_s),
            *_vector_values(event.angular_velocity_before_rad_s),
            *_vector_values(event.angular_velocity_after_rad_s),
        ]
        for event in result.events
    ]
    return _csv_text(headers, rows)


__all__ = [
    "GROUND_PLAYBACK_WORKSPACE_SCHEMA",
    "MAX_CAMERA_ZOOM",
    "MIN_CAMERA_ZOOM",
    "SUPPORTED_PLAYBACK_SPEEDS",
    "GroundPlaybackState",
    "GroundPlaybackViewState",
    "GroundPlaybackWorkspace",
    "ground_event_csv",
    "ground_result_json",
    "ground_trajectory_csv",
    "ground_workspace_from_json",
    "ground_workspace_to_json",
]
