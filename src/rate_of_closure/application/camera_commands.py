"""UI-neutral camera commands for Rate 3D engineering viewports.

The canonical frame is ``x=downrange, y=up, z=right``.  Renderers adapt the
declared view direction to their own projection APIs; no viewport owns a
second set of convention constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

Vector3 = tuple[float, float, float]

_ISOMETRIC_DIRECTION: Vector3 = (
    0.7071067811865476,
    -0.4082482904638631,
    -0.5773502691896258,
)
_ISOMETRIC_SCREEN_UP: Vector3 = (
    0.316227766016838,
    0.9128709291752768,
    -0.2581988897471612,
)
_VERTICAL_UP: Vector3 = (0.0, 1.0, 0.0)
_MIN_ZOOM = 0.25
_MAX_ZOOM = 8.0
_TRACKING_CLEARANCE_FRACTION = 0.16


class CameraCommandId(StrEnum):
    """Stable command identifiers shared by every camera adapter."""

    VIEW_ISOMETRIC = "camera.view.isometric"
    VIEW_FACE_ON = "camera.view.face_on"
    VIEW_DOWN_THE_LINE = "camera.view.down_the_line"
    VIEW_OVERHEAD = "camera.view.overhead"
    AUTO_FIT = "camera.auto_fit"
    RECENTER = "camera.recenter"
    TRACK_SUBJECT = "camera.track_subject"


class FaceOnSide(StrEnum):
    """Side from which the face-on camera observes the target line."""

    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class CameraPreset:
    """Canonical orientation independent of a rendering toolkit."""

    command_id: CameraCommandId
    view_direction: Vector3
    screen_up: Vector3

    def __post_init__(self) -> None:
        _finite_vector(self.view_direction, "view_direction")
        _finite_vector(self.screen_up, "screen_up")
        if not math.isclose(_norm(self.view_direction), 1.0, abs_tol=1e-12):
            raise ValueError("view_direction must be a unit vector")
        if not math.isclose(_norm(self.screen_up), 1.0, abs_tol=1e-12):
            raise ValueError("screen_up must be a unit vector")
        if abs(_dot(self.view_direction, self.screen_up)) > 1e-12:
            raise ValueError("screen_up must be perpendicular to view_direction")


@dataclass(frozen=True)
class CameraState:
    """Per-viewport target, scale, and follow state.

    Zoom is dimensionless and retained during tracking.  A manual orbit marks
    tracking suspended until :func:`recenter_camera` is invoked.
    """

    preset_id: CameraCommandId | None = CameraCommandId.VIEW_ISOMETRIC
    face_on_side: FaceOnSide = FaceOnSide.RIGHT
    target_m: Vector3 = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    tracking_enabled: bool = False
    tracking_suspended: bool = False
    auto_fit_enabled: bool = False

    def __post_init__(self) -> None:
        _finite_vector(self.target_m, "target_m")
        if not math.isfinite(self.zoom) or not _MIN_ZOOM <= self.zoom <= _MAX_ZOOM:
            raise ValueError(
                f"zoom must be finite and within [{_MIN_ZOOM}, {_MAX_ZOOM}]"
            )
        if self.tracking_suspended and not self.tracking_enabled:
            raise ValueError("tracking cannot be suspended while disabled")


def camera_preset(command_id: CameraCommandId, side: FaceOnSide) -> CameraPreset:
    """Return one exact canonical view, failing closed for non-view commands."""
    if command_id is CameraCommandId.VIEW_ISOMETRIC:
        return CameraPreset(command_id, _ISOMETRIC_DIRECTION, _ISOMETRIC_SCREEN_UP)
    if command_id is CameraCommandId.VIEW_FACE_ON:
        direction = (0.0, 0.0, -1.0 if side is FaceOnSide.RIGHT else 1.0)
        return CameraPreset(command_id, direction, _VERTICAL_UP)
    if command_id is CameraCommandId.VIEW_DOWN_THE_LINE:
        return CameraPreset(command_id, (1.0, 0.0, 0.0), _VERTICAL_UP)
    if command_id is CameraCommandId.VIEW_OVERHEAD:
        return CameraPreset(command_id, (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))
    raise ValueError(f"{command_id.value!r} is not a camera-view command")


def canvas_angles(preset: CameraPreset) -> tuple[float, float]:
    """Return dependency-free canvas yaw and pitch [rad]."""
    downrange, up, right = preset.view_direction
    return math.atan2(right, downrange), math.asin(max(-1.0, min(1.0, up)))


def matplotlib_angles(preset: CameraPreset) -> tuple[float, float]:
    """Return Matplotlib elevation/azimuth [deg] for display axes z,x,y."""
    downrange, up, right = preset.view_direction
    if preset.command_id is CameraCommandId.VIEW_OVERHEAD:
        return 90.0, -90.0
    camera_display = (-right, -downrange, -up)
    elevation = math.degrees(math.asin(camera_display[2]))
    azimuth = math.degrees(math.atan2(camera_display[1], camera_display[0]))
    return elevation, azimuth


def apply_camera_preset(state: CameraState, command_id: CameraCommandId) -> CameraState:
    """Select a view idempotently without altering target or zoom."""
    camera_preset(command_id, state.face_on_side)
    return replace(state, preset_id=command_id)


def set_tracking_enabled(
    state: CameraState, enabled: bool, subject_m: Vector3
) -> CameraState:
    """Enable tracking centered on the current subject, or disable it."""
    _finite_vector(subject_m, "subject_m")
    return replace(
        state,
        target_m=subject_m if enabled else state.target_m,
        tracking_enabled=enabled,
        tracking_suspended=False,
    )


def apply_manual_override(state: CameraState) -> CameraState:
    """Suspend an enabled tracker after an intentional manual orbit/pan."""
    if not state.tracking_enabled:
        return replace(state, preset_id=None)
    return replace(state, preset_id=None, tracking_suspended=True)


def recenter_camera(state: CameraState, subject_m: Vector3) -> CameraState:
    """Center on the subject and resume an enabled tracker in one action."""
    _finite_vector(subject_m, "subject_m")
    return replace(state, target_m=subject_m, tracking_suspended=False)


def update_tracking_target(
    state: CameraState, subject_m: Vector3, max_step_m: float
) -> CameraState:
    """Advance the focus by a bounded step while preserving zoom exactly."""
    _finite_vector(subject_m, "subject_m")
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("max_step_m must be finite and positive")
    if not state.tracking_enabled or state.tracking_suspended:
        return state
    delta: Vector3 = (
        subject_m[0] - state.target_m[0],
        subject_m[1] - state.target_m[1],
        subject_m[2] - state.target_m[2],
    )
    distance = _norm(delta)
    if distance <= 1e-12:
        return state
    fraction = min(1.0, max_step_m / distance)
    target: Vector3 = (
        state.target_m[0] + fraction * delta[0],
        state.target_m[1] + fraction * delta[1],
        state.target_m[2] + fraction * delta[2],
    )
    return replace(state, target_m=target)


def safe_tracking_zoom(
    requested_zoom: float,
    subject_radius_m: float,
    base_half_extent_m: float,
) -> float:
    """Preserve safe zoom or reduce it to retain the declared clearance."""
    values = (requested_zoom, subject_radius_m, base_half_extent_m)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("zoom, subject radius, and base half extent must be positive")
    maximum = (
        base_half_extent_m * (1.0 - _TRACKING_CLEARANCE_FRACTION) / subject_radius_m
    )
    return max(_MIN_ZOOM, min(requested_zoom, maximum, _MAX_ZOOM))


def _finite_vector(vector: Vector3, name: str) -> None:
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain three finite values")


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


__all__ = [
    "CameraCommandId",
    "CameraPreset",
    "CameraState",
    "FaceOnSide",
    "apply_camera_preset",
    "apply_manual_override",
    "camera_preset",
    "canvas_angles",
    "matplotlib_angles",
    "recenter_camera",
    "safe_tracking_zoom",
    "set_tracking_enabled",
    "update_tracking_target",
]
