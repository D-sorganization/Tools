"""Canonical camera presets for the matched Rate clubhead viewports.

The engineering frame is always ``x=downrange, y=up, z=right``.  This
module owns orientation and scale invariants; renderers only adapt the exact
vectors to their projection API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

Vector3 = tuple[float, float, float]
MIN_ZOOM = 0.3
MAX_ZOOM = 4.0
AUTO_FIT_CLEARANCE_FRACTION = 0.16
TRACKING_MAX_TARGET_STEP_M = 0.05


class CameraViewId(StrEnum):
    """Stable identifiers for canonical view orientations."""

    ISOMETRIC = "camera.view.isometric"
    FACE_ON = "camera.view.face_on"
    DOWN_THE_LINE = "camera.view.down_the_line"
    OVERHEAD = "camera.view.overhead"


class CameraCommandId(StrEnum):
    """Stable identifiers for non-preset camera actions."""

    RESET_VIEW = "camera.reset_view"
    AUTO_FIT = "camera.auto_fit"
    TRACK_CLUBHEAD = "camera.track_clubhead"
    RECENTER = "camera.recenter"


class CameraTrackingStateId(StrEnum):
    """Stable visible states for the opt-in clubhead tracker."""

    OFF = "camera.tracking.off"
    ACTIVE = "camera.tracking.active"
    SUSPENDED = "camera.tracking.suspended"


class FaceOnSide(StrEnum):
    """Physical side from which Face On observes the target line."""

    RIGHT = "right"
    LEFT = "left"


CAMERA_TRACKING_COMMAND_IDS: tuple[str, ...] = (
    CameraCommandId.TRACK_CLUBHEAD.value,
    CameraCommandId.RECENTER.value,
)
CAMERA_PRESET_COMMAND_IDS: tuple[str, ...] = tuple(
    view.value for view in CameraViewId
) + (
    CameraCommandId.RESET_VIEW.value,
    CameraCommandId.AUTO_FIT.value,
)
CAMERA_COMMAND_IDS: tuple[str, ...] = (
    *CAMERA_PRESET_COMMAND_IDS,
    *CAMERA_TRACKING_COMMAND_IDS,
)
CAMERA_CONTROL_IDS: tuple[str, ...] = ("camera.auto_fit_fallback",)
CAMERA_TRACKING_STATE_IDS: tuple[str, ...] = tuple(
    state.value for state in CameraTrackingStateId
)

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


@dataclass(frozen=True, slots=True)
class CameraPreset:
    """One validated toolkit-independent orientation."""

    command_id: CameraViewId
    view_direction: Vector3
    screen_up: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.command_id, CameraViewId):
            raise ValueError("unknown camera view")
        _finite_vector(self.view_direction, "view_direction")
        _finite_vector(self.screen_up, "screen_up")
        if not math.isclose(
            _norm(self.view_direction), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("view_direction must be a unit vector")
        if not math.isclose(_norm(self.screen_up), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("screen_up must be a unit vector")
        if abs(_dot(self.view_direction, self.screen_up)) > 1e-12:
            raise ValueError("screen_up must be perpendicular to view_direction")


@dataclass(frozen=True, slots=True)
class CameraState:
    """Isolated orientation, target, scale, and explicit Face-On side."""

    preset_id: CameraViewId = CameraViewId.ISOMETRIC
    face_on_side: FaceOnSide = FaceOnSide.RIGHT
    target_m: Vector3 = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    tracking_enabled: bool = False
    tracking_suspended: bool = False
    auto_fit_fallback_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.preset_id, CameraViewId):
            raise ValueError("unknown camera view")
        if not isinstance(self.face_on_side, FaceOnSide):
            raise ValueError("unknown face-on side")
        _finite_vector(self.target_m, "target_m")
        if not _finite_number(self.zoom) or not MIN_ZOOM <= self.zoom <= MAX_ZOOM:
            raise ValueError(f"zoom must be finite and within [{MIN_ZOOM}, {MAX_ZOOM}]")
        for value, name in (
            (self.tracking_enabled, "tracking_enabled"),
            (self.tracking_suspended, "tracking_suspended"),
            (self.auto_fit_fallback_enabled, "auto_fit_fallback_enabled"),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a Boolean")
        if self.tracking_suspended and not self.tracking_enabled:
            raise ValueError("tracking cannot be suspended while disabled")


def camera_preset(
    command_id: CameraViewId | str, side: FaceOnSide | str
) -> CameraPreset:
    """Return an exact canonical preset and fail closed for unknown inputs."""
    view_id = _camera_view_id(command_id)
    face_side = _face_on_side(side)
    if view_id is CameraViewId.ISOMETRIC:
        return CameraPreset(view_id, _ISOMETRIC_DIRECTION, _ISOMETRIC_SCREEN_UP)
    if view_id is CameraViewId.FACE_ON:
        direction = (0.0, 0.0, -1.0 if face_side is FaceOnSide.RIGHT else 1.0)
        return CameraPreset(view_id, direction, _VERTICAL_UP)
    if view_id is CameraViewId.DOWN_THE_LINE:
        return CameraPreset(view_id, (1.0, 0.0, 0.0), _VERTICAL_UP)
    return CameraPreset(view_id, (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))


def canvas_angles(preset: CameraPreset) -> tuple[float, float]:
    """Return dependency-free canvas yaw and pitch in radians."""
    downrange, up, right = preset.view_direction
    return math.atan2(right, downrange), math.asin(max(-1.0, min(1.0, up)))


def matplotlib_angles(preset: CameraPreset) -> tuple[float, float]:
    """Return Matplotlib elevation/azimuth for display axes ``z,x,y``."""
    downrange, up, right = preset.view_direction
    if preset.command_id is CameraViewId.OVERHEAD:
        return 90.0, -90.0
    camera_display = (-right, -downrange, -up)
    return (
        math.degrees(math.asin(camera_display[2])),
        math.degrees(math.atan2(camera_display[1], camera_display[0])),
    )


def apply_camera_view(
    state: CameraState, command_id: CameraViewId | str
) -> CameraState:
    """Select a view idempotently without changing target or scale."""
    view_id = _camera_view_id(command_id)
    camera_preset(view_id, state.face_on_side)
    return replace(state, preset_id=view_id)


def set_face_on_side(state: CameraState, side: FaceOnSide | str) -> CameraState:
    """Select a deliberate lateral side without inferring handedness."""
    return replace(state, face_on_side=_face_on_side(side))


def with_camera_zoom(state: CameraState, zoom: float) -> CameraState:
    """Return ``state`` with finite zoom clamped to the supported range."""
    if not _finite_number(zoom):
        raise ValueError("zoom must be finite")
    return replace(state, zoom=max(MIN_ZOOM, min(MAX_ZOOM, zoom)))


def auto_fit_camera(
    state: CameraState,
    subject_radius_m: float,
    base_half_extent_m: float,
    clearance_fraction: float = AUTO_FIT_CLEARANCE_FRACTION,
) -> CameraState:
    """Fit a bounding sphere while changing only dimensionless zoom."""
    if not all(
        _finite_number(value) and value > 0.0
        for value in (subject_radius_m, base_half_extent_m)
    ):
        raise ValueError(
            "subject radius and base half extent must be finite and positive"
        )
    if not _finite_number(clearance_fraction) or not 0.0 <= clearance_fraction < 1.0:
        raise ValueError("clearance_fraction must be finite and within [0, 1)")
    fitted = base_half_extent_m * (1.0 - clearance_fraction) / subject_radius_m
    return with_camera_zoom(state, fitted)


def set_camera_tracking(
    state: CameraState, enabled: bool, subject_m: Vector3
) -> CameraState:
    """Enable centered tracking, or disable it without moving the target."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a Boolean")
    _finite_vector(subject_m, "subject_m")
    return replace(
        state,
        target_m=subject_m if enabled else state.target_m,
        tracking_enabled=enabled,
        tracking_suspended=False,
    )


def set_auto_fit_fallback(state: CameraState, enabled: bool) -> CameraState:
    """Set the explicit reduction-only clearance fallback."""
    if type(enabled) is not bool:
        raise ValueError("enabled must be a Boolean")
    return replace(state, auto_fit_fallback_enabled=enabled)


def apply_manual_camera_override(
    state: CameraState, target_m: Vector3 | None = None
) -> CameraState:
    """Retain a manual target and suspend active tracking deterministically."""
    manual_target = state.target_m if target_m is None else target_m
    _finite_vector(manual_target, "target_m")
    return replace(
        state,
        target_m=manual_target,
        tracking_suspended=state.tracking_enabled,
    )


def recenter_camera(state: CameraState, subject_m: Vector3) -> CameraState:
    """Center exactly on the subject and resume an enabled tracker."""
    _finite_vector(subject_m, "subject_m")
    return replace(state, target_m=subject_m, tracking_suspended=False)


def update_tracking_target(
    state: CameraState,
    subject_m: Vector3,
    max_step_m: float = TRACKING_MAX_TARGET_STEP_M,
) -> CameraState:
    """Advance the target by at most ``max_step_m`` while preserving zoom."""
    _finite_vector(subject_m, "subject_m")
    if not _finite_number(max_step_m) or max_step_m <= 0.0:
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


def enforce_tracking_clearance(
    state: CameraState,
    subject_radius_m: float,
    base_half_extent_m: float,
    clearance_fraction: float = AUTO_FIT_CLEARANCE_FRACTION,
) -> CameraState:
    """Reduce unsafe zoom only when the user enabled the tracking fallback."""
    fitted = auto_fit_camera(
        state, subject_radius_m, base_half_extent_m, clearance_fraction
    )
    if not state.auto_fit_fallback_enabled or state.zoom <= fitted.zoom:
        return state
    return fitted


def tracking_state_id(state: CameraState) -> CameraTrackingStateId:
    """Return the stable visible state represented by ``state``."""
    if not state.tracking_enabled:
        return CameraTrackingStateId.OFF
    if state.tracking_suspended:
        return CameraTrackingStateId.SUSPENDED
    return CameraTrackingStateId.ACTIVE


def _camera_view_id(value: CameraViewId | str) -> CameraViewId:
    try:
        return value if isinstance(value, CameraViewId) else CameraViewId(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown camera view: {value!r}") from exc


def _face_on_side(value: FaceOnSide | str) -> FaceOnSide:
    try:
        return value if isinstance(value, FaceOnSide) else FaceOnSide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown face-on side: {value!r}") from exc


def _finite_vector(vector: Vector3, name: str) -> None:
    if len(vector) != 3 or not all(_finite_number(value) for value in vector):
        raise ValueError(f"{name} must contain three finite values")


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(left * right for left, right in zip(first, second, strict=True))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


__all__ = [
    "AUTO_FIT_CLEARANCE_FRACTION",
    "CAMERA_CONTROL_IDS",
    "CAMERA_COMMAND_IDS",
    "CAMERA_PRESET_COMMAND_IDS",
    "CAMERA_TRACKING_COMMAND_IDS",
    "CAMERA_TRACKING_STATE_IDS",
    "MAX_ZOOM",
    "MIN_ZOOM",
    "CameraCommandId",
    "CameraPreset",
    "CameraState",
    "CameraTrackingStateId",
    "CameraViewId",
    "FaceOnSide",
    "apply_camera_view",
    "apply_manual_camera_override",
    "auto_fit_camera",
    "camera_preset",
    "canvas_angles",
    "enforce_tracking_clearance",
    "matplotlib_angles",
    "recenter_camera",
    "set_auto_fit_fallback",
    "set_camera_tracking",
    "set_face_on_side",
    "tracking_state_id",
    "TRACKING_MAX_TARGET_STEP_M",
    "update_tracking_target",
    "with_camera_zoom",
]
