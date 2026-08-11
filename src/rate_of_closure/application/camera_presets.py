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


class FaceOnSide(StrEnum):
    """Physical side from which Face On observes the target line."""

    RIGHT = "right"
    LEFT = "left"


CAMERA_COMMAND_IDS: tuple[str, ...] = tuple(
    view.value for view in CameraViewId
) + tuple(command.value for command in CameraCommandId)

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

    def __post_init__(self) -> None:
        if not isinstance(self.preset_id, CameraViewId):
            raise ValueError("unknown camera view")
        if not isinstance(self.face_on_side, FaceOnSide):
            raise ValueError("unknown face-on side")
        _finite_vector(self.target_m, "target_m")
        if not _finite_number(self.zoom) or not MIN_ZOOM <= self.zoom <= MAX_ZOOM:
            raise ValueError(f"zoom must be finite and within [{MIN_ZOOM}, {MAX_ZOOM}]")


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
    "CAMERA_COMMAND_IDS",
    "MAX_ZOOM",
    "MIN_ZOOM",
    "CameraCommandId",
    "CameraPreset",
    "CameraState",
    "CameraViewId",
    "FaceOnSide",
    "apply_camera_view",
    "auto_fit_camera",
    "camera_preset",
    "canvas_angles",
    "matplotlib_angles",
    "set_face_on_side",
    "with_camera_zoom",
]
