"""Shared 3D camera controls, preferences, and viewport integration."""

from __future__ import annotations

from .controls import CameraControls, CameraViewport, CameraViewportMixin
from .models import (
    CameraCommandId,
    CameraPreset,
    CameraState,
    FaceOnSide,
    Vector3,
    apply_camera_preset,
    apply_manual_override,
    camera_preset,
    canvas_angles,
    matplotlib_angles,
    moving_subject_camera_state,
    recenter_camera,
    safe_tracking_zoom,
    set_tracking_enabled,
    update_tracking_target,
)
from .preferences import (
    CAMERA_PREFERENCES_FORMAT,
    CAMERA_VIEWPORT_IDS,
    CameraPreference,
    CameraPreferences,
    apply_camera_preference,
    default_camera_preferences,
    preference_from_camera_state,
)

__all__ = [
    "CAMERA_PREFERENCES_FORMAT",
    "CAMERA_VIEWPORT_IDS",
    "CameraCommandId",
    "CameraControls",
    "CameraPreference",
    "CameraPreferences",
    "CameraPreset",
    "CameraState",
    "CameraViewport",
    "CameraViewportMixin",
    "FaceOnSide",
    "Vector3",
    "apply_camera_preference",
    "apply_camera_preset",
    "apply_manual_override",
    "camera_preset",
    "canvas_angles",
    "default_camera_preferences",
    "matplotlib_angles",
    "moving_subject_camera_state",
    "preference_from_camera_state",
    "recenter_camera",
    "safe_tracking_zoom",
    "set_tracking_enabled",
    "update_tracking_target",
]
