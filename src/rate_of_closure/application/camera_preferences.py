"""Strict durable camera preferences shared by every 3D viewport.

Only deliberate user preferences are serialized.  A moving subject target and
manual-tracking suspension are runtime state and must never enter a workspace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from .camera_commands import CameraCommandId, CameraState, FaceOnSide

CAMERA_PREFERENCES_FORMAT = "camera-preferences/v1"
CAMERA_VIEWPORT_IDS = ("impact", "swing", "flight")
_VIEW_COMMANDS = frozenset(
    {
        CameraCommandId.VIEW_ISOMETRIC,
        CameraCommandId.VIEW_FACE_ON,
        CameraCommandId.VIEW_DOWN_THE_LINE,
        CameraCommandId.VIEW_OVERHEAD,
    }
)
_MIN_ZOOM = 0.25
_MAX_ZOOM = 8.0


@dataclass(frozen=True, slots=True)
class CameraPreference:
    """Durable, viewport-local camera choices."""

    preset_id: CameraCommandId = CameraCommandId.VIEW_ISOMETRIC
    face_on_side: FaceOnSide = FaceOnSide.RIGHT
    zoom: float = 1.0
    tracking_enabled: bool = False
    auto_fit_enabled: bool = False

    def __post_init__(self) -> None:
        if self.preset_id not in _VIEW_COMMANDS:
            raise ValueError("camera preference preset_id must be a view command")
        if not isinstance(self.face_on_side, FaceOnSide):
            raise TypeError("camera preference face_on_side is invalid")
        if isinstance(self.zoom, bool) or not isinstance(self.zoom, (int, float)):
            raise TypeError("camera preference zoom must be numeric")
        if not math.isfinite(self.zoom) or not _MIN_ZOOM <= self.zoom <= _MAX_ZOOM:
            raise ValueError(
                f"camera preference zoom must be within [{_MIN_ZOOM}, {_MAX_ZOOM}]"
            )
        if not isinstance(self.tracking_enabled, bool) or not isinstance(
            self.auto_fit_enabled, bool
        ):
            raise TypeError("camera preference tracking and Auto Fit must be booleans")

    def to_document(self) -> dict[str, object]:
        """Return the exact JSON-safe preference object."""
        return {
            "preset_id": self.preset_id.value,
            "face_on_side": self.face_on_side.value,
            "zoom": float(self.zoom),
            "tracking_enabled": self.tracking_enabled,
            "auto_fit_enabled": self.auto_fit_enabled,
        }

    @classmethod
    def from_document(cls, value: object) -> CameraPreference:
        """Parse one exact preference without coercing malformed values."""
        data = _exact_mapping(
            value,
            {
                "preset_id",
                "face_on_side",
                "zoom",
                "tracking_enabled",
                "auto_fit_enabled",
            },
            "camera preference",
        )
        preset_value = data["preset_id"]
        side_value = data["face_on_side"]
        if not isinstance(preset_value, str) or not isinstance(side_value, str):
            raise TypeError("camera preference enum values must be strings")
        try:
            preset = CameraCommandId(preset_value)
            side = FaceOnSide(side_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("camera preference contains an unsupported enum") from exc
        zoom = data["zoom"]
        tracking = data["tracking_enabled"]
        auto_fit = data["auto_fit_enabled"]
        if isinstance(zoom, bool) or not isinstance(zoom, (int, float)):
            raise TypeError("camera preference zoom must be numeric")
        if not isinstance(tracking, bool) or not isinstance(auto_fit, bool):
            raise TypeError("camera preference tracking and Auto Fit must be booleans")
        return cls(
            preset_id=preset,
            face_on_side=side,
            zoom=float(zoom),
            tracking_enabled=tracking,
            auto_fit_enabled=auto_fit,
        )


@dataclass(frozen=True, slots=True)
class CameraPreferences:
    """Complete camera preferences keyed by stable viewport identity."""

    viewports: Mapping[str, CameraPreference]

    def __post_init__(self) -> None:
        if not isinstance(self.viewports, Mapping):
            raise TypeError("camera preference viewports must be a mapping")
        values = dict(self.viewports)
        if set(values) != set(CAMERA_VIEWPORT_IDS):
            raise ValueError(
                "camera preferences must contain every stable viewport once"
            )
        if any(not isinstance(value, CameraPreference) for value in values.values()):
            raise TypeError("camera preferences contain an invalid viewport value")
        object.__setattr__(self, "viewports", MappingProxyType(values))

    def to_document(self) -> dict[str, object]:
        """Return canonical camera-preferences/v1 JSON."""
        return {
            "format": CAMERA_PREFERENCES_FORMAT,
            "viewports": {
                viewport_id: self.viewports[viewport_id].to_document()
                for viewport_id in CAMERA_VIEWPORT_IDS
            },
        }

    @classmethod
    def from_document(cls, value: object) -> CameraPreferences:
        """Parse an exact current document and reject future formats."""
        data = _exact_mapping(value, {"format", "viewports"}, "camera preferences")
        if data["format"] != CAMERA_PREFERENCES_FORMAT:
            raise ValueError(
                f"unsupported camera preferences format: {data['format']!r}"
            )
        viewports = _exact_mapping(
            data["viewports"], set(CAMERA_VIEWPORT_IDS), "camera preference viewports"
        )
        return cls(
            {
                viewport_id: CameraPreference.from_document(viewports[viewport_id])
                for viewport_id in CAMERA_VIEWPORT_IDS
            }
        )


def default_camera_preferences() -> CameraPreferences:
    """Return #4303 defaults: neutral impact and tracked 2x moving views."""
    stationary = CameraPreference()
    moving = CameraPreference(
        zoom=2.0,
        tracking_enabled=True,
        auto_fit_enabled=True,
    )
    return CameraPreferences({"impact": stationary, "swing": moving, "flight": moving})


def preference_from_camera_state(
    state: CameraState, fallback: CameraPreference
) -> CameraPreference:
    """Capture durable fields, retaining the last preset after manual orbit."""
    if not isinstance(state, CameraState) or not isinstance(fallback, CameraPreference):
        raise TypeError("camera state and fallback preference are required")
    preset = (
        state.preset_id if state.preset_id in _VIEW_COMMANDS else fallback.preset_id
    )
    return CameraPreference(
        preset_id=preset,
        face_on_side=state.face_on_side,
        zoom=state.zoom,
        tracking_enabled=state.tracking_enabled,
        auto_fit_enabled=state.auto_fit_enabled,
    )


def apply_camera_preference(
    state: CameraState, preference: CameraPreference
) -> CameraState:
    """Apply durable fields while retaining the live target and clearing suspension."""
    if not isinstance(state, CameraState) or not isinstance(
        preference, CameraPreference
    ):
        raise TypeError("camera state and preference are required")
    return replace(
        state,
        preset_id=preference.preset_id,
        face_on_side=preference.face_on_side,
        zoom=preference.zoom,
        tracking_enabled=preference.tracking_enabled,
        tracking_suspended=False,
        auto_fit_enabled=preference.auto_fit_enabled,
    )


def _exact_mapping(
    value: object, expected: set[str], context: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{context} has invalid fields")
    return value


__all__ = [
    "CAMERA_PREFERENCES_FORMAT",
    "CAMERA_VIEWPORT_IDS",
    "CameraPreference",
    "CameraPreferences",
    "apply_camera_preference",
    "default_camera_preferences",
    "preference_from_camera_state",
]
