"""Versioned, fail-closed visual layout preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from rate_of_closure.club_camera import (
    DEFAULT_CLUB_CAMERA,
    MAX_ELEVATION_DEG,
    MAX_ZOOM,
    MIN_ELEVATION_DEG,
    MIN_ZOOM,
    ClubCamera,
)

VISUAL_LAYOUT_STATE_KEY = "visual_layout_v1"
MIN_SIDEBAR_FRACTION = 0.20
MAX_SIDEBAR_FRACTION = 0.38
DEFAULT_SIDEBAR_FRACTION = 0.27


class LayoutSettings(Protocol):
    """Small QSettings-compatible persistence boundary."""

    def value(self, key: str, default_value: object = None) -> object: ...

    def setValue(self, key: str, value: object) -> None: ...  # noqa: N802


@dataclass(frozen=True)
class VisualLayoutPreferences:
    """Presentation-only state safe to restore across application launches."""

    club_camera: ClubCamera = DEFAULT_CLUB_CAMERA
    module_help_open: bool = False
    shell_sidebar_fraction: float = DEFAULT_SIDEBAR_FRACTION


DEFAULT_VISUAL_LAYOUT = VisualLayoutPreferences()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_visual_layout(value: object) -> VisualLayoutPreferences:
    """Parse one exact v1 document without clamping forged persisted values."""
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("visual layout must be a version 1 object")
    camera = value.get("clubCamera")
    if not isinstance(camera, dict):
        raise ValueError("visual layout clubCamera must be an object")
    azimuth = _number(camera.get("azimuthDeg"), "camera azimuth")
    elevation = _number(camera.get("elevationDeg"), "camera elevation")
    zoom = _number(camera.get("zoom"), "camera zoom")
    if not -180.0 <= azimuth < 180.0:
        raise ValueError("camera azimuth is outside the canonical range")
    if not MIN_ELEVATION_DEG <= elevation <= MAX_ELEVATION_DEG:
        raise ValueError("camera elevation is outside the supported range")
    if not MIN_ZOOM <= zoom <= MAX_ZOOM:
        raise ValueError("camera zoom is outside the supported range")
    help_open = value.get("moduleHelpOpen")
    if type(help_open) is not bool:
        raise ValueError("moduleHelpOpen must be boolean")
    fraction = _number(value.get("shellSidebarFraction"), "sidebar fraction")
    if not MIN_SIDEBAR_FRACTION <= fraction <= MAX_SIDEBAR_FRACTION:
        raise ValueError("sidebar fraction is outside the supported range")
    return VisualLayoutPreferences(
        ClubCamera(azimuth, elevation, zoom), help_open, fraction
    )


def visual_layout_document(preferences: VisualLayoutPreferences) -> dict[str, object]:
    """Return the portable v1 JSON document for validated preferences."""
    return {
        "version": 1,
        "clubCamera": {
            "azimuthDeg": preferences.club_camera.azimuth_deg,
            "elevationDeg": preferences.club_camera.elevation_deg,
            "zoom": preferences.club_camera.zoom,
        },
        "moduleHelpOpen": preferences.module_help_open,
        "shellSidebarFraction": preferences.shell_sidebar_fraction,
    }


def load_visual_layout(settings: LayoutSettings) -> VisualLayoutPreferences:
    """Load supported state or return exact defaults on any storage corruption."""
    raw = settings.value(VISUAL_LAYOUT_STATE_KEY)
    if not isinstance(raw, str):
        return DEFAULT_VISUAL_LAYOUT
    try:
        return parse_visual_layout(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_VISUAL_LAYOUT


def save_visual_layout(
    settings: LayoutSettings, preferences: VisualLayoutPreferences
) -> bool:
    """Persist one validated document without surfacing storage failures."""
    try:
        document = visual_layout_document(preferences)
        validated = parse_visual_layout(document)
        settings.setValue(
            VISUAL_LAYOUT_STATE_KEY,
            json.dumps(visual_layout_document(validated), sort_keys=True),
        )
    except (TypeError, ValueError, OSError):
        return False
    return True
