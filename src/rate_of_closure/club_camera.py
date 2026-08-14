"""Toolkit-neutral clubhead orbit camera transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

MIN_ELEVATION_DEG = -80.0
MAX_ELEVATION_DEG = 80.0
MIN_ZOOM = 0.3
MAX_ZOOM = 4.0
ORBIT_STEP_DEG = 5.0
ZOOM_STEP = 1.1


class ClubCameraAction(StrEnum):
    """Discrete camera actions shared by keyboard and buttons."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    HOME = "home"


@dataclass(frozen=True)
class ClubCamera:
    """Canonical orbit camera in degrees plus a dimensionless zoom."""

    azimuth_deg: float
    elevation_deg: float
    zoom: float

    def __post_init__(self) -> None:
        """Reject non-finite values and store one bounded canonical view."""
        values = (self.azimuth_deg, self.elevation_deg, self.zoom)
        if not all(type(value) in {int, float} and isfinite(value) for value in values):
            raise ValueError("club camera values must be finite")
        object.__setattr__(self, "azimuth_deg", _normalize_azimuth(self.azimuth_deg))
        object.__setattr__(
            self,
            "elevation_deg",
            min(MAX_ELEVATION_DEG, max(MIN_ELEVATION_DEG, self.elevation_deg)),
        )
        object.__setattr__(self, "zoom", min(MAX_ZOOM, max(MIN_ZOOM, self.zoom)))


def _normalize_azimuth(value: float) -> float:
    normalized = (value + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == 0.0 else normalized


DEFAULT_CLUB_CAMERA = ClubCamera(150.0, 30.0, 1.0)


def apply_club_camera_action(
    camera: ClubCamera, action: ClubCameraAction
) -> ClubCamera:
    """Return the bounded camera after one discrete action."""
    if action is ClubCameraAction.HOME:
        return DEFAULT_CLUB_CAMERA
    azimuth = camera.azimuth_deg
    elevation = camera.elevation_deg
    zoom = camera.zoom
    if action is ClubCameraAction.LEFT:
        azimuth -= ORBIT_STEP_DEG
    elif action is ClubCameraAction.RIGHT:
        azimuth += ORBIT_STEP_DEG
    elif action is ClubCameraAction.UP:
        elevation = min(MAX_ELEVATION_DEG, elevation + ORBIT_STEP_DEG)
    elif action is ClubCameraAction.DOWN:
        elevation = max(MIN_ELEVATION_DEG, elevation - ORBIT_STEP_DEG)
    elif action is ClubCameraAction.ZOOM_IN:
        zoom = min(MAX_ZOOM, zoom * ZOOM_STEP)
    elif action is ClubCameraAction.ZOOM_OUT:
        zoom = max(MIN_ZOOM, zoom / ZOOM_STEP)
    return ClubCamera(azimuth, elevation, zoom)


def apply_club_camera_drag(
    camera: ClubCamera, delta_x: float, delta_y: float
) -> ClubCamera:
    """Return the bounded camera after a finite pointer drag delta."""
    if not all(
        type(value) in {int, float} and isfinite(value) for value in (delta_x, delta_y)
    ):
        raise ValueError("club camera drag deltas must be finite")
    return ClubCamera(
        camera.azimuth_deg - delta_x * 0.45,
        camera.elevation_deg + delta_y * 0.45,
        camera.zoom,
    )


def matplotlib_view(camera: ClubCamera) -> tuple[float, float]:
    """Return Matplotlib elevation/azimuth for the canonical camera."""
    return camera.elevation_deg, 90.0 - camera.azimuth_deg


def camera_status(camera: ClubCamera, source: str) -> str:
    """Format one visible and assistive camera/source status."""
    return (
        f"{source}; camera azimuth {camera.azimuth_deg:.0f}°, "
        f"elevation {camera.elevation_deg:.0f}°, zoom {camera.zoom:.2f}×."
    )
