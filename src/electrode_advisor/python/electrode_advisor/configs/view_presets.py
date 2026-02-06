"""View presets and camera angle constants for the Electrode Advisor.

This module contains all view preset definitions and camera configuration
used by the Electrode Advisor 3D visualization system.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class ViewAngle(NamedTuple):
    """Camera view angle configuration."""

    elev: float  # Elevation angle in degrees
    azim: float  # Azimuth angle in degrees


# View preset definitions
# Each preset defines the camera elevation and azimuth angles
VIEW_PRESETS: Final[dict[str, ViewAngle]] = {
    "default": ViewAngle(elev=20, azim=45),
    "top": ViewAngle(elev=90, azim=0),
    "side": ViewAngle(elev=0, azim=0),
    "front": ViewAngle(elev=0, azim=90),
}

# Default view preset name
DEFAULT_VIEW_PRESET: Final[str] = "default"

# Zoom and scaling constants
DEFAULT_ZOOM_SCALE_FACTOR: Final[float] = 1.1  # Extra margin for default view
DEFAULT_Z_SCALE_FACTOR: Final[float] = 1.2  # Extra margin for z-axis in default view


def get_view_preset(preset_name: str) -> ViewAngle:
    """Get the view angle for a specific preset.

    Args:
        preset_name: Name of the view preset ('default', 'top', 'side', 'front').

    Returns:
        ViewAngle with elevation and azimuth values.
        Returns default view if preset is not found.
    """
    return VIEW_PRESETS.get(preset_name, VIEW_PRESETS[DEFAULT_VIEW_PRESET])


def get_available_presets() -> list[str]:
    """Get list of available view preset names.

    Returns:
        List of view preset names.
    """
    return list(VIEW_PRESETS.keys())
