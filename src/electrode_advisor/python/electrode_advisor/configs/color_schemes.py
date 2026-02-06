"""Color schemes and visual constants for the Electrode Advisor.

This module contains all color scheme definitions and color utility functions
used by the Electrode Advisor visualization system.
"""

from __future__ import annotations

from typing import Final

# Default color scheme name
DEFAULT_COLOR_SCHEME: Final[str] = "Default"

# Available color schemes for electrode visualization
# Each scheme contains 3 colors: [primary, secondary, tertiary]
COLOR_SCHEMES: Final[dict[str, list[str]]] = {
    "Default": ["#FF4444", "#44FF44", "#4444FF"],
    "Heat Map": ["#FF0000", "#FF8C00", "#FFD700"],
    "Cool Tones": ["#4169E1", "#00CED1", "#32CD32"],
    "High Contrast": ["#FF0000", "#00FF00", "#0000FF"],
    "Viridis": ["#440154", "#31688E", "#35B779"],
    "Plasma": ["#0D0887", "#7E03A8", "#F89441"],
    "Copper": ["#000000", "#8B4513", "#CD853F"],
}

# Status indicator colors
STATUS_COLORS: Final[dict[str, str]] = {
    "ok": "#4CAF50",  # Green - success
    "success": "#4CAF50",
    "warning": "#FF9800",  # Orange - warning
    "warn": "#FF9800",
    "error": "#F44336",  # Red - error
    "neutral": "#666666",  # Gray - neutral/inactive
    "info": "#2196F3",  # Blue - informational
}

# Glass integration status colors
GLASS_INTEGRATION_COLORS: Final[dict[str, str]] = {
    "ok": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336",
    "neutral": "#666666",
}


def get_color_scheme(scheme_name: str) -> list[str]:
    """Get colors for a specific color scheme.

    Args:
        scheme_name: Name of the color scheme to retrieve.

    Returns:
        List of hex color strings [primary, secondary, tertiary].
        Returns Default scheme if the requested scheme is not found.
    """
    return COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES[DEFAULT_COLOR_SCHEME])


def get_status_color(status: str) -> str:
    """Get the color for a status indicator.

    Args:
        status: Status type ('ok', 'warning', 'error', 'neutral', 'info').

    Returns:
        Hex color string for the status.
        Returns neutral color if status is not recognized.
    """
    return STATUS_COLORS.get(status.lower(), STATUS_COLORS["neutral"])


def get_available_schemes() -> list[str]:
    """Get list of available color scheme names.

    Returns:
        List of color scheme names.
    """
    return list(COLOR_SCHEMES.keys())
