"""Electrode Advisor Configuration Module.

This module provides configuration constants, color schemes, view presets,
and UI defaults for the Electrode Advisor widget.
"""

from .color_schemes import (
    COLOR_SCHEMES,
    DEFAULT_COLOR_SCHEME,
    GLASS_INTEGRATION_COLORS,
    STATUS_COLORS,
    get_available_schemes,
    get_color_scheme,
    get_status_color,
)
from .ui_defaults import (
    COLOR_SCALE_MAX,
    COLOR_SCALE_MIN,
    DEFAULT_INTERACTION_MODE,
    ELECTRODE_EXTENSION_SLIDER,
    PERIODIC_UPDATE_MS,
    TRANSPARENCY_SLIDERS,
    UPDATE_DELAY_MS,
    ZOOM_SLIDER,
    SliderConfig,
    SpinBoxConfig,
    get_transparency_default,
    get_transparency_range,
)
from .view_presets import (
    DEFAULT_VIEW_PRESET,
    DEFAULT_Z_SCALE_FACTOR,
    DEFAULT_ZOOM_SCALE_FACTOR,
    VIEW_PRESETS,
    ViewAngle,
    get_available_presets,
    get_view_preset,
)

__all__ = [
    # Color schemes
    "COLOR_SCHEMES",
    "DEFAULT_COLOR_SCHEME",
    "GLASS_INTEGRATION_COLORS",
    "STATUS_COLORS",
    "get_available_schemes",
    "get_color_scheme",
    "get_status_color",
    # UI defaults
    "COLOR_SCALE_MAX",
    "COLOR_SCALE_MIN",
    "DEFAULT_INTERACTION_MODE",
    "ELECTRODE_EXTENSION_SLIDER",
    "PERIODIC_UPDATE_MS",
    "TRANSPARENCY_SLIDERS",
    "UPDATE_DELAY_MS",
    "ZOOM_SLIDER",
    "SliderConfig",
    "SpinBoxConfig",
    "get_transparency_default",
    "get_transparency_range",
    # View presets
    "DEFAULT_VIEW_PRESET",
    "DEFAULT_Z_SCALE_FACTOR",
    "DEFAULT_ZOOM_SCALE_FACTOR",
    "VIEW_PRESETS",
    "ViewAngle",
    "get_available_presets",
    "get_view_preset",
]
