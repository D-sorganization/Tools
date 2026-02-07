"""UI default values and widget configuration for the Electrode Advisor.

This module contains all default values, ranges, and configuration constants
for UI widgets used by the Electrode Advisor widget.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class SliderConfig(NamedTuple):
    """Configuration for a slider widget."""

    min_value: int
    max_value: int
    default_value: int
    single_step: int = 1
    tick_interval: int = 0


class SpinBoxConfig(NamedTuple):
    """Configuration for a spinbox widget."""

    min_value: float
    max_value: float
    default_value: float
    decimals: int = 2
    single_step: float = 0.1


# Zoom slider configuration
ZOOM_SLIDER: Final[SliderConfig] = SliderConfig(
    min_value=50,  # 50% minimum zoom
    max_value=200,  # 200% maximum zoom
    default_value=100,  # 100% default
)

# Transparency slider configurations
TRANSPARENCY_SLIDERS: Final[dict[str, SliderConfig]] = {
    "electrodes": SliderConfig(min_value=10, max_value=100, default_value=80),
    "glass": SliderConfig(min_value=10, max_value=100, default_value=30),
    "metal": SliderConfig(min_value=10, max_value=100, default_value=70),
    "paths": SliderConfig(min_value=10, max_value=100, default_value=60),
    "refractory": SliderConfig(min_value=10, max_value=100, default_value=40),
    "metal_shell": SliderConfig(min_value=10, max_value=100, default_value=50),
}

# Electrode extension slider configuration
ELECTRODE_EXTENSION_SLIDER: Final[SliderConfig] = SliderConfig(
    min_value=0,
    max_value=36,
    default_value=24,
    single_step=1,
    tick_interval=2,
)

# Color scale spinbox configurations
COLOR_SCALE_MIN: Final[SpinBoxConfig] = SpinBoxConfig(
    min_value=0,
    max_value=1000,
    default_value=0,
    decimals=0,
    single_step=10,
)

COLOR_SCALE_MAX: Final[SpinBoxConfig] = SpinBoxConfig(
    min_value=0,
    max_value=10000,
    default_value=1000,
    decimals=0,
    single_step=100,
)

# Default interaction mode
DEFAULT_INTERACTION_MODE: Final[str] = "rotation"

# Update timer intervals (in milliseconds)
UPDATE_DELAY_MS: Final[int] = 100  # Debounce delay for input changes
PERIODIC_UPDATE_MS: Final[int] = 5000  # Periodic refresh interval


def get_transparency_default(component: str) -> int:
    """Get the default transparency value for a component.

    Args:
        component: Component name ('electrodes', 'glass', 'metal', etc.)

    Returns:
        Default transparency percentage (10-100).
    """
    config = TRANSPARENCY_SLIDERS.get(component)
    if config:
        return config.default_value
    return 50  # Fallback default


def get_transparency_range(component: str) -> tuple[int, int]:
    """Get the transparency range for a component.

    Args:
        component: Component name ('electrodes', 'glass', 'metal', etc.)

    Returns:
        Tuple of (min_value, max_value).
    """
    config = TRANSPARENCY_SLIDERS.get(component)
    if config:
        return (config.min_value, config.max_value)
    return (10, 100)  # Fallback default
