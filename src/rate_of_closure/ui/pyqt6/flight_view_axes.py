"""Shared distance-axis formatting for Rate of Closure plots."""

from __future__ import annotations

from typing import Any

from matplotlib.ticker import FuncFormatter

from rate_of_closure.units import DISTANCE_UNITS, display_distance_unit


def distance_axis(axes: Any, which: str) -> str:
    """Format a canonical-metre axis in the selected display distance unit."""
    if which not in {"x", "y", "z"}:
        raise ValueError(f"which must be x, y, or z; got {which!r}")
    unit = display_distance_unit()
    factor = DISTANCE_UNITS[unit]
    formatter = FuncFormatter(lambda value, _position: f"{value / factor:.0f}")
    getattr(axes, f"{which}axis").set_major_formatter(formatter)
    return str(unit)


__all__ = ["distance_axis"]
