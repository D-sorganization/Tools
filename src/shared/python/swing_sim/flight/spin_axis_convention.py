"""Shared launch-monitor-style spin-axis convention calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable

_ZERO_SPIN_TOLERANCE = 1e-12


def spin_axis_tilt_deg(spin_vector: Iterable[float]) -> float | None:
    """Return fade/right-positive tilt after projecting out forward gyro spin.

    Preconditions:
        spin_vector has exactly three finite target-frame components.

    Returns:
        None for zero total spin; otherwise the target-frame tilt angle.
    """
    values = tuple(float(component) for component in spin_vector)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("spin_vector must contain three finite components")
    if math.sqrt(sum(value * value for value in values)) <= _ZERO_SPIN_TOLERANCE:
        return None
    return math.degrees(math.atan2(-values[1], values[2]))


__all__ = ["spin_axis_tilt_deg"]
