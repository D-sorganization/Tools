"""Small reusable aerodynamic coefficient helpers."""

from __future__ import annotations

from ._constants import MAX_GOLF_BALL_LIFT_COEFFICIENT


def capped_lift_coefficient(value: float) -> float:
    """Return a physically bounded golf-ball lift coefficient."""
    if value <= 0.0:
        return 0.0
    return float(min(MAX_GOLF_BALL_LIFT_COEFFICIENT, value))


def spin_ratio_lift_coefficient(spin_ratio: float, maximum: float) -> float:
    """Calibrate low-spin lift without allowing high-spin ballooning."""
    if spin_ratio <= 0.0 or maximum <= 0.0:
        return 0.0
    return capped_lift_coefficient(min(maximum, 1.7 * spin_ratio))


__all__ = [
    "capped_lift_coefficient",
    "spin_ratio_lift_coefficient",
]
