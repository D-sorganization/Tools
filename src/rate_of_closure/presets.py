"""Named delivery presets for the rate-of-closure impact explorer.

The angular-velocity figures are *representative of ranges reported in
published 3-D golf motion studies* (Cheetham's AMM datasets and later
work): in-plane club rotation around 2,000-2,400 deg/s at impact for
skilled players, about-shaft rotation commonly 1,000-2,500 deg/s with
wide individual spread. They are starting points for exploration, not
claims about any player — every value is editable in the UI, and users
with measured data should enter their own.
"""

from __future__ import annotations

from .model import ImpactScenario

__all__ = ["PRESETS", "preset_names"]

#: Ordered mapping of preset name to scenario.
PRESETS: dict[str, ImpactScenario] = {
    "Tour representative": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=2200.0,
        omega_shaft_dps=1700.0,
    ),
    "High rate of closure": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=2300.0,
        omega_shaft_dps=2500.0,
    ),
    "Low rate of closure (body release)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=2200.0,
        omega_shaft_dps=800.0,
    ),
    "Amateur 95 mph": ImpactScenario(
        clubhead_speed_mph=95.0,
        omega_plane_dps=1900.0,
        omega_shaft_dps=1500.0,
    ),
    "Forum example (2000 dps, vertical axis)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=2000.0,
        lie_angle_deg=90.0,
    ),
    "Zero rotation (control)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=0.0,
    ),
}


def preset_names() -> list[str]:
    """Return preset names in display order."""
    return list(PRESETS)
