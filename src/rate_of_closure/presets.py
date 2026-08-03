"""Named delivery presets for the rate-of-closure impact explorer.

Every rate figure here is sourced from the AffineDrift closure-rate
literature dossier (verified against the Cheetham 2014 paper text):
tour-driver horizontal turning velocity (HTV, about the shaft) is
1,307 +/- 304 deg/s with a range of 652-2,432 deg/s (n = 94), and the
global club closure velocity reconciles as

    CCV = HTV * sin(lie) + SPV * cos(lie)  ~ 2,100 deg/s tour mean.

The swing-plane rate (SPV) of 1,870 deg/s is chosen so the median
preset reproduces that CCV mean at a 58-degree impact lie. Presets are
starting points, not claims about any player — every value is editable
in the UI, and users with measured data should enter their own.
"""

from __future__ import annotations

from .model import ImpactScenario

__all__ = ["PRESETS", "preset_names"]

#: Ordered mapping of preset name to scenario.
PRESETS: dict[str, ImpactScenario] = {
    # Dossier tour median: HTV 1,307 with SPV set so CCV ~ 2,100.
    "Cheetham tour median (HTV 1,307)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=1870.0,
        omega_shaft_dps=1307.0,
    ),
    # +1 standard deviation of HTV (1,307 + 304).
    "Cheetham +1 SD (HTV 1,611)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=1870.0,
        omega_shaft_dps=1611.0,
    ),
    # Extremes of the reported n=94 tour range.
    "Cheetham range low (HTV 652)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=1870.0,
        omega_shaft_dps=652.0,
    ),
    "Cheetham range high (HTV 2,432)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=1870.0,
        omega_shaft_dps=2432.0,
    ),
    # Back-solves TrackMan's published ~3 degree GC-vs-face-center path
    # gap at d = 40 mm — the implied closure exceeds the Cheetham range,
    # the tension the AffineDrift derivation documents (R_ISA ~ 0.77 m).
    "TrackMan ~3 deg worked example": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=1870.0,
        omega_shaft_dps=3575.0,
    ),
    "Forum example (2,000 dps, vertical axis)": ImpactScenario(
        clubhead_speed_mph=120.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=2000.0,
        lie_angle_deg=90.0,
        com_to_face_mm=35.0,
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
