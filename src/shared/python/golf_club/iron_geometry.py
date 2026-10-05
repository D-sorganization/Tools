"""Kernel-independent heel and toe body sections for the iron family."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .iron_parameters import IronHeadParameters
from .wedge_parameters import Handedness


@dataclass(frozen=True)
class IronBodySections:
    """Heel and toe section polygons and the ``z`` planes they lie in.

    Each profile is ``(leading edge, face top, topline back, back-wall top,
    trailing edge)`` as head-frame ``(x, y)`` metres, counter-clockwise. The
    body is the ruled loft between them with vertex-wise correspondence.
    """

    heel_profile_m: tuple[tuple[float, float], ...]
    toe_profile_m: tuple[tuple[float, float], ...]
    heel_z_m: float
    toe_z_m: float


def iron_body_sections_m(parameters: IronHeadParameters) -> IronBodySections:
    """Return the canonical heel and toe sections of one iron body.

    Postconditions: both profiles are simple, counter-clockwise polygons with
    the trailing edge on the ground (``y = 0``), and the heel-to-toe length
    exceeds the face-to-back depth (USGA/R&A Equipment Rules clubhead
    dimension requirement for non-putters).
    """
    if not isinstance(parameters, IronHeadParameters):
        raise TypeError("parameters must be IronHeadParameters")
    heel = _section_profile_m(parameters, parameters.heel_face_height_m)
    toe = _section_profile_m(parameters, parameters.toe_face_height_m)
    heel_sign = -1.0 if parameters.handedness is Handedness.RIGHT else 1.0
    half_length = 0.5 * parameters.blade_length_m
    depth = max(x_value for x_value, _ in heel + toe) - min(
        x_value for x_value, _ in heel + toe
    )
    assert depth < parameters.blade_length_m, "heel-toe must exceed face-back"
    assert _signed_area(heel) > 0.0 and _signed_area(toe) > 0.0
    return IronBodySections(
        heel_profile_m=heel,
        toe_profile_m=toe,
        heel_z_m=heel_sign * half_length,
        toe_z_m=-heel_sign * half_length,
    )


def _section_profile_m(
    parameters: IronHeadParameters, face_height_m: float
) -> tuple[tuple[float, float], ...]:
    loft = math.radians(parameters.loft_deg)
    bounce = math.radians(parameters.bounce_deg)
    leading = (0.0, parameters.sole_width_m * math.sin(bounce))
    face_top = (
        leading[0] - face_height_m * math.sin(loft),
        leading[1] + face_height_m * math.cos(loft),
    )
    topline_back = (
        face_top[0] - parameters.topline_thickness_m * math.cos(loft),
        face_top[1] - parameters.topline_thickness_m * math.sin(loft),
    )
    trailing = (-parameters.sole_width_m * math.cos(bounce), 0.0)
    back_wall_top = (trailing[0], parameters.back_wall_height_m)
    return (leading, face_top, topline_back, back_wall_top, trailing)


def _signed_area(profile: tuple[tuple[float, float], ...]) -> float:
    total = 0.0
    for index, (x_value, y_value) in enumerate(profile):
        x_next, y_next = profile[(index + 1) % len(profile)]
        total += x_value * y_next - x_next * y_value
    return 0.5 * total


__all__ = [
    "IronBodySections",
    "iron_body_sections_m",
]
