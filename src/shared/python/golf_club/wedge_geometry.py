"""Kernel-independent wedge profile and ground-contact candidate geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ._validation import Vector3, require_vector3
from .wedge_parameters import Handedness, WedgeHeadParameters


class WedgeContactFeature(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Stable regions used by clearance and ground-contact consumers."""

    LEADING_EDGE_CENTER = "leading_edge_center"
    LEADING_EDGE_HEEL = "leading_edge_heel"
    LEADING_EDGE_TOE = "leading_edge_toe"
    PRIMARY_SOLE_CENTER = "primary_sole_center"
    PRIMARY_SOLE_HEEL = "primary_sole_heel"
    PRIMARY_SOLE_TOE = "primary_sole_toe"
    TRAILING_SOLE_CENTER = "trailing_sole_center"
    TRAILING_SOLE_HEEL = "trailing_sole_heel"
    TRAILING_SOLE_TOE = "trailing_sole_toe"


@dataclass(frozen=True)
class WedgeContactCandidate:
    """One named local head-frame point eligible for planar ground contact."""

    feature: WedgeContactFeature
    local_point_m: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.feature, WedgeContactFeature):
            raise TypeError("feature must be WedgeContactFeature")
        object.__setattr__(
            self,
            "local_point_m",
            require_vector3(self.local_point_m, "local_point_m"),
        )


def wedge_body_profile_m(
    parameters: WedgeHeadParameters,
) -> tuple[tuple[float, float], ...]:
    """Return the canonical closed body profile as head-frame ``(x, y)`` points."""
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    loft = math.radians(parameters.loft_deg)
    bounce = math.radians(parameters.bounce_deg)
    progression = parameters.face_progression_m
    leading_y = parameters.leading_edge_radius_m
    face_top = (
        progression - parameters.face_height_m * math.sin(loft),
        leading_y + parameters.face_height_m * math.cos(loft),
    )
    top_back = (
        face_top[0] - parameters.topline_thickness_m * math.cos(loft),
        face_top[1] - parameters.topline_thickness_m * math.sin(loft),
    )
    trailing = (
        progression - parameters.sole_width_m * math.cos(bounce),
        leading_y + parameters.sole_width_m * math.sin(bounce),
    )
    vertical_span = top_back[1] - trailing[1]
    upper_control = (
        top_back[0] - 0.12 * parameters.sole_width_m,
        top_back[1] - 0.18 * vertical_span,
    )
    lower_control = (
        trailing[0] - parameters.rear_curve_depth_fraction * parameters.sole_width_m,
        trailing[1] + 0.32 * vertical_span,
    )
    return (
        (progression, leading_y),
        face_top,
        top_back,
        upper_control,
        lower_control,
        trailing,
    )


def wedge_contact_candidates(
    parameters: WedgeHeadParameters,
) -> tuple[WedgeContactCandidate, ...]:
    """Derive the nine stable leading-edge/sole candidates from canonical datums."""
    profile = wedge_body_profile_m(parameters)
    leading_center = (profile[0][0], 0.0)
    trailing_center = profile[-1]
    primary_center = (
        0.5 * (profile[0][0] + trailing_center[0]),
        0.5 * (profile[0][1] + trailing_center[1]),
    )
    half_span = 0.5 * parameters.face_length_m
    heel_sign = -1.0 if parameters.handedness is Handedness.RIGHT else 1.0
    z_by_region = (
        ("center", 0.0),
        ("heel", heel_sign * half_span),
        ("toe", -heel_sign * half_span),
    )
    rows = (
        ("leading_edge", leading_center),
        ("primary_sole", primary_center),
        ("trailing_sole", trailing_center),
    )
    candidates: list[WedgeContactCandidate] = []
    for row_name, (x_value, y_value) in rows:
        for region_name, z_value in z_by_region:
            feature = WedgeContactFeature(f"{row_name}_{region_name}")
            candidates.append(
                WedgeContactCandidate(feature, (x_value, y_value, z_value))
            )
    return tuple(candidates)


__all__ = [
    "WedgeContactCandidate",
    "WedgeContactFeature",
    "wedge_body_profile_m",
    "wedge_contact_candidates",
]
