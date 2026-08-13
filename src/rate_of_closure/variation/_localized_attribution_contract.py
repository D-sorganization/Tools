"""Shared bounds and numerical policy for localized attribution authority."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from shared.python.contracts import require

MAX_TEXT_LENGTH = 256
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_SOURCES = 32
MAX_TARGETS = 64
MAX_PAIRS = 4096
MAX_OBSERVATIONS = 131_072
RESPONSE_ULPS = 4.0
STATUS_VALUES = frozenset({"evaluated_hit", "evaluated_no_impact", "numerical_failure"})


@dataclass(frozen=True)
class TargetDefinition:
    """Canonical meaning for one target name."""

    kind: str
    unit: str
    convention: str
    coordinate_frame: str | None


TARGET_REGISTRY = {
    "position_x_m": TargetDefinition(
        "state", "m", "app-frame-cartesian-v1", "app_frame:x_target,y_up,z_right"
    ),
    "position_y_m": TargetDefinition(
        "state", "m", "app-frame-cartesian-v1", "app_frame:x_target,y_up,z_right"
    ),
    "position_z_m": TargetDefinition(
        "state", "m", "app-frame-cartesian-v1", "app_frame:x_target,y_up,z_right"
    ),
    "impact_time_s": TargetDefinition("impact", "s", "rate-of-closure-impact-v1", None),
    "clubhead_speed_mps": TargetDefinition(
        "impact", "m/s", "rate-of-closure-impact-v1", None
    ),
    "spin_loft_deg": TargetDefinition(
        "impact", "deg", "rate-of-closure-impact-v1", None
    ),
    "face_to_path_deg": TargetDefinition(
        "impact", "deg", "rate-of-closure-impact-v1", None
    ),
    "spin_axis_tilt_deg": TargetDefinition(
        "impact", "deg", "rate-of-closure-impact-v1", None
    ),
    "ball_speed_mph": TargetDefinition(
        "shot", "mph", "rate-of-closure-flight-v1", None
    ),
    "launch_angle_deg": TargetDefinition(
        "shot", "deg", "rate-of-closure-flight-v1", None
    ),
    "launch_azimuth_deg": TargetDefinition(
        "shot", "deg", "rate-of-closure-flight-v1", None
    ),
    "spin_rpm": TargetDefinition("shot", "rpm", "rate-of-closure-flight-v1", None),
    "carry_m": TargetDefinition("shot", "m", "rate-of-closure-flight-v1", None),
    "lateral_m": TargetDefinition("shot", "m", "rate-of-closure-flight-v1", None),
    "max_height_m": TargetDefinition("shot", "m", "rate-of-closure-flight-v1", None),
    "flight_time_s": TargetDefinition("shot", "s", "rate-of-closure-flight-v1", None),
    "landing_angle_deg": TargetDefinition(
        "shot", "deg", "rate-of-closure-flight-v1", None
    ),
}


def require_bounded_count(size: int, maximum: int, label: str) -> None:
    """Reject resource-heavy arrays before dependent allocation."""
    require(0 <= size <= maximum, f"{label} exceeds resource cap", (size, maximum))


def require_authority_shape(
    sources: int, targets: int, pairs: int, observations: int
) -> None:
    """Bound all arrays and the complete matrix before materialization."""
    require_bounded_count(sources, MAX_SOURCES, "sources")
    require_bounded_count(targets, MAX_TARGETS, "targets")
    require_bounded_count(pairs, MAX_PAIRS, "pairs")
    require_bounded_count(observations, MAX_OBSERVATIONS, "observations")
    require(
        pairs * targets <= MAX_OBSERVATIONS, "pair-target matrix exceeds resource cap"
    )


def response_matches(actual: float, expected: float) -> bool:
    """Use the cross-runtime four-scaled-ULP response policy."""
    tolerance = RESPONSE_ULPS * sys.float_info.epsilon * max(1.0, abs(expected))
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


__all__ = [
    "MAX_OBSERVATIONS",
    "MAX_PAIRS",
    "MAX_SAFE_INTEGER",
    "MAX_SOURCES",
    "MAX_TARGETS",
    "MAX_TEXT_LENGTH",
    "TARGET_REGISTRY",
    "require_bounded_count",
    "require_authority_shape",
    "response_matches",
]
