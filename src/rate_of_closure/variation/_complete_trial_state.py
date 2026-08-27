"""Strict immutable state projections for complete trial records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

import numpy as np

from shared.python.contracts import require

IMPACT_FIELDS = frozenset(
    {
        "mode",
        "status",
        "candidate_time_s",
        "closest_approach_m",
        "contact_threshold_m",
        "contact_margin_m",
        "ball_position_m",
        "frame",
        "geometry_model",
        "geometry_limitations",
    }
)
DELIVERY_FIELDS = frozenset(
    {
        "clubhead_velocity",
        "face_normal",
        "impact_offset",
        "clubhead_angular_velocity",
        "spin_loft_deg",
        "face_to_path_deg",
        "spin_axis",
        "spin_axis_tilt_deg",
        "dplane",
    }
)
POST_IMPACT_FIELDS = frozenset(
    {
        "ball_velocity",
        "ball_angular_velocity",
        "clubhead_velocity",
        "clubhead_angular_velocity",
        "contact_duration",
        "energy_transfer",
        "impact_location",
    }
)
LAUNCH_FIELDS = frozenset(
    {
        "ball_speed_mph",
        "launch_angle_deg",
        "launch_azimuth_deg",
        "spin_rpm",
        "spin_axis_tilt_deg",
        "carry_m",
        "max_height_m",
        "flight_time_s",
        "landing_angle_deg",
    }
)


def _freeze_state(value: object) -> object:
    """Convert dataclass state to immutable strict-JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        require(math.isfinite(number), "state numbers must be finite", number)
        return number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Enum):
        return _freeze_state(value.value)
    if isinstance(value, np.ndarray):
        require(bool(np.all(np.isfinite(value))), "state arrays must be finite")
        return tuple(_freeze_state(item) for item in value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return MappingProxyType(
            {
                item.name: _freeze_state(getattr(value, item.name))
                for item in fields(value)
            }
        )
    if isinstance(value, Mapping):
        require(
            all(isinstance(key, str) for key in value),
            "state mapping keys must be strings",
        )
        return MappingProxyType(
            {str(key): _freeze_state(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_state(item) for item in value)
    require(False, "unsupported complete-trial state value", type(value).__name__)
    raise AssertionError("unreachable")


def state_mapping(
    value: object, expected_fields: frozenset[str], name: str
) -> Mapping[str, object]:
    """Freeze and validate one exact top-level domain-state mapping."""
    frozen = _freeze_state(value)
    require(isinstance(frozen, Mapping), f"{name} must be an object")
    result = cast(Mapping[str, object], frozen)
    require(set(result) == expected_fields, f"{name} fields are incompatible")
    return result


__all__ = [
    "DELIVERY_FIELDS",
    "IMPACT_FIELDS",
    "LAUNCH_FIELDS",
    "POST_IMPACT_FIELDS",
    "state_mapping",
]
