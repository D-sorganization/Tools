"""Validated manual delivery and head-geometry declarations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from rate_of_closure._contracts import ensure, require

__all__ = [
    "ManualDeliveryConfig",
    "ShaftAxisDatum",
    "manual_head_rotation",
    "manual_reference_velocity",
]

_MANUAL_ANGLE_LIMIT_DEG: dict[str, float] = {
    "attack_angle_deg": 89.0,
    "club_path_deg": 89.0,
    "forward_shaft_lean_deg": 60.0,
}


class ShaftAxisDatum(StrEnum):
    """Physical point used to declare the manual shaft axis."""

    TRACKED_REFERENCE = "tracked_reference"
    GENERATED_HOSEL = "generated_hosel"


@dataclass(frozen=True)
class ManualDeliveryConfig:
    """Manual reference delivery and head pose in the app frame.

    Positive club path travels right of target (+z), positive attack angle
    travels upward (+y), and positive forward shaft lean tips the shaft
    downrange (+x) by rotating the rigid head about local -z.
    """

    attack_angle_deg: float = 0.0
    club_path_deg: float = 0.0
    forward_shaft_lean_deg: float = 0.0
    shaft_axis_datum: ShaftAxisDatum = ShaftAxisDatum.TRACKED_REFERENCE

    def __post_init__(self) -> None:
        """Normalize the datum and reject non-finite or singular angles."""
        for name, limit_deg in _MANUAL_ANGLE_LIMIT_DEG.items():
            value = getattr(self, name)
            require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"{name} must be numeric",
                value,
            )
            require(
                math.isfinite(float(value)) and abs(float(value)) <= limit_deg,
                f"{name} must be finite and within +/-{limit_deg} deg",
                value,
            )
        try:
            datum = ShaftAxisDatum(self.shaft_axis_datum)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unknown manual shaft-axis datum {self.shaft_axis_datum!r}"
            ) from error
        object.__setattr__(self, "shaft_axis_datum", datum)


def manual_reference_velocity(
    speed_mps: float, delivery: ManualDeliveryConfig
) -> np.ndarray:
    """Return the declared reference velocity in the app frame [m/s]."""
    require(math.isfinite(speed_mps) and speed_mps > 0.0, "speed_mps must be > 0")
    attack = math.radians(delivery.attack_angle_deg)
    path = math.radians(delivery.club_path_deg)
    velocity: np.ndarray = speed_mps * np.array(
        (
            math.cos(attack) * math.cos(path),
            math.sin(attack),
            math.cos(attack) * math.sin(path),
        )
    )
    ensure(
        math.isclose(float(np.linalg.norm(velocity)), speed_mps, rel_tol=1e-12),
        "manual direction must preserve speed magnitude",
    )
    return velocity


def manual_head_rotation(delivery: ManualDeliveryConfig) -> np.ndarray:
    """Return the rigid impact pose for positive downrange shaft lean."""
    angle = math.radians(-delivery.forward_shaft_lean_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation: np.ndarray = np.array(
        (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    ensure(
        bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)),
        "manual head pose must be orthonormal",
    )
    return rotation
