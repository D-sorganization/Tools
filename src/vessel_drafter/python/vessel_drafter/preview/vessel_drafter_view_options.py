"""Validated 3D view options for vessel preview rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians

from vessel_drafter.contracts import require_finite


@dataclass(frozen=True)
class Vessel3DViewOptions:
    split_enabled: bool = False
    split_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        require_finite("split_angle_degrees", self.split_angle_degrees)

    @property
    def normalized_split_angle_degrees(self) -> float:
        return self.split_angle_degrees % 360.0

    @property
    def normalized_split_angle_radians(self) -> float:
        return radians(self.normalized_split_angle_degrees)
