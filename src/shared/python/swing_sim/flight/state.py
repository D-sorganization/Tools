"""Full kinematic trajectory state for qualified flight-to-ground transfer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import TrajectoryPoint


@dataclass(frozen=True)
class FlightStatePoint(TrajectoryPoint):
    """Flight sample carrying signed angular velocity in radians per second."""

    angular_velocity_rad_s: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the inherited linear state and full angular vector."""
        super().__post_init__()
        omega = np.asarray(self.angular_velocity_rad_s, dtype=float)
        if omega.shape != (3,) or not np.all(np.isfinite(omega)):
            raise ValueError("angular_velocity_rad_s must be a finite 3-vector")
        object.__setattr__(self, "angular_velocity_rad_s", omega)


__all__ = ["FlightStatePoint"]
