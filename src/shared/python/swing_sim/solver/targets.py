"""Landing target regions for the impact-parameter solver (#4125 H7b).

A :class:`TargetRegion` describes where the ball should finish on the
course, in the landing plane (carry downrange [m], lateral [m, + right
of the target line] — the goals module's sign convention):

- ``green``: a circle at a downrange distance (optionally offset
  laterally) with a radius — the putting green;
- ``fairway``: an axis-aligned corridor, a downrange distance band
  crossed with a half-width about the target line.

The region exposes an exact signed distance (negative inside, zero on
the boundary, positive outside) and a containment test built on it.
The solver residual is the distance *outside* the region (0 anywhere
inside) plus a small centering term so the optimizer has gradient
toward the middle once inside — see ``objective.evaluate_candidate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from shared.python.contracts import require

__all__ = ["TargetRegion"]

#: Weight of the inside-the-region centering term relative to the
#: outside distance (small: containment dominates, centering breaks ties).
CENTERING_WEIGHT = 0.05


@dataclass(frozen=True)
class TargetRegion:
    """A landing target: green circle or fairway corridor.

    Attributes:
        kind: ``"green"`` (circle) or ``"fairway"`` (corridor).
        distance_m: Downrange center of the region [m] — circle center
            for a green, distance-band midpoint for a fairway.
        radius_m: Green radius [m] (``green`` only, > 0).
        lateral_m: Lateral center offset [m, + right] (``green`` only).
        band_half_length_m: Half-length of the fairway distance band [m]
            (``fairway`` only, > 0).
        half_width_m: Fairway half-width about the target line [m]
            (``fairway`` only, > 0).
    """

    kind: Literal["green", "fairway"]
    distance_m: float
    radius_m: float = 10.0
    lateral_m: float = 0.0
    band_half_length_m: float = 15.0
    half_width_m: float = 16.0

    def __post_init__(self) -> None:
        require(self.kind in ("green", "fairway"), "unknown region kind", self.kind)
        for name in (
            "distance_m",
            "radius_m",
            "lateral_m",
            "band_half_length_m",
            "half_width_m",
        ):
            require(
                math.isfinite(getattr(self, name)),
                f"{name} must be finite",
                getattr(self, name),
            )
        require(self.distance_m > 0.0, "distance must be > 0", self.distance_m)
        require(self.radius_m > 0.0, "radius must be > 0", self.radius_m)
        require(
            self.band_half_length_m > 0.0,
            "band half-length must be > 0",
            self.band_half_length_m,
        )
        require(self.half_width_m > 0.0, "half-width must be > 0", self.half_width_m)

    @property
    def center(self) -> tuple[float, float]:
        """(carry, lateral) center of the region [m]."""
        if self.kind == "green":
            return (self.distance_m, self.lateral_m)
        return (self.distance_m, 0.0)

    def signed_distance(self, carry_m: float, lateral_m: float) -> float:
        """Exact signed distance [m]: negative inside, 0 on the boundary.

        Green: Euclidean distance to the circle. Fairway: the standard
        axis-aligned box SDF over the distance band x half-width.
        """
        require(
            math.isfinite(carry_m) and math.isfinite(lateral_m),
            "landing point must be finite",
            (carry_m, lateral_m),
        )
        if self.kind == "green":
            return (
                math.hypot(carry_m - self.distance_m, lateral_m - self.lateral_m)
                - self.radius_m
            )
        dx = abs(carry_m - self.distance_m) - self.band_half_length_m
        dz = abs(lateral_m) - self.half_width_m
        if dx <= 0.0 and dz <= 0.0:
            return max(dx, dz)
        return math.hypot(max(dx, 0.0), max(dz, 0.0))

    def contains(self, carry_m: float, lateral_m: float) -> bool:
        """Whether the landing point is inside (or on) the region."""
        return self.signed_distance(carry_m, lateral_m) <= 0.0

    def residual_m(self, carry_m: float, lateral_m: float) -> float:
        """Solver residual [m]: distance outside (0 inside) + centering.

        The centering term is :data:`CENTERING_WEIGHT` times the distance
        to the region center, keeping gradient alive inside the region
        without ever dominating containment.
        """
        outside = max(self.signed_distance(carry_m, lateral_m), 0.0)
        cx, cz = self.center
        centering = CENTERING_WEIGHT * math.hypot(carry_m - cx, lateral_m - cz)
        return outside + centering
