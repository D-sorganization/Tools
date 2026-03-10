"""Asteroid shape generation.

Supports circular, elliptical, and random-polygon asteroids.
All shapes return vertices in body-frame (centred at origin).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto


class ShapeKind(Enum):
    """Available asteroid shape types."""

    CIRCLE = auto()
    ELLIPSE = auto()
    RANDOM = auto()


@dataclass(frozen=True)
class AsteroidShape:
    """Immutable asteroid shape description.

    Attributes:
        kind: Which shape type was generated.
        vertices: Polygon vertices in body frame (closed loop).
        semi_a: Semi-major axis (m) used for MoI calculation.
        semi_b: Semi-minor axis (m) used for MoI calculation.
    """

    kind: ShapeKind
    vertices: tuple[tuple[float, float], ...]
    semi_a: float
    semi_b: float

    def __post_init__(self) -> None:
        assert len(self.vertices) >= 3, "Need at least 3 vertices"
        assert self.semi_a > 0 and self.semi_b > 0


def _polar_to_xy(r: float, theta: float) -> tuple[float, float]:
    """Convert polar to Cartesian coordinates."""
    return r * math.cos(theta), r * math.sin(theta)


def make_circle(radius: float, n_pts: int = 32) -> AsteroidShape:
    """Create a circular asteroid shape."""
    assert radius > 0, f"radius must be positive, got {radius}"
    assert n_pts >= 8
    verts = tuple(_polar_to_xy(radius, 2 * math.pi * i / n_pts) for i in range(n_pts))
    return AsteroidShape(
        kind=ShapeKind.CIRCLE,
        vertices=verts,
        semi_a=radius,
        semi_b=radius,
    )


def make_ellipse(semi_a: float, semi_b: float, n_pts: int = 32) -> AsteroidShape:
    """Create an elliptical asteroid shape."""
    assert semi_a > 0 and semi_b > 0
    assert n_pts >= 8
    verts = tuple(
        (
            semi_a * math.cos(2 * math.pi * i / n_pts),
            semi_b * math.sin(2 * math.pi * i / n_pts),
        )
        for i in range(n_pts)
    )
    return AsteroidShape(
        kind=ShapeKind.ELLIPSE,
        vertices=verts,
        semi_a=semi_a,
        semi_b=semi_b,
    )


def make_random(
    base_radius: float,
    roughness: float = 0.3,
    n_pts: int = 20,
    seed: int | None = None,
) -> AsteroidShape:
    """Create a random lumpy asteroid polygon.

    Args:
        base_radius: Mean radius (m).
        roughness: Fractional variation in radius [0, 1].
        n_pts: Number of polygon vertices.
        seed: RNG seed for reproducibility.
    """
    assert base_radius > 0
    assert 0.0 <= roughness <= 1.0
    assert n_pts >= 6
    rng = random.Random(seed)
    radii = [
        base_radius * (1.0 + roughness * (rng.random() * 2 - 1)) for _ in range(n_pts)
    ]
    verts = tuple(_polar_to_xy(radii[i], 2 * math.pi * i / n_pts) for i in range(n_pts))
    # Estimate bounding semi-axes for MoI
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    semi_a = max(abs(min(xs)), abs(max(xs)))
    semi_b = max(abs(min(ys)), abs(max(ys)))
    return AsteroidShape(
        kind=ShapeKind.RANDOM,
        vertices=verts,
        semi_a=max(semi_a, 0.01),
        semi_b=max(semi_b, 0.01),
    )


def surface_normal_at_angle(
    shape: AsteroidShape, angle_rad: float
) -> tuple[float, float]:
    """Approximate outward surface normal at the given body-frame angle.

    Finds the nearest vertex and uses the polygon edge normal.
    Returns a unit vector (nx, ny).
    """
    assert len(shape.vertices) >= 3
    n = len(shape.vertices)
    # Find vertex index closest to the desired angle
    best_idx = 0
    best_diff = float("inf")
    for i, (x, y) in enumerate(shape.vertices):
        v_angle = math.atan2(y, x)
        diff = abs(_angle_diff(v_angle, angle_rad))
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    # Edge normal between best_idx and next
    i0 = best_idx
    i1 = (best_idx + 1) % n
    x0, y0 = shape.vertices[i0]
    x1, y1 = shape.vertices[i1]
    # Outward normal = perpendicular to edge, pointing away from origin
    ex, ey = x1 - x0, y1 - y0
    # Perpendicular (rotated 90° CCW)
    nx, ny = -ey, ex
    length = math.hypot(nx, ny)
    if length < 1e-12:
        return math.cos(angle_rad), math.sin(angle_rad)
    nx, ny = nx / length, ny / length
    # Ensure outward direction
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if nx * cx + ny * cy < 0:
        nx, ny = -nx, -ny
    return nx, ny


def surface_point_at_angle(
    shape: AsteroidShape, angle_rad: float
) -> tuple[float, float]:
    """Body-frame point on the asteroid surface at the given angle."""
    n = len(shape.vertices)
    # Find the two vertices that bracket the angle
    angles = [math.atan2(v[1], v[0]) for v in shape.vertices]
    # Normalize angles relative to angle_rad
    best_i = 0
    best_diff = float("inf")
    for i, a in enumerate(angles):
        diff = abs(_angle_diff(a, angle_rad))
        if diff < best_diff:
            best_diff = diff
            best_i = i
    # Interpolate between best_i and next
    i0, i1 = best_i, (best_i + 1) % n
    a0, a1 = angles[i0], angles[i1]
    da = _angle_diff(a1, a0)
    dr = _angle_diff(angle_rad, a0)
    t = (dr / da) if abs(da) > 1e-9 else 0.0
    t = max(0.0, min(1.0, t))
    x0, y0 = shape.vertices[i0]
    x1, y1 = shape.vertices[i1]
    return x0 + t * (x1 - x0), y0 + t * (y1 - y0)


def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference a - b, wrapped to [-π, π]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d
