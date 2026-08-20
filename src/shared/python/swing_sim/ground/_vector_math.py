"""Small immutable-vector operations used by the ground impact solver."""

from __future__ import annotations

import math
from typing import cast

from .contract_types import Vector3


def add(left: Vector3, right: Vector3) -> Vector3:
    """Return the componentwise vector sum."""
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def subtract(left: Vector3, right: Vector3) -> Vector3:
    """Return the componentwise vector difference."""
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale(vector: Vector3, factor: float) -> Vector3:
    """Return a vector multiplied by a scalar."""
    return (factor * vector[0], factor * vector[1], factor * vector[2])


def dot(left: Vector3, right: Vector3) -> float:
    """Return the Euclidean dot product."""
    return cast(float, sum(a * b for a, b in zip(left, right, strict=True)))


def cross(left: Vector3, right: Vector3) -> Vector3:
    """Return the right-handed cross product."""
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Vector3) -> float:
    """Return the Euclidean vector magnitude."""
    return math.sqrt(dot(vector, vector))


def unit(vector: Vector3, *, tolerance: float) -> Vector3:
    """Return a unit vector, rejecting a magnitude within ``tolerance``."""
    magnitude = norm(vector)
    if magnitude <= tolerance:
        raise ValueError("cannot normalize a near-zero vector")
    return scale(vector, 1.0 / magnitude)


def intrinsic_tangent_axis(normal: Vector3) -> Vector3:
    """Return a deterministic unit tangent for an arbitrary unit normal."""
    references = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    reference = min(references, key=lambda axis: abs(dot(axis, normal)))
    projected = subtract(reference, scale(normal, dot(reference, normal)))
    return unit(projected, tolerance=1e-15)


def interpolate(left: Vector3, right: Vector3, fraction: float) -> Vector3:
    """Return the componentwise affine interpolation."""
    return add(left, scale(subtract(right, left), fraction))


__all__ = [
    "add",
    "cross",
    "dot",
    "interpolate",
    "intrinsic_tangent_axis",
    "norm",
    "scale",
    "subtract",
    "unit",
]
