"""Small strict vector helpers shared by neutral terrain adapter records."""

from __future__ import annotations

import math

from .contract_types import Vector3
from .profile_validation import finite_number

UNIT_TOLERANCE = 1e-10


def vector(value: object, name: str) -> Vector3:
    """Return one canonical finite three-vector."""
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain three components")
    return (
        finite_number(value[0], name),
        finite_number(value[1], name),
        finite_number(value[2], name),
    )


def dot(left: Vector3, right: Vector3) -> float:
    """Return a scalar dot product."""
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


def unit_vector(value: object, name: str) -> Vector3:
    """Return one canonical unit vector."""
    normalized = vector(value, name)
    if abs(math.hypot(*normalized) - 1.0) > UNIT_TOLERANCE:
        raise ValueError(f"{name} must be a unit vector")
    return normalized


def matrix_vector(rows: tuple[Vector3, Vector3, Vector3], value: Vector3) -> Vector3:
    """Multiply one three-row matrix by one vector."""
    return (dot(rows[0], value), dot(rows[1], value), dot(rows[2], value))


def determinant(rows: tuple[Vector3, Vector3, Vector3]) -> float:
    """Return a three-row matrix determinant."""
    first, second, third = rows
    return float(
        first[0] * (second[1] * third[2] - second[2] * third[1])
        - first[1] * (second[0] * third[2] - second[2] * third[0])
        + first[2] * (second[0] * third[1] - second[1] * third[0])
    )


__all__ = [
    "UNIT_TOLERANCE",
    "determinant",
    "dot",
    "matrix_vector",
    "unit_vector",
    "vector",
]
