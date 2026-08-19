"""Reusable point and frame transforms for convention adapters."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Vector3: TypeAlias = tuple[float, float, float]
Matrix3: TypeAlias = tuple[Vector3, Vector3, Vector3]


def _vector(value: object, name: str) -> npt.NDArray[np.float64]:
    vector: npt.NDArray[np.float64] = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three components")
    if not bool(np.all(np.isfinite(vector))):
        raise ValueError(f"{name} must be finite")
    return vector


def _tuple(value: npt.NDArray[np.float64]) -> Vector3:
    return float(value[0]), float(value[1]), float(value[2])


def shift_point_velocity(
    reference_velocity: object,
    angular_velocity: object,
    point_offset: object,
) -> Vector3:
    """Return `v(point) = v(reference) + omega x point_offset`."""
    velocity = _vector(reference_velocity, "reference_velocity")
    angular = _vector(angular_velocity, "angular_velocity")
    offset = _vector(point_offset, "point_offset")
    return _tuple(velocity + np.cross(angular, offset))


def transform_vector(vector: object, source_to_destination: object) -> Vector3:
    """Apply a proper orthonormal source-to-destination rotation."""
    value = _vector(vector, "vector")
    rotation = np.asarray(source_to_destination, dtype=float)
    if rotation.shape != (3, 3) or not bool(np.all(np.isfinite(rotation))):
        raise ValueError("source_to_destination must be a finite 3x3 matrix")
    orthogonality = rotation @ rotation.T
    if not np.allclose(orthogonality, np.eye(3), atol=1e-10) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-10
    ):
        raise ValueError("source_to_destination must be a proper orthonormal rotation")
    return _tuple(rotation @ value)
