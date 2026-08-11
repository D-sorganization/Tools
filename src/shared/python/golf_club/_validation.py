"""Private validation and immutable-normalization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, cast

import numpy as np

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]

_SYMMETRY_TOLERANCE = 1e-12
_PSD_TOLERANCE = 1e-12
_ROTATION_TOLERANCE = 1e-10


def require_identifier(value: object, name: str) -> str:
    """Return a nonempty, trimmed identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty and trimmed")
    return value


def require_finite_float(value: object, name: str, *, positive: bool = False) -> float:
    """Return a finite real number, optionally requiring strict positivity."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def require_vector3(value: object, name: str) -> Vector3:
    """Return a finite immutable three-vector."""
    array = _numeric_array(value, name)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(item) for item in array)  # type: ignore[return-value]


def require_matrix3(value: object, name: str) -> Matrix3:
    """Return a finite immutable 3-by-3 matrix."""
    array = _numeric_array(value, name)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values")
    rows = tuple(tuple(float(item) for item in row) for row in array)
    return rows  # type: ignore[return-value]


def require_rotation(value: object) -> Matrix3:
    """Return a proper orthonormal rotation matrix."""
    rotation = require_matrix3(value, "rotation")
    array = np.asarray(rotation)
    identity = np.eye(3)
    determinant = float(np.linalg.det(array))
    if not np.allclose(
        array.T @ array,
        identity,
        atol=_ROTATION_TOLERANCE,
        rtol=0.0,
    ) or not math.isclose(
        determinant,
        1.0,
        abs_tol=_ROTATION_TOLERANCE,
        rel_tol=0.0,
    ):
        raise ValueError("rotation must be a proper orthonormal 3x3 matrix")
    return rotation


def require_inertia(value: object) -> Matrix3:
    """Return a finite, symmetric, physically realizable inertia tensor."""
    inertia = require_matrix3(value, "inertia_at_com_kg_m2")
    array = np.asarray(inertia)
    if not np.allclose(array, array.T, atol=_SYMMETRY_TOLERANCE, rtol=0.0):
        raise ValueError("inertia_at_com_kg_m2 must be symmetric")
    principal_moments = np.linalg.eigvalsh(array)
    if float(np.min(principal_moments)) < -_PSD_TOLERANCE:
        raise ValueError("inertia_at_com_kg_m2 must be positive semidefinite")
    if float(principal_moments[2] - principal_moments[1] - principal_moments[0]) > (
        _PSD_TOLERANCE
    ):
        raise ValueError(
            "inertia_at_com_kg_m2 principal moments must satisfy the "
            "triangle inequality"
        )
    return inertia


def require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a string-keyed mapping boundary value."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def reject_unknown_fields(
    data: Mapping[str, Any], allowed: frozenset[str], name: str
) -> None:
    """Reject schema fields that this version does not understand."""
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


def _numeric_array(value: object, name: str) -> np.ndarray:
    """Convert an accepted numeric sequence without retaining caller storage."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(
        value, (Sequence, np.ndarray)
    ):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        return cast(np.ndarray, np.array(value, dtype=float, copy=True))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numbers") from error


__all__: list[str] = []
