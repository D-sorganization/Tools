"""Unified rotation converter API.

Provides two main interfaces:

1. ``Rotation`` — an immutable rotation object that can be constructed from
   any representation and output any representation. Internal storage is
   a unit quaternion (hub-and-spoke, DRY).

2. ``RotationConverter`` — a static utility class exposing all pairwise
   conversion functions for callers who prefer a functional/static style.

DbC: all factory methods validate inputs via preconditions; all output
methods guarantee postconditions (unit quaternion, SO(3), etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotation_converter._contracts import require, require_finite
from rotation_converter.core import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_euler,
    quaternion_to_rodrigues,
    quaternion_to_rotation_matrix,
    rodrigues_to_quaternion,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)


class Rotation:
    """Immutable rotation stored internally as a unit quaternion.

    Construct via factory methods (``from_*``), output via ``as_*`` accessors.
    All conversions route through the quaternion hub (DRY).

    Invariant: ``_q`` is always a unit quaternion with w >= 0.
    """

    __slots__ = ("_q",)

    def __init__(self, q: np.ndarray) -> None:
        """Private — use factory methods instead."""
        q = np.asarray(q, dtype=float)
        require(q.shape == (4,), "internal quaternion must have 4 elements")
        norm = np.linalg.norm(q)
        require(bool(abs(norm - 1.0) < 1e-6), "internal quaternion must be unit")
        # Canonical form
        if q[0] < 0:
            q = -q
        self._q = q

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def identity(cls) -> Rotation:
        """Create the identity rotation."""
        return cls(np.array([1.0, 0.0, 0.0, 0.0]))

    @classmethod
    def from_quaternion(cls, q: Any) -> Rotation:
        """Create from quaternion (w, x, y, z).

        Precondition: q has 4 elements, non-zero norm.
        """
        q = np.asarray(q, dtype=float)
        require(q.shape == (4,), "quaternion must have 4 elements", q.shape)
        require_finite(q, "quaternion")
        return cls(normalize_quaternion(q))

    @classmethod
    def from_rotation_matrix(cls, R: Any) -> Rotation:
        """Create from a 3x3 rotation matrix.

        Precondition: R is in SO(3).
        """
        R = np.asarray(R, dtype=float)
        require(R.shape == (3, 3), "rotation matrix must be 3x3", R.shape)
        q = rotation_matrix_to_quaternion(R)
        return cls(q)

    @classmethod
    def from_euler(cls, a: float, b: float, c: float, convention: str) -> Rotation:
        """Create from Euler angles.

        Args:
            a, b, c: angles in radians.
            convention: e.g. "xyz", "zyx", "zyz".
        """
        q = euler_to_quaternion(a, b, c, convention)
        return cls(q)

    @classmethod
    def from_axis_angle(cls, axis: Any, angle: float) -> Rotation:
        """Create from axis-angle.

        Precondition: axis is a unit vector.
        """
        axis = np.asarray(axis, dtype=float)
        require(axis.shape == (3,), "axis must have 3 elements", axis.shape)
        q = axis_angle_to_quaternion(axis, angle)
        return cls(q)

    @classmethod
    def from_rodrigues(cls, r: Any) -> Rotation:
        """Create from Rodrigues vector (axis * angle).

        Precondition: r has 3 elements.
        """
        r = np.asarray(r, dtype=float)
        require(r.shape == (3,), "Rodrigues vector must have 3 elements", r.shape)
        q = rodrigues_to_quaternion(r)
        return cls(q)

    # ── Output accessors ──────────────────────────────────────────

    def as_quaternion(self) -> np.ndarray:
        """Return unit quaternion (w, x, y, z)."""
        return self._q.copy()

    def as_rotation_matrix(self) -> np.ndarray:
        """Return 3x3 rotation matrix in SO(3)."""
        return quaternion_to_rotation_matrix(self._q)

    def as_euler(self, convention: str) -> tuple[float, float, float]:
        """Return Euler angles (a, b, c) for the given convention."""
        return quaternion_to_euler(self._q, convention)

    def as_axis_angle(self) -> tuple[np.ndarray, float]:
        """Return (axis, angle) where axis is unit and angle in [0, pi]."""
        return quaternion_to_axis_angle(self._q)

    def as_rodrigues(self) -> np.ndarray:
        """Return Rodrigues vector (axis * angle)."""
        return quaternion_to_rodrigues(self._q)

    # ── Composition ───────────────────────────────────────────────

    def compose(self, other: Rotation) -> Rotation:
        """Compose this rotation with another: self * other."""
        q = quaternion_multiply(self._q, other._q)
        return Rotation(normalize_quaternion(q))

    def inverse(self) -> Rotation:
        """Return the inverse rotation."""
        return Rotation(normalize_quaternion(quaternion_conjugate(self._q)))

    # ── Dunder methods ────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Rotation(q={self._q})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rotation):
            return NotImplemented
        # q and -q represent the same rotation
        return bool(np.allclose(self._q, other._q, atol=1e-10))


class RotationConverter:
    """Static utility class exposing all pairwise conversion functions.

    Thin delegation layer over the core module functions.
    Provides a single namespace for all conversions (DRY).
    """

    # Quaternion <-> Rotation Matrix
    quaternion_to_rotation_matrix = staticmethod(quaternion_to_rotation_matrix)
    rotation_matrix_to_quaternion = staticmethod(rotation_matrix_to_quaternion)

    # Quaternion <-> Euler
    euler_to_quaternion = staticmethod(euler_to_quaternion)
    quaternion_to_euler = staticmethod(quaternion_to_euler)

    # Quaternion <-> Axis-Angle
    axis_angle_to_quaternion = staticmethod(axis_angle_to_quaternion)
    quaternion_to_axis_angle = staticmethod(quaternion_to_axis_angle)

    # Quaternion <-> Rodrigues
    rodrigues_to_quaternion = staticmethod(rodrigues_to_quaternion)
    quaternion_to_rodrigues = staticmethod(quaternion_to_rodrigues)

    # Euler <-> Rotation Matrix
    euler_to_rotation_matrix = staticmethod(euler_to_rotation_matrix)
    rotation_matrix_to_euler = staticmethod(rotation_matrix_to_euler)

    # Axis-Angle <-> Rotation Matrix
    axis_angle_to_rotation_matrix = staticmethod(axis_angle_to_rotation_matrix)
    rotation_matrix_to_axis_angle = staticmethod(rotation_matrix_to_axis_angle)
