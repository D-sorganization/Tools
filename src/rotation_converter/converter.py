# mypy: disable-error-code="no-any-return"
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

from rotation_transforms.rotation import Rotation

from rotation_converter.core import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    euler_to_quaternion,
    euler_to_rotation_matrix,
    quaternion_to_axis_angle,
    quaternion_to_euler,
    quaternion_to_rodrigues,
    quaternion_to_rotation_matrix,
    rodrigues_to_quaternion,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)

__all__ = ["Rotation", "RotationConverter"]


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
