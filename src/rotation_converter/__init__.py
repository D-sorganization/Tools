"""
Rotation Converter
==================

Comprehensive converter between rotational representations including
quaternions, Euler angles, rotation matrices, axis-angle, Rodrigues vectors,
twists, and screw axis representations.
"""

__version__ = "1.0.0"

from rotation_converter.converter import Rotation, RotationConverter
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
from rotation_converter.twist_screw import (
    adjoint_representation,
    homogeneous_to_twist_angle,
    screw_to_twist,
    se3_matrix_to_twist_vector,
    twist_angle_to_homogeneous,
    twist_to_screw,
    twist_vector_to_se3_matrix,
)

__all__ = [
    "Rotation",
    "RotationConverter",
    "adjoint_representation",
    "axis_angle_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "euler_to_quaternion",
    "euler_to_rotation_matrix",
    "homogeneous_to_twist_angle",
    "normalize_quaternion",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_euler",
    "quaternion_to_rodrigues",
    "quaternion_to_rotation_matrix",
    "rodrigues_to_quaternion",
    "rotation_matrix_to_axis_angle",
    "rotation_matrix_to_euler",
    "rotation_matrix_to_quaternion",
    "screw_to_twist",
    "se3_matrix_to_twist_vector",
    "twist_angle_to_homogeneous",
    "twist_to_screw",
    "twist_vector_to_se3_matrix",
]
