"""
Rotation Converter
==================

Comprehensive converter between rotational representations including
quaternions, Euler angles, rotation matrices, axis-angle, Rodrigues vectors,
twists, screw axis representations, and Modern Robotics (Lynch & Park)
kinematics functions.
"""

__version__ = "1.3.0"

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
from rotation_converter.modern_robotics import (
    FKinBody,
    FKinSpace,
    IKinBody,
    JacobianBody,
    JacobianSpace,
    MatrixExp3,
    MatrixExp6,
    MatrixLog3,
    MatrixLog6,
    RpToTrans,
    ScrewTrajectory,
    TransInv,
    TransToRp,
    VecTose3,
    VecToso3,
    se3ToVec,
    so3ToVec,
)
from rotation_converter.rigid_transform import (
    FrameError,
    RigidTransform,
)
from rotation_converter.motion_examples import (
    football_spiral,
    frisbee_flight,
)
from rotation_converter.screw_visualization import (
    ScrewAxisAnimator,
    build_animation_frames,
    extract_screw_axes_from_trajectory,
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
    "FKinBody",
    "FKinSpace",
    "IKinBody",
    "JacobianBody",
    "JacobianSpace",
    "MatrixExp3",
    "MatrixExp6",
    "MatrixLog3",
    "MatrixLog6",
    "RpToTrans",
    "ScrewTrajectory",
    "TransInv",
    "TransToRp",
    "VecTose3",
    "VecToso3",
    "se3ToVec",
    "so3ToVec",
    "ScrewAxisAnimator",
    "build_animation_frames",
    "extract_screw_axes_from_trajectory",
    "FrameError",
    "RigidTransform",
    "football_spiral",
    "frisbee_flight",
]
