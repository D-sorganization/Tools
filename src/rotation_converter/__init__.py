"""
Rotation Converter
==================

Comprehensive, reusable library for converting between rotational and
rigid-body representations.  Designed for use as a shared tool — it can
be imported by any project in the Tools monorepo **or** pip-installed
standalone (``pip install .`` from ``src/rotation_converter/``).

Representations
---------------
- **Quaternion** (w, x, y, z) — Hamilton convention, unit norm
- **Rotation matrix** — 3x3 SO(3)
- **Euler angles** — 12 conventions (xyz, zyx, zyz, …)
- **Axis-angle** — unit axis + scalar angle
- **Rodrigues vector** — axis * angle compact form
- **Twist** (6-vector) — [omega; v] in se(3)
- **Screw axis** — {axis, point, pitch} geometric parameterisation
- **SE(3) homogeneous matrix** — 4x4 rigid-body transform
- **Frame-aware SE(3)** — ``RigidTransform`` with source/target labels

Key Classes
-----------
``Rotation``
    Immutable rotation object (quaternion hub).  Construct from any
    representation, output any representation.

``RigidTransform``
    Frame-aware SE(3) wrapper.  Enforces frame chain compatibility on
    composition (``FrameError``), provides body/space twist conversion,
    and point/vector transformations.

``RotationConverter``
    Static utility exposing all pairwise conversion functions.

Modern Robotics (Lynch & Park)
------------------------------
Product-of-exponentials FK/IK, space/body Jacobians, screw trajectory
generation — following textbook naming conventions.

Standalone Usage
----------------
::

    # Outside the monorepo
    cd src/rotation_converter
    pip install .            # core (numpy only)
    pip install .[viz]       # + matplotlib for animation
    pip install .[all]       # everything

The package carries its own Design-by-Contract shim
(``_contracts.py``) so it works with or without the monorepo's shared
``contracts`` module.
"""

__version__ = "1.3.0"

# ── High-level classes ────────────────────────────────────────────
from rotation_converter.converter import Rotation, RotationConverter

# ── Core rotation conversions ────────────────────────────────────
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

# ── Modern Robotics (Lynch & Park) ───────────────────────────────
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

# ── Visualization / examples ─────────────────────────────────────
from rotation_converter.motion_examples import (
    football_spiral,
    frisbee_flight,
)
from rotation_converter.rigid_transform import (
    FrameError,
    RigidTransform,
)
from rotation_converter.screw_visualization import (
    ScrewAxisAnimator,
    build_animation_frames,
    extract_screw_axes_from_trajectory,
)

# ── Twist / screw axis conversions ───────────────────────────────
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
    # Classes
    "FrameError",
    "RigidTransform",
    "Rotation",
    "RotationConverter",
    # Core conversions
    "axis_angle_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "euler_to_quaternion",
    "euler_to_rotation_matrix",
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
    # Twist / screw
    "adjoint_representation",
    "homogeneous_to_twist_angle",
    "screw_to_twist",
    "se3_matrix_to_twist_vector",
    "twist_angle_to_homogeneous",
    "twist_to_screw",
    "twist_vector_to_se3_matrix",
    # Modern Robotics
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
    # Visualization
    "ScrewAxisAnimator",
    "build_animation_frames",
    "extract_screw_axes_from_trajectory",
    "football_spiral",
    "frisbee_flight",
]
