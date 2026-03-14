"""
Rotation Converter
==================

.. deprecated:: 1.4.0
   This pure-Python module is deprecated in favor of the ``math-primitives``
   Rust crate, exposed to Python via ``tools_core.math_primitives``.  The Rust
   implementation provides identical SE3/SO3 operations with zero-copy NumPy
   support and significantly higher throughput.

   Migration guide::

       # Before (deprecated)
       from rotation_converter.core import euler_to_quaternion

       # After (recommended)
       from tools_core.math_primitives import euler_to_quaternion

   The ``rotation_converter`` module will continue to work during the
   transition period but will be removed in a future release.

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

import warnings

warnings.warn(
    "rotation_converter is deprecated. "
    "Use tools_core.math_primitives for Rust-native SE3/SO3 operations. "
    "See issue #1255 for migration details.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "1.4.0"

# ── High-level classes ────────────────────────────────────────────
from rotation_converter.converter import Rotation, RotationConverter  # noqa: E402

# ── Core rotation conversions ────────────────────────────────────
from rotation_converter.core import (  # noqa: E402
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
from rotation_converter.modern_robotics import (  # noqa: E402
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
from rotation_converter.motion_examples import (  # noqa: E402
    football_spiral,
    frisbee_flight,
)
from rotation_converter.rigid_transform import (  # noqa: E402
    FrameError,
    RigidTransform,
)
from rotation_converter.screw_visualization import (  # noqa: E402
    ScrewAxisAnimator,
    build_animation_frames,
    extract_screw_axes_from_trajectory,
)

# ── Twist / screw axis conversions ───────────────────────────────
from rotation_converter.twist_screw import (  # noqa: E402
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
