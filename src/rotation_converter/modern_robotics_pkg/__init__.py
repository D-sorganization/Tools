"""Modern Robotics (Lynch & Park) core algorithms — submodule package.

This package decomposes the monolithic ``modern_robotics.py`` (2099 lines) into
domain-specific submodules while preserving every public function unchanged.

Submodules
----------
- ``_helpers``    — shared internal utilities (_near_zero)
- ``so3``         — SO(3) rotation helpers
- ``se3``         — SE(3) rigid-body transformation helpers
- ``kinematics``  — FK, IK, Jacobians
- ``trajectory``  — trajectory generation and time scaling
- ``dynamics``    — Newton-Euler inverse/forward dynamics, mass matrix, control
- ``utils``       — Normalize, ProjectToSO3/SE3, Distance/Test functions

References
----------
Lynch, K.M. & Park, F.C. (2017). *Modern Robotics: Mechanics, Planning,
and Control*. Cambridge University Press.
"""

# Re-export everything so ``from rotation_converter.modern_robotics_pkg import X``
# works the same as the old monolith.

from rotation_converter.modern_robotics_pkg._helpers import _near_zero  # noqa: F401
from rotation_converter.modern_robotics_pkg.dynamics import (  # noqa: F401
    ComputedTorque,
    EulerStep,
    EndEffectorForces,
    ForwardDynamics,
    ForwardDynamicsTrajectory,
    GravityForces,
    InverseDynamics,
    InverseDynamicsTrajectory,
    MassMatrix,
    SimulateControl,
    VelQuadraticForces,
    ad,
)
from rotation_converter.modern_robotics_pkg.kinematics import (  # noqa: F401
    FKinBody,
    FKinSpace,
    IKinBody,
    IKinSpace,
    JacobianBody,
    JacobianSpace,
)
from rotation_converter.modern_robotics_pkg.se3 import (  # noqa: F401
    Adjoint,
    MatrixExp6,
    MatrixLog6,
    RpToTrans,
    TransInv,
    TransToRp,
    VecTose3,
    _Adjoint,
    se3ToVec,
)
from rotation_converter.modern_robotics_pkg.so3 import (  # noqa: F401
    MatrixExp3,
    MatrixLog3,
    VecToso3,
    so3ToVec,
)
from rotation_converter.modern_robotics_pkg.trajectory import (  # noqa: F401
    CartesianTrajectory,
    CubicTimeScaling,
    JointTrajectory,
    QuinticTimeScaling,
    ScrewTrajectory,
)
from rotation_converter.modern_robotics_pkg.utils import (  # noqa: F401
    AxisAng3,
    AxisAng6,
    DistanceToSE3,
    DistanceToSO3,
    Normalize,
    ProjectToSE3,
    ProjectToSO3,
    RotInv,
    ScrewToAxis,
    TestIfSE3,
    TestIfSO3,
)
