# mypy: ignore-errors
"""Modern Robotics (Lynch & Park) core algorithms — public re-export shim.

This module was refactored from a single 2079-line file into three focused
submodules to comply with the 1200-line budget:

    _mr_rotation_matrices  — SO(3) / SE(3) helpers
    _mr_kinematics         — FK, Jacobians, IK, trajectory generation
    _mr_dynamics           — inverse/forward dynamics, control simulation

All public symbols are re-exported here so that existing callers remain
unaffected::

    from rotation_converter.modern_robotics import FKinSpace, InverseDynamics

Functions follow the textbook naming conventions for discoverability:
- SO(3): VecToso3, so3ToVec, MatrixExp3, MatrixLog3
- SE(3): VecTose3, se3ToVec, MatrixExp6, MatrixLog6, TransToRp, RpToTrans, TransInv
- FK: FKinSpace, FKinBody (product of exponentials)
- IK: IKinBody, IKinSpace (iterative Newton-Raphson)
- Jacobians: JacobianSpace, JacobianBody
- Trajectory: ScrewTrajectory, JointTrajectory, CartesianTrajectory
- Dynamics: InverseDynamics, ForwardDynamics, MassMatrix, ...
- Control: ComputedTorque, SimulateControl

References:
    Lynch, K.M. & Park, F.C. (2017). Modern Robotics: Mechanics,
    Planning, and Control. Cambridge University Press.
"""

from __future__ import annotations

# --- Dynamics ---------------------------------------------------------------
from rotation_converter._mr_dynamics import (  # noqa: F401
    ComputedTorque,
    EndEffectorForces,
    EulerStep,
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

# --- Kinematics (FK, Jacobians, IK, Trajectory) -----------------------------
from rotation_converter._mr_kinematics import (  # noqa: F401
    CartesianTrajectory,
    CubicTimeScaling,
    FKinBody,
    FKinSpace,
    IKinBody,
    IKinSpace,
    JacobianBody,
    JacobianSpace,
    JointTrajectory,
    QuinticTimeScaling,
    ScrewTrajectory,
    _cubic_time_scaling,
    _quintic_time_scaling,
)

# --- SO(3) / SE(3) helpers --------------------------------------------------
from rotation_converter._mr_rotation_matrices import (  # noqa: F401
    Adjoint,
    AxisAng3,
    AxisAng6,
    DistanceToSE3,
    DistanceToSO3,
    MatrixExp3,
    MatrixExp6,
    MatrixLog3,
    MatrixLog6,
    Normalize,
    ProjectToSE3,
    ProjectToSO3,
    RotInv,
    RpToTrans,
    ScrewToAxis,
    TestIfSE3,
    TestIfSO3,
    TransInv,
    TransToRp,
    VecTose3,
    VecToso3,
    _Adjoint,
    _near_zero,
    se3ToVec,
    so3ToVec,
)

__all__ = [
    # SO(3) helpers
    "VecToso3",
    "so3ToVec",
    "MatrixExp3",
    "MatrixLog3",
    # SE(3) helpers
    "VecTose3",
    "se3ToVec",
    "TransToRp",
    "RpToTrans",
    "TransInv",
    "MatrixExp6",
    "MatrixLog6",
    # Legacy utility
    "Normalize",
    "RotInv",
    "AxisAng3",
    "Adjoint",
    "ScrewToAxis",
    "AxisAng6",
    "ProjectToSO3",
    "ProjectToSE3",
    "DistanceToSO3",
    "DistanceToSE3",
    "TestIfSO3",
    "TestIfSE3",
    # Kinematics
    "FKinSpace",
    "FKinBody",
    "JacobianSpace",
    "JacobianBody",
    "IKinBody",
    "IKinSpace",
    "ScrewTrajectory",
    "CubicTimeScaling",
    "QuinticTimeScaling",
    "JointTrajectory",
    "CartesianTrajectory",
    # Dynamics
    "ad",
    "InverseDynamics",
    "MassMatrix",
    "VelQuadraticForces",
    "GravityForces",
    "EndEffectorForces",
    "ForwardDynamics",
    "EulerStep",
    "InverseDynamicsTrajectory",
    "ForwardDynamicsTrajectory",
    "ComputedTorque",
    "SimulateControl",
]
