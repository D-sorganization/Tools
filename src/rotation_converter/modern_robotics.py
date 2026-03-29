# mypy: ignore-errors
"""Modern Robotics (Lynch & Park) core algorithms — compatibility shim.

This module has been decomposed into ``rotation_converter.modern_robotics_pkg``
sub-modules (so3, se3, kinematics, dynamics, trajectory, utils).  All public
symbols are re-exported here so existing ``from rotation_converter.modern_robotics
import X`` imports continue to work unchanged.

See issue #1805.
"""

from __future__ import annotations

# Re-export every public symbol from the decomposed sub-package.
from rotation_converter.modern_robotics_pkg import (  # noqa: F401
    Adjoint,
    AxisAng3,
    AxisAng6,
    CartesianTrajectory,
    ComputedTorque,
    CubicTimeScaling,
    DistanceToSE3,
    DistanceToSO3,
    EndEffectorForces,
    EulerStep,
    FKinBody,
    # Kinematics
    FKinSpace,
    ForwardDynamics,
    ForwardDynamicsTrajectory,
    GravityForces,
    IKinBody,
    IKinSpace,
    InverseDynamics,
    InverseDynamicsTrajectory,
    JacobianBody,
    JacobianSpace,
    JointTrajectory,
    MassMatrix,
    MatrixExp3,
    MatrixExp6,
    MatrixLog3,
    MatrixLog6,
    # Utils
    Normalize,
    ProjectToSE3,
    ProjectToSO3,
    QuinticTimeScaling,
    RotInv,
    RpToTrans,
    ScrewToAxis,
    # Trajectory
    ScrewTrajectory,
    SimulateControl,
    TestIfSE3,
    TestIfSO3,
    TransInv,
    TransToRp,
    # SE(3)
    VecTose3,
    # SO(3)
    VecToso3,
    VelQuadraticForces,
    _Adjoint,
    _near_zero,
    # Dynamics
    ad,
    se3ToVec,
    so3ToVec,
)
