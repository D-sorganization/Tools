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
    _near_zero,
    # SO(3)
    VecToso3,
    so3ToVec,
    MatrixExp3,
    MatrixLog3,
    # SE(3)
    VecTose3,
    se3ToVec,
    TransToRp,
    RpToTrans,
    TransInv,
    _Adjoint,
    Adjoint,
    MatrixExp6,
    MatrixLog6,
    # Kinematics
    FKinSpace,
    FKinBody,
    JacobianSpace,
    JacobianBody,
    IKinBody,
    IKinSpace,
    # Trajectory
    ScrewTrajectory,
    CubicTimeScaling,
    QuinticTimeScaling,
    JointTrajectory,
    CartesianTrajectory,
    # Utils
    Normalize,
    RotInv,
    AxisAng3,
    ScrewToAxis,
    AxisAng6,
    ProjectToSO3,
    ProjectToSE3,
    DistanceToSO3,
    DistanceToSE3,
    TestIfSO3,
    TestIfSE3,
    # Dynamics
    ad,
    InverseDynamics,
    MassMatrix,
    VelQuadraticForces,
    GravityForces,
    EndEffectorForces,
    ForwardDynamics,
    EulerStep,
    InverseDynamicsTrajectory,
    ForwardDynamicsTrajectory,
    ComputedTorque,
    SimulateControl,
)
