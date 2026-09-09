"""Explicit inertial-observer states and moving grip ports for a nonlinear shaft."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import Vector6, _node_index, finite_array, vector6
from ._grip_finite_response import FinitePoseGrip
from ._grip_moving_kinematics import MaterialPointMotion
from ._shaft_gripped_operating import _owned_poses
from ._shaft_rotating_chain import RotatingSectionChain
from ._shaft_spectrum import SpectrumScales
from ._validation import require_identifier


@dataclass(frozen=True)
class MovingGripAttachment:
    """An ideal finite grip and prescribed anchor motion at a material node.

    This separate record permits nonzero anchor twist/rate. The existing
    stationary GripAttachment contract remains stationary. Anchor motion is
    supplied, not estimated from pressure, a player label or a pose alone.
    """

    node: int
    grip: FinitePoseGrip
    anchor: MaterialPointMotion

    def __post_init__(self) -> None:
        object.__setattr__(self, "node", _node_index(self.node))
        if not isinstance(self.grip, FinitePoseGrip):
            raise TypeError("grip must be FinitePoseGrip")
        if not isinstance(self.anchor, MaterialPointMotion):
            raise TypeError("anchor must be MaterialPointMotion")


@dataclass(frozen=True)
class InertialMovingChain:
    """Select an inertial observer and reuse explicit section/head quadratures.

    Construction prescribes an inertial observer for this model; it does not
    infer an observer history from a zero-motion frame snapshot. Nonzero
    observer angular velocity/acceleration or origin acceleration is refused.
    Loads retain their existing observer-resolved spatial force/couple laws.
    Every node is retained; zero or repeated grip ports are allowed.
    """

    shaft: RotatingSectionChain
    grips: tuple[MovingGripAttachment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shaft, RotatingSectionChain):
            raise TypeError("shaft must be RotatingSectionChain")
        frame = self.shaft.frame
        motion = (
            frame.angular_velocity_rad_s,
            frame.angular_acceleration_rad_s2,
            frame.origin_acceleration_m_s2,
        )
        if np.any(np.asarray(motion) != 0):
            raise ValueError("moving-chain model requires an inertial observer")
        grips = tuple(self.grips)
        for port in grips:
            if not isinstance(port, MovingGripAttachment):
                raise TypeError("grips must contain MovingGripAttachment")
            if port.node >= self.shaft.node_count:
                raise ValueError("grip node is outside the shaft")
            if port.anchor.observer_id != frame.frame_id:
                raise ValueError("grip and shaft observers must agree")
        object.__setattr__(self, "grips", grips)


@dataclass(frozen=True)
class MovingChainState:
    """Owned proper poses and physical body twists in a declared observer.

    Hdot=H hat(V), with linear-first V in m/s and rad/s. Accelerations are
    unknown outputs, not silently discarded input fields. Pose/twist history
    consistency is a separate trajectory or measurement obligation.
    """

    poses: object
    twists: object
    observer_id: str

    def __post_init__(self) -> None:
        shape = np.asarray(self.twists).shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != 6:
            raise ValueError("twists require one six-axis row per node")
        twists = finite_array(self.twists, shape, "body twists")
        object.__setattr__(self, "poses", _owned_poses(self.poses, shape[0]))
        object.__setattr__(
            self, "twists", tuple(tuple(float(x) for x in row) for row in twists)
        )
        object.__setattr__(
            self, "observer_id", require_identifier(self.observer_id, "observer")
        )


@dataclass(frozen=True)
class MovingChainControls:
    """Section strain bounds and dimensionless scaled mass/solve tolerances.

    Translation scaling is work-conjugate to moment/force coordinates. The
    time member of SpectrumScales is not used by this instantaneous solve.
    Strain bounds describe the supplied constitutive law, not material failure.
    """

    strain_limits: Vector6
    scales: SpectrumScales

    def __post_init__(self) -> None:
        limits = vector6(self.strain_limits, "strain limits")
        if np.any(np.asarray(limits) <= 0):
            raise ValueError("strain limits must be positive")
        object.__setattr__(self, "strain_limits", limits)
        if not isinstance(self.scales, SpectrumScales):
            raise TypeError("scales must be SpectrumScales")


__all__ = ()
