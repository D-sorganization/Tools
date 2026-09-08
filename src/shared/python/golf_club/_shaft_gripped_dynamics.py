"""Full-node frozen operators for grip-supported rotating shaft chains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._grip_stationary import stationary_grip_operators
from ._rotating_body_contracts import RotatingFrameState
from ._shaft_gripped_chain import GrippedSectionChain
from ._shaft_loaded_dynamics import linearized_chain_dynamics
from ._shaft_rotating_chain import RotatingSectionChain


@dataclass(frozen=True)
class GrippedChainDynamics:
    """Fresh r/M/G/C/K arrays and explicit observer motion at relative rest.

    The balance is r+M delta_acceleration+(G+C) delta_velocity+K delta_pose=0.
    G remains the shaft's gyroscopic transport; C contains the grip's passive
    damping. K includes its preload derivative. No nodes are eliminated and
    residual, singular mass, nonsymmetric stiffness and frame motion remain
    explicit. A driven-frame snapshot is not a stable autonomous trajectory.
    """

    residual: np.ndarray
    mass: np.ndarray
    gyroscopic: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    frame: RotatingFrameState

    @property
    def stability_status(self) -> str:
        return "unqualified"


def linearized_gripped_dynamics(
    chain: GrippedSectionChain, poses: object
) -> GrippedChainDynamics:
    """Compose verified distributed/body inertia and stationary finite grip ports.

    Missing shaft inertia is refused. Relative moving anchors require the full
    trajectory model and cannot be silently frozen by this constructor.
    """
    if not isinstance(chain, GrippedSectionChain) or not isinstance(
        chain.shaft, RotatingSectionChain
    ):
        raise TypeError("gripped dynamics require a RotatingSectionChain")
    current = finite_array(poses, (chain.node_count, 4, 4), "gripped dynamic poses")
    base = linearized_chain_dynamics(chain.shaft, current)
    size = 6 * chain.node_count
    damping = np.zeros((size, size))
    for item in chain.grips:
        port = stationary_grip_operators(
            item.grip, item.root_motion(current[item.node]), item.anchor
        )
        rows = slice(6 * item.node, 6 * (item.node + 1))
        base.residual[rows] -= port.response.root_wrench
        base.mass[rows, rows] += port.mass
        damping[rows, rows] += port.damping
        base.stiffness[rows, rows] += port.stiffness
    return GrippedChainDynamics(
        finite_array(base.residual, (size,), "gripped dynamic residual"),
        finite_array(base.mass, (size, size), "gripped mass"),
        base.gyroscopic,
        finite_array(damping, (size, size), "gripped damping"),
        finite_array(base.stiffness, (size, size), "gripped stiffness"),
        chain.shaft.frame,
    )


__all__ = ()
