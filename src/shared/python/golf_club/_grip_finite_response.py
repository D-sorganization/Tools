"""Private coordinate-potential grip law with finite-pose physical work ports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import Matrix6, Vector6, factor6, vector6
from ._grip_energy import CoordinateImpedanceResponse, coordinate_impedance
from ._grip_moving_kinematics import (
    MaterialPointMotion,
    MovingGripKinematics,
    moving_grip_kinematics,
)
from ._validation import require_identifier


@dataclass(frozen=True)
class FinitePoseGrip:
    """Constant Gram factors in actual separation/principal rotation coordinates.

    The supplied anchor pose defines zero displacement and material coefficient
    axes. Coefficients are ideal coordinate inertance/damping/stiffness, not
    identified hand mass or a pressure law. Source identity does not establish
    calibration, bandwidth, loaded equilibrium or coupled-system stability.
    """

    inertance_factor: Matrix6
    damping_factor: Matrix6
    stiffness_factor: Matrix6
    source_id: str

    def __post_init__(self) -> None:
        for name in ("inertance_factor", "damping_factor", "stiffness_factor"):
            object.__setattr__(self, name, factor6(getattr(self, name), name))
        object.__setattr__(
            self, "source_id", require_identifier(self.source_id, "source_id")
        )


@dataclass(frozen=True)
class FiniteGripResponse:
    """Owned material reactions and observer-specific port powers in SI units.

    Both reactions act from the grip on their attached bodies. Storage and
    dissipation belong to relative coordinates. Individual port power is
    observer dependent; an inertial energy ledger requires inertial motion
    states or separate moving-frame work accounting. No trajectory is solved.
    """

    source_id: str
    root_wrench: Vector6
    anchor_wrench: Vector6
    storage: CoordinateImpedanceResponse
    root_power_w: float
    anchor_power_w: float

    @property
    def power_residual_w(self) -> float:
        """Numerical closure, never counted as physical damping."""
        return float(
            self.root_power_w
            + self.anchor_power_w
            + self.storage.stored_energy_rate_w
            + self.storage.dissipated_power_w
        )


def finite_grip_response(
    grip: FinitePoseGrip, root: MaterialPointMotion, anchor: MaterialPointMotion
) -> FiniteGripResponse:
    """Evaluate g=M qdd+C qd+K q and reactions -Ar.T g, -Aa.T g.

    The finite kinematic contract checks poses, observers and chart domain.
    Nonfinite algebra is refused; singular passive coefficients are allowed.
    This separate law does not reinterpret the existing small-rotation API.
    """
    if not isinstance(grip, FinitePoseGrip):
        raise TypeError("grip must be FinitePoseGrip")
    motion = moving_grip_kinematics(root, anchor)
    return _response_from_kinematics(grip, root, anchor, motion)


def _response_from_kinematics(
    grip: FinitePoseGrip,
    root: MaterialPointMotion,
    anchor: MaterialPointMotion,
    motion: MovingGripKinematics,
) -> FiniteGripResponse:
    """Evaluate the common law from already validated relative kinematics.

    Internal callers must supply motion from these same root/anchor states,
    optionally adding the exact root-acceleration contribution Ar*a. All
    effort, energy and power outputs retain their finite-value contracts.
    """
    storage = coordinate_impedance(
        (
            np.asarray(grip.inertance_factor),
            np.asarray(grip.damping_factor),
            np.asarray(grip.stiffness_factor),
        ),
        (motion.displacement, motion.velocity, motion.acceleration),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        root_wrench = vector6(-motion.root_motion_map.T @ storage.effort, "root wrench")
        anchor_wrench = vector6(
            -motion.anchor_motion_map.T @ storage.effort, "anchor wrench"
        )
        result = FiniteGripResponse(
            grip.source_id,
            root_wrench,
            anchor_wrench,
            storage,
            float(np.dot(root_wrench, np.asarray(root.twist))),
            float(np.dot(anchor_wrench, np.asarray(anchor.twist))),
        )
    if not np.all(
        np.isfinite(
            (result.root_power_w, result.anchor_power_w, result.power_residual_w)
        )
    ):
        raise ValueError("finite grip power must be finite")
    return result


__all__ = ()
