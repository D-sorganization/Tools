"""Immutable output contract for double-pendulum swing kinetics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from rate_of_closure._contracts import require

__all__ = ["CLUBHEAD_MASS_KG", "KINETIC_JOINT_NAMES", "KineticsSeries"]

#: Joint names in coordinate order (proximal to distal), mirroring the
#: movement optimizer's lowercase joint-name legend convention.
KINETIC_JOINT_NAMES: tuple[str, ...] = ("shoulder", "wrist")

#: Clubhead point mass [kg] used for the clubhead-force estimate — the
#: shared golf-default head mass (swing_sim.types _CLUBHEAD_MASS_KG).
#: The double-pendulum lumps shaft + head into segment 2, so the head
#: force is a point-mass estimate at the segment tip (documented
#: approximation).
CLUBHEAD_MASS_KG = 0.20


@dataclass(frozen=True)
class KineticsSeries:
    """Per-sample swing kinetics of one double-pendulum run.

    All arrays share the sample count N of the run's swing grid. Torque columns
    follow :data:`KINETIC_JOINT_NAMES` order and satisfy::

        torque_applied + torque_constraint_reaction
            = torque_inertial - torque_gravity - torque_damping

    ``torque_inertial`` is ``M(q)·qdd + C(q, qdot)``; gravity and damping
    arrays are the corresponding generalized driving torques. ZTCF fields are
    pointwise state-matched zero-command-torque counterfactuals, not one
    integrated alternate trajectory. Force and position vectors use the app
    frame (x target, y up, z right), while torque columns are ordered by
    :attr:`joint_names`.
    """

    t: np.ndarray = field(repr=False)
    joint_names: tuple[str, ...]
    torque_applied_nm: np.ndarray = field(repr=False)
    torque_constraint_reaction_nm: np.ndarray = field(repr=False)
    torque_gravity_nm: np.ndarray = field(repr=False)
    torque_damping_nm: np.ndarray = field(repr=False)
    torque_inertial_nm: np.ndarray = field(repr=False)
    ztcf_acceleration_rad_s2: np.ndarray = field(repr=False)
    ztcf_inertial_torque_nm: np.ndarray = field(repr=False)
    power_w: np.ndarray = field(repr=False)
    shoulder_force_n: np.ndarray = field(repr=False)
    wrist_force_n: np.ndarray = field(repr=False)
    clubhead_force_n: np.ndarray = field(repr=False)
    ztcf_shoulder_force_n: np.ndarray = field(repr=False)
    ztcf_wrist_force_n: np.ndarray = field(repr=False)
    ztcf_clubhead_force_n: np.ndarray = field(repr=False)
    pivot_position_m: np.ndarray = field(repr=False)
    wrist_positions_m: np.ndarray = field(repr=False)
    clubhead_positions_m: np.ndarray = field(repr=False)
    plane_x_app: np.ndarray = field(repr=False)
    plane_up_app: np.ndarray = field(repr=False)
    impact_time_s: float

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        j = len(self.joint_names)
        require(n >= 3, "kinetics needs at least 3 samples", n)
        require(j >= 2, "kinetics needs at least 2 joints", j)
        for name in (
            "torque_applied_nm",
            "torque_constraint_reaction_nm",
            "torque_gravity_nm",
            "torque_damping_nm",
            "torque_inertial_nm",
            "ztcf_acceleration_rad_s2",
            "ztcf_inertial_torque_nm",
            "power_w",
        ):
            require(
                getattr(self, name).shape == (n, j),
                f"{name} must be (N, {j})",
                getattr(self, name).shape,
            )
        for name in (
            "shoulder_force_n",
            "wrist_force_n",
            "clubhead_force_n",
            "ztcf_shoulder_force_n",
            "ztcf_wrist_force_n",
            "ztcf_clubhead_force_n",
            "wrist_positions_m",
            "clubhead_positions_m",
        ):
            require(
                getattr(self, name).shape == (n, 3),
                f"{name} must be (N, 3)",
                getattr(self, name).shape,
            )
        require(
            self.pivot_position_m.shape == (3,),
            "pivot_position_m must be a 3-vector",
            self.pivot_position_m.shape,
        )
        require(
            math.isfinite(self.impact_time_s) and self.impact_time_s >= 0.0,
            "impact_time_s must be finite and >= 0",
            self.impact_time_s,
        )

    def force_magnitude_n(self, which: str) -> np.ndarray:
        """Return the magnitude of one physical force series."""
        require(
            which in ("shoulder", "wrist", "clubhead"),
            "unknown force series",
            which,
        )
        vectors = getattr(self, f"{which}_force_n")
        return cast(
            np.ndarray, np.asarray(np.linalg.norm(vectors, axis=1), dtype=float)
        )

    def ztcf_force_magnitude_n(self, which: str) -> np.ndarray:
        """Return the state-matched ZTCF magnitude of one force series."""
        require(
            which in ("shoulder", "wrist", "clubhead"),
            "unknown ZTCF force series",
            which,
        )
        vectors = getattr(self, f"ztcf_{which}_force_n")
        return cast(
            np.ndarray, np.asarray(np.linalg.norm(vectors, axis=1), dtype=float)
        )
