"""Physical point-load work and its full fixed-chart SE(3) derivative.

Private loaded-state building block; no contact law or equilibrium solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_se3 import _rigid_pose, twist_ad
from ._validation import Vector3, require_finite_float, require_vector3


@dataclass(frozen=True)
class LoadLinearization:
    """Fresh canonical force and derivative at H(q)=H0 Exp(q), q=0.

    Linear-first coordinates are local translations [m] and rotations [rad].
    Tangent is dQ_external/dq; the residual uses K_internal minus this matrix.
    A physical couple may make this derivative nonsymmetric.
    """

    wrench: np.ndarray
    tangent: np.ndarray


@dataclass(frozen=True)
class SpatialPointLoad:
    """Configuration-independent spatial force and free couple at a body point.

    Force [N] and couple [N m] are resolved in the common observer frame;
    offset [m] is fixed in the section's material frame, from its pose origin.
    The couple is additional to the offset force's moment. This contract does
    not represent a body-following load or infer a hand/contact control law.
    """

    force_n: Vector3
    couple_nm: Vector3
    offset_m: Vector3

    def __post_init__(self) -> None:
        for name in ("force_n", "couple_nm", "offset_m"):
            # The legacy tuple normalizer accepts coercion; keep this boundary strict.
            numeric = finite_array(getattr(self, name), (3,), name)
            object.__setattr__(self, name, require_vector3(numeric, name))

    def _local_load(self, pose: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = _rigid_pose(pose)[:3, :3]
        force = rotation.T @ self.force_n
        couple = rotation.T @ self.couple_nm
        wrench = np.r_[force, couple + np.cross(self.offset_m, force)]
        return force, couple, finite_array(wrench, (6,), "point-load wrench")

    def linearize(self, pose: object) -> LoadLinearization:
        """Differentiate physical directions, lever arm and the canonical chart.

        No numerical differentiation or imposed symmetry is used. The force
        alone is conservative; a fixed spatial couple is generally not.
        """
        force, couple, wrench = self._local_load(pose)
        columns = []
        for axis in np.eye(6):
            dforce = -np.cross(axis[3:], force)
            dcouple = -np.cross(axis[3:], couple)
            physical = np.r_[dforce, dcouple + np.cross(self.offset_m, dforce)]
            columns.append(physical - 0.5 * twist_ad(axis).T @ wrench)
        return LoadLinearization(
            wrench,
            finite_array(np.column_stack(columns), (6, 6), "point-load tangent"),
        )

    def power(self, pose: object, body_twist: object) -> float:
        """Return instantaneous external power [W] for a local material twist.

        Its first three components are pose-origin velocity [m/s], and its
        last three are physical angular velocity [rad/s], in material axes.
        """
        velocity = finite_array(body_twist, (6,), "body velocity")
        return require_finite_float(
            self._local_load(pose)[2] @ velocity, "point-load power"
        )

    def force_potential(self, pose: object) -> float:
        """Return -f dot (p+R offset) [J], excluding all couple work.

        A constant spatial couple has no general scalar rotation potential.
        Its work must be integrated separately; this is never total energy.
        """
        current = _rigid_pose(pose)
        point = current[:3, 3] + current[:3, :3] @ self.offset_m
        return require_finite_float(
            -np.asarray(self.force_n) @ point, "force potential"
        )


__all__ = ()
