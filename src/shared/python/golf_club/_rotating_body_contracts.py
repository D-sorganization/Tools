"""Frame-explicit instantaneous motion and body-tangent snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ._grip_contracts import finite_array
from ._validation import Vector3, require_identifier


def strict_vector3(value: object, name: str) -> Vector3:
    return cast(Vector3, tuple(float(x) for x in finite_array(value, (3,), name)))


@dataclass(frozen=True)
class RotatingFrameState:
    """Instantaneous motion of an observer frame, expressed in that frame.

    Origin acceleration is the physical inertial acceleration of the frame
    origin, not a derivative of its coordinates. Angular acceleration is the
    physical derivative of angular velocity. Units are rad/s, rad/s² and m/s².
    The frame's orientation is supplied separately when mapping results to world.
    """

    frame_id: str
    angular_velocity_rad_s: Vector3
    angular_acceleration_rad_s2: Vector3
    origin_acceleration_m_s2: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        for name in (
            "angular_velocity_rad_s",
            "angular_acceleration_rad_s2",
            "origin_acceleration_m_s2",
        ):
            object.__setattr__(self, name, strict_vector3(getattr(self, name), name))


@dataclass(frozen=True)
class _BodyTangent:
    """Independent coefficient arrays for local [translation, rotation] DOFs.

    Exp(theta) maps the reference body orientation into the observer frame.
    M qdd + G qd + (Kc + Ke + Ka)q = f0 + applied generalized wrench.
    Arrays accompany the exact frame sample and component identity. They are
    writable calculation snapshots, not calibrated interchange records.
    """

    body_id: str
    frame: RotatingFrameState
    reference_position_m: Vector3
    mass: np.ndarray
    velocity_position: np.ndarray
    gyroscopic: np.ndarray
    centrifugal_stiffness: np.ndarray
    euler_stiffness: np.ndarray
    acceleration_stiffness: np.ndarray
    equilibrium_force: np.ndarray
    model_name: str = "rotating_rigid_body_tangent/1"
    assumptions: tuple[str, ...] = (
        "first-order perturbations about zero relative displacement and velocity",
        "full COM inertia; reference body axes aligned with the observer frame",
        "origin and angular accelerations are prescribed, not solved",
        "equilibrium force must be balanced or retained as an actual forcing",
        "no shaft geometric stiffness, elastic stiffness, damping or contact",
        "no inferred calibration, stable equilibrium or finite-rotation validity band",
    )


__all__: list[str] = []
