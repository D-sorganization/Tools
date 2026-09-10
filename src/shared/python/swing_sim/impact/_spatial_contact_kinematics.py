"""Candidate sphere/plane geometry with work-conjugate common-point loads.

This private port supplies kinematics, not an impact law, bounded face patch,
curved/deforming surface model, trajectory or physical qualification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...golf_club._grip_contracts import Vector6, finite_array, vector6
from ...golf_club._shaft_point_load import SpatialPointLoad
from ...golf_club._shaft_se3 import _rigid_pose
from ...golf_club._validation import (
    Vector3,
    require_finite_float,
    require_identifier,
    require_vector3,
)

_ZERO_COUPLE: Vector3 = (0.0, 0.0, 0.0)
_NORMAL_ROUNDOFF_TOLERANCE = 1e-12


def _vector(value: object, name: str) -> Vector3:
    return require_vector3(finite_array(value, (3,), name), name)


def _unit_normal(value: object, name: str) -> Vector3:
    """Normalize only unit-vector roundoff, never an arbitrary direction."""
    normal = np.asarray(_vector(value, name))
    length = np.linalg.norm(normal)
    if not np.isclose(length, 1, rtol=0, atol=_NORMAL_ROUNDOFF_TOLERANCE):
        raise ValueError(f"{name} must be unit length")
    return _vector(normal / length, name)


@dataclass(frozen=True)
class ContactBodyState:
    """Owned proper pose and linear-first physical body twist in one observer.

    Pose maps material axes into the observer; Hdot=H hat(twist). The linear
    velocity belongs to the pose origin, which need not be the mass center.
    No acceleration, mass tensor or measured-history consistency is inferred.
    """

    pose: object
    twist: object
    observer_id: str

    def __post_init__(self) -> None:
        pose = _rigid_pose(self.pose)
        object.__setattr__(
            self, "pose", tuple(tuple(float(x) for x in row) for row in pose)
        )
        object.__setattr__(self, "twist", vector6(self.twist, "contact body twist"))
        object.__setattr__(
            self, "observer_id", require_identifier(self.observer_id, "observer_id")
        )

    def point_velocity(self, material_offset: object) -> np.ndarray:
        """Return material-point velocity in observer axes, not point migration."""
        offset = finite_array(material_offset, (3,), "material offset")
        twist = np.asarray(self.twist)
        velocity = np.asarray(self.pose)[:3, :3] @ (
            twist[:3] + np.cross(twist[3:], offset)
        )
        return finite_array(velocity, (3,), "material point velocity")


@dataclass(frozen=True)
class PlaneSphereGeometry:
    """An oriented infinite face plane and a positive undeformed sphere radius.

    Plane point [m] and outward unit normal are in face material axes. The
    sphere's pose origin is its center. A unit normal is required to 1e-12;
    only that roundoff is normalized. No finite patch or edge law is assumed.
    """

    radius_m: float
    plane_point_m: object
    plane_normal: object

    def __post_init__(self) -> None:
        radius = require_finite_float(self.radius_m, "radius_m", positive=True)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(
            self, "plane_normal", _unit_normal(self.plane_normal, "plane normal")
        )
        object.__setattr__(
            self, "plane_point_m", _vector(self.plane_point_m, "plane point")
        )


@dataclass(frozen=True)
class ContactPairLoad:
    """Equal/opposite common-point body wrenches and their combined power [W]."""

    ball_wrench: Vector6
    face_wrench: Vector6
    power_w: float


def _projected_point(
    face: ContactBodyState, ball: ContactBodyState, geometry: PlaneSphereGeometry
) -> tuple[float, np.ndarray, np.ndarray]:
    face_pose, ball_pose = np.asarray(face.pose), np.asarray(ball.pose)
    normal = face_pose[:3, :3] @ geometry.plane_normal
    plane_point = face_pose[:3, 3] + face_pose[:3, :3] @ geometry.plane_point_m
    distance = normal @ (ball_pose[:3, 3] - plane_point)
    point = ball_pose[:3, 3] - distance * normal
    return float(distance - geometry.radius_m), normal, point


def _material_offset(body: ContactBodyState, point: np.ndarray) -> Vector3:
    pose = np.asarray(body.pose)
    return _vector(pose[:3, :3].T @ (point - pose[:3, 3]), "contact offset")


@dataclass(frozen=True)
class PlaneSphereKinematics:
    """Owned common-point contact snapshot derived from the two body states.

    Gap is positive at clearance. Contact point is the sphere-center projection
    onto the moving plane; its migration is not the face material velocity.
    A compression-dependent ball lever arm is intentional: applying both
    tangential reactions at this one point preserves total angular momentum.
    This convention approximates a compliant patch, not two separated rigid
    surface points. Its physical suitability needs deformation/patch evidence.
    """

    face: ContactBodyState
    ball: ContactBodyState
    geometry: PlaneSphereGeometry
    gap_m: float = field(init=False)
    gap_rate_mps: float = field(init=False)
    normal: Vector3 = field(init=False)
    contact_point_m: Vector3 = field(init=False)
    relative_velocity_mps: Vector3 = field(init=False)
    face_offset_m: Vector3 = field(init=False)
    ball_offset_m: Vector3 = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.face, ContactBodyState) or not isinstance(
            self.ball, ContactBodyState
        ):
            raise TypeError("face and ball must be ContactBodyState")
        if not isinstance(self.geometry, PlaneSphereGeometry):
            raise TypeError("geometry must be PlaneSphereGeometry")
        if self.face.observer_id != self.ball.observer_id:
            raise ValueError("contact bodies must use the same observer")
        gap, normal, point = _projected_point(self.face, self.ball, self.geometry)
        face_offset, ball_offset = (
            _material_offset(self.face, point),
            _material_offset(self.ball, point),
        )
        velocity = self.ball.point_velocity(ball_offset) - self.face.point_velocity(
            face_offset
        )
        for name, value in [
            ("normal", normal),
            ("contact_point_m", point),
            ("relative_velocity_mps", velocity),
            ("face_offset_m", face_offset),
            ("ball_offset_m", ball_offset),
        ]:
            object.__setattr__(self, name, _vector(value, name))
        object.__setattr__(self, "gap_m", require_finite_float(gap, "gap"))
        object.__setattr__(
            self,
            "gap_rate_mps",
            require_finite_float(float(normal @ velocity), "gap rate"),
        )

    def load_pair(self, force_n: object) -> ContactPairLoad:
        """Map force on the ball into both body wrenches at the common point.

        Force is observer-resolved [N]. No unilateral/friction law is applied.
        Recompute this snapshot at each contact stage. Existing point-load
        wrench/power maps are reused, but their fixed-material-point tangent
        must not be interpreted as the derivative of this migrating contact.
        """
        force = finite_array(force_n, (3,), "contact force")
        ball_load = SpatialPointLoad(
            _vector(force, "force"), _ZERO_COUPLE, self.ball_offset_m
        )
        face_load = SpatialPointLoad(
            _vector(-force, "reaction"), _ZERO_COUPLE, self.face_offset_m
        )
        ball_wrench = vector6(ball_load.wrench(self.ball.pose), "ball wrench")
        face_wrench = vector6(face_load.wrench(self.face.pose), "face wrench")
        power = ball_load.power(self.ball.pose, self.ball.twist) + face_load.power(
            self.face.pose, self.face.twist
        )
        return ContactPairLoad(
            ball_wrench, face_wrench, require_finite_float(power, "contact power")
        )


__all__ = ()
