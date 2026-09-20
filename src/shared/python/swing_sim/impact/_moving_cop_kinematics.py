"""Kinematics of moving Center of Pressure (COP) on curved golf club face.

Derives dynamic contact point, moving lever arms to club head COM,
work-conjugate contact load pairs, and dynamic torque (gear effect).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...golf_club._grip_contracts import Vector6, finite_array, vector6
from ...golf_club._shaft_point_load import SpatialPointLoad
from ...golf_club._validation import (
    Vector3,
    require_finite_float,
    require_vector3,
)
from ._curved_contact_geometry import NonSphericalContactGeometry
from ._spatial_contact_kinematics import ContactBodyState, ContactPairLoad

_ZERO_COUPLE: Vector3 = (0.0, 0.0, 0.0)


def _vector(val: object, name: str) -> Vector3:
    return require_vector3(finite_array(val, (3,), name), name)


@dataclass(frozen=True)
class MovingCOPKinematics:
    """Kinematics snapshot of moving Center of Pressure (COP) on a curved face.

    Preserves the fundamental law that force power is work-conjugate to
    material surface velocities, not geometric point migration.
    """

    face: ContactBodyState
    ball: ContactBodyState
    geometry: NonSphericalContactGeometry
    head_com_material_offset_m: Vector3 = (0.0, 0.0, 0.0)

    gap_m: float = field(init=False)
    gap_rate_mps: float = field(init=False)
    normal: Vector3 = field(init=False)
    contact_point_m: Vector3 = field(init=False)
    cop_material_m: Vector3 = field(init=False)
    relative_velocity_mps: Vector3 = field(init=False)
    face_offset_m: Vector3 = field(init=False)
    ball_offset_m: Vector3 = field(init=False)
    lever_arm_to_head_com_m: Vector3 = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.face, ContactBodyState) or not isinstance(
            self.ball, ContactBodyState
        ):
            raise TypeError("face and ball must be ContactBodyState")
        if not isinstance(self.geometry, NonSphericalContactGeometry):
            raise TypeError("geometry must be NonSphericalContactGeometry")
        if self.face.observer_id != self.ball.observer_id:
            raise ValueError("contact bodies must use the same observer")

        head_com_offset = _vector(
            self.head_com_material_offset_m, "head_com_material_offset_m"
        )
        object.__setattr__(self, "head_com_material_offset_m", head_com_offset)

        face_pose = np.asarray(self.face.pose)
        ball_pose = np.asarray(self.ball.pose)

        # Express ball center in face material frame
        r_face = face_pose[:3, :3]
        p_face = face_pose[:3, 3]
        ball_in_face = r_face.T @ (ball_pose[:3, 3] - p_face)

        # Project ball center onto curved face to find COP
        cop_x, cop_y, cop_z, norm_face = self.geometry.project_cop(
            float(ball_in_face[0]),
            float(ball_in_face[1]),
            float(ball_in_face[2]),
        )
        cop_mat = np.array([cop_x, cop_y, cop_z])

        # World / observer coordinates
        cop_world = p_face + r_face @ cop_mat
        normal_world = r_face @ np.asarray(norm_face)

        # Distance along outward normal to ball center
        ball_to_cop = ball_pose[:3, 3] - cop_world
        distance = float(np.dot(normal_world, ball_to_cop))
        gap = distance - self.geometry.nominal_ball_radius_m

        # Offsets in material coordinates
        face_offset = _vector(cop_mat, "face offset")
        ball_offset = _vector(
            np.asarray(self.ball.pose)[:3, :3].T
            @ (cop_world - np.asarray(self.ball.pose)[:3, 3]),
            "ball offset",
        )

        # Material surface velocities at COP
        v_face_mat = self.face.point_velocity(face_offset)
        v_ball_mat = self.ball.point_velocity(ball_offset)
        v_rel = v_ball_mat - v_face_mat
        gap_rate = float(np.dot(normal_world, v_rel))

        # Lever arm from club head COM to COP in observer frame
        head_com_world = p_face + r_face @ np.asarray(head_com_offset)
        lever_arm = cop_world - head_com_world

        for name, val in [
            ("normal", normal_world),
            ("contact_point_m", cop_world),
            ("cop_material_m", cop_mat),
            ("relative_velocity_mps", v_rel),
            ("face_offset_m", face_offset),
            ("ball_offset_m", ball_offset),
            ("lever_arm_to_head_com_m", lever_arm),
        ]:
            object.__setattr__(self, name, _vector(val, name))

        object.__setattr__(self, "gap_m", require_finite_float(gap, "gap"))
        object.__setattr__(
            self, "gap_rate_mps", require_finite_float(gap_rate, "gap rate")
        )

    @property
    def is_in_contact(self) -> bool:
        """True when normal gap is compressed (gap <= 0)."""
        return self.gap_m <= 0.0

    def load_pair(self, force_on_ball_n: object) -> ContactPairLoad:
        """Map contact force on the ball to work-conjugate wrenches on both bodies.

        Force is resolved in observer coordinates [N].
        """
        force = finite_array(force_on_ball_n, (3,), "contact force")
        ball_load = SpatialPointLoad(
            _vector(force, "force"), _ZERO_COUPLE, self.ball_offset_m
        )
        ball_wrench = vector6(ball_load.wrench(self.ball.pose), "ball wrench")

        # Dynamic torque about head COM
        torque_about_com = np.cross(np.asarray(self.lever_arm_to_head_com_m), -force)
        # Combine into face wrench (linear reaction force, torque about COM)
        modified_face_wrench: Vector6 = (
            float(-force[0]),
            float(-force[1]),
            float(-force[2]),
            float(torque_about_com[0]),
            float(torque_about_com[1]),
            float(torque_about_com[2]),
        )

        # Power = F . v_rel_mat strictly using material velocities
        power = float(np.dot(force, np.asarray(self.relative_velocity_mps)))

        return ContactPairLoad(
            ball_wrench=ball_wrench,
            face_wrench=modified_face_wrench,
            power_w=require_finite_float(power, "contact power"),
        )


__all__ = ("MovingCOPKinematics",)
