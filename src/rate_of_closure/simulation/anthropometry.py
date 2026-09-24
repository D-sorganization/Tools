# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Golfer anthropometry and swing delivery generation (Linear D-7520, epic #4103).

Provides the canonical Dempster/Winter body segment scaling calculations and
deterministic generation of reproducible swing deliveries for movement-optimizer
integration into Rate of Closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rate_of_closure._contracts import require
from shared.python.swing_sim import reference
from shared.python.swing_sim.integration_grid import (
    DEFAULT_SWING_RK4_DT_S,
    effective_rk4_duration,
)
from shared.python.swing_sim.types import (
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
)

if TYPE_CHECKING:
    from rate_of_closure.simulation.optimized_swing import OptimizedSwingResult

STANDARD_ARM_LENGTH_FRACTION: float = 0.42
STANDARD_ARM_MASS_FRACTION: float = 0.10

_ARM_COM_FRACTION: float = 0.45
_ARM_INERTIA_SCALING: float = 1.0 / 12.0
_DEFAULT_SHAFT_LENGTH_M: float = 1.0
_DEFAULT_SHAFT_MASS_KG: float = 0.15
_DEFAULT_CLUBHEAD_MASS_KG: float = 0.20
_SHAFT_COM_FRACTION: float = 0.43
_DEFAULT_DAMPING_SHOULDER: float = 0.4
_DEFAULT_DAMPING_WRIST: float = 0.25

__all__ = [
    "DEFAULT_GOLFER_HEIGHT_M",
    "DEFAULT_GOLFER_MASS_KG",
    "GolferAnthropometry",
    "STANDARD_ARM_LENGTH_FRACTION",
    "STANDARD_ARM_MASS_FRACTION",
]

DEFAULT_GOLFER_HEIGHT_M: float = 1.75
DEFAULT_GOLFER_MASS_KG: float = 75.0


@dataclass(frozen=True)
class GolferAnthropometry:
    """Anthropometric parameters for scaling golfer mechanics (Dempster / Winter).

    Attributes:
        height_m: Total golfer height in meters [m].
        mass_kg: Total body mass in kilograms [kg].
        lead_arm_length_m: Optional explicit lead arm length [m].
        lead_arm_mass_kg: Optional explicit lead arm mass [kg].
        club_length_m: Optional club length override [m].
        clubhead_mass_kg: Optional clubhead mass override [kg].
    """

    height_m: float = DEFAULT_GOLFER_HEIGHT_M
    mass_kg: float = DEFAULT_GOLFER_MASS_KG
    lead_arm_length_m: float | None = None
    lead_arm_mass_kg: float | None = None
    club_length_m: float | None = None
    clubhead_mass_kg: float | None = None

    def __post_init__(self) -> None:
        require(
            math.isfinite(self.height_m) and self.height_m > 0.0,
            "height_m must be finite and positive",
            self.height_m,
        )
        require(
            math.isfinite(self.mass_kg) and self.mass_kg > 0.0,
            "mass_kg must be finite and positive",
            self.mass_kg,
        )
        for name in (
            "lead_arm_length_m",
            "lead_arm_mass_kg",
            "club_length_m",
            "clubhead_mass_kg",
        ):
            val = getattr(self, name)
            if val is not None:
                require(
                    math.isfinite(val) and val > 0.0,
                    f"{name} must be finite and positive",
                    val,
                )

    @property
    def effective_lead_arm_length_m(self) -> float:
        """Lead arm length [m] (hub to wrist)."""
        if self.lead_arm_length_m is not None:
            return float(self.lead_arm_length_m)
        return float(self.height_m * STANDARD_ARM_LENGTH_FRACTION)

    @property
    def effective_lead_arm_mass_kg(self) -> float:
        """Lumped arm mass [kg]."""
        if self.lead_arm_mass_kg is not None:
            return float(self.lead_arm_mass_kg)
        return float(self.mass_kg * STANDARD_ARM_MASS_FRACTION)

    @property
    def effective_club_length_m(self) -> float:
        """Club length [m] (wrist to clubhead)."""
        if self.club_length_m is not None:
            return float(self.club_length_m)
        return _DEFAULT_SHAFT_LENGTH_M

    @property
    def effective_clubhead_mass_kg(self) -> float:
        """Clubhead mass [kg]."""
        if self.clubhead_mass_kg is not None:
            return float(self.clubhead_mass_kg)
        return _DEFAULT_CLUBHEAD_MASS_KG

    def to_pendulum_parameters(self) -> PendulumParameters:
        """Compute physically consistent double-pendulum parameters."""
        m1 = self.effective_lead_arm_mass_kg
        l1 = self.effective_lead_arm_length_m
        lc1 = l1 * _ARM_COM_FRACTION
        i1_com = _ARM_INERTIA_SCALING * m1 * l1 * l1
        i1 = i1_com + m1 * lc1 * lc1

        l2 = self.effective_club_length_m
        ms = _DEFAULT_SHAFT_MASS_KG
        mh = self.effective_clubhead_mass_kg
        m2 = ms + mh
        shaft_com = l2 * _SHAFT_COM_FRACTION
        lc2 = (shaft_com * ms + l2 * mh) / m2
        shaft_inertia_com = (1.0 / 12.0) * ms * l2 * l2
        parallel_axis = ms * (shaft_com - lc2) ** 2 + mh * (l2 - lc2) ** 2
        i2_com = shaft_inertia_com + parallel_axis
        i2 = i2_com + m2 * lc2 * lc2

        return PendulumParameters(
            m1=m1,
            l1=l1,
            lc1=lc1,
            i1=i1,
            m2=m2,
            l2=l2,
            lc2=lc2,
            i2=i2,
            d1=_DEFAULT_DAMPING_SHOULDER,
            d2=_DEFAULT_DAMPING_WRIST,
        )

    def generate_delivery(
        self,
        duration_s: float = 1.5,
        dt: float = DEFAULT_SWING_RK4_DT_S,
        initial_theta1: float = -math.pi / 2.0,
        initial_theta2: float = 0.0,
        plane: PlaneOrientation | None = None,
    ) -> OptimizedSwingResult:
        """Generate a deterministic, reproducible swing delivery trajectory."""
        from rate_of_closure.simulation.optimized_swing import OptimizedSwingResult

        params = self.to_pendulum_parameters()
        effective_dur = effective_rk4_duration(duration_s, dt)
        n_steps = int(round(effective_dur / dt))
        time_s = np.linspace(0.0, effective_dur, n_steps + 1)

        plane_cfg = plane or PlaneOrientation()
        plane_r = reference.plane_rotation(
            plane_cfg.yaw_rad, plane_cfg.side_tilt_rad, plane_cfg.forward_tilt_rad
        )
        g_inplane = reference.in_plane_gravity(plane_r, 9.80665)

        init_state = PendulumState(
            theta1=initial_theta1, theta2=initial_theta2, omega1=0.0, omega2=0.0
        )
        states = reference.simulate(params, init_state, g_inplane, dt, n_steps)

        n = len(time_s)
        poses = np.zeros((n, 4, 4), dtype=float)
        twists = np.zeros((n, 6), dtype=float)
        joints = np.zeros((n, 3, 3), dtype=float)

        l1 = params.l1
        l2 = params.l2
        x_axis, normal, up_axis = plane_r[:, 0], plane_r[:, 1], plane_r[:, 2]

        for i in range(n):
            t1, t2, w1, w2 = (
                float(states[i, 0]),
                float(states[i, 1]),
                float(states[i, 2]),
                float(states[i, 3]),
            )

            p_wrist = l1 * math.sin(t1) * x_axis - l1 * math.cos(t1) * up_axis
            t_total = t1 + t2
            p_head = (
                p_wrist
                + l2 * math.sin(t_total) * x_axis
                - l2 * math.cos(t_total) * up_axis
            )

            # In-plane rotation about normal (local Y)
            cy = math.cos(t_total)
            sy = math.sin(t_total)
            local_rot = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            rot = plane_r @ local_rot

            poses[i, :3, :3] = rot
            poses[i, :3, 3] = p_head
            poses[i, 3, 3] = 1.0

            w_total = w1 + w2
            v_wrist = (l1 * w1 * math.cos(t1)) * x_axis + (
                l1 * w1 * math.sin(t1)
            ) * up_axis
            v_head = (
                v_wrist
                + (l2 * w_total * math.cos(t_total)) * x_axis
                + (l2 * w_total * math.sin(t_total)) * up_axis
            )
            twists[i] = np.concatenate([w_total * normal, v_head])

            joints[i, 0] = np.zeros(3)
            joints[i, 1] = p_wrist
            joints[i, 2] = p_head

        return OptimizedSwingResult(
            time_s=time_s,
            clubhead_poses=poses,
            clubhead_twists=twists,
            frame_convention="swing_frame",
            joint_positions_m=joints,
            joint_ids=("hub", "wrist", "clubhead"),
            success=True,
        )
