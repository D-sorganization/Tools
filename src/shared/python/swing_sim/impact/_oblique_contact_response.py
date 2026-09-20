"""Coupled oblique contact response with moving COP, friction, and face/hosel modes.

Evaluates instantaneous normal compliance, modal deflection, tangential slip/friction,
body wrenches with dynamic gear effect, and strict multi-channel energy balance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...golf_club._grip_contracts import finite_array
from ...golf_club._validation import (
    Vector3,
    require_finite_float,
    require_vector3,
)
from ._curved_contact_geometry import NonSphericalContactGeometry
from ._face_hosel_modes import FaceHoselModalState, FaceHoselModalSystem
from ._moving_cop_kinematics import MovingCOPKinematics
from ._spatial_contact_kinematics import ContactBodyState, ContactPairLoad
from .contact import KelvinVoigtContactLaw


def _vector(val: object, name: str) -> Vector3:
    return require_vector3(finite_array(val, (3,), name), name)


@dataclass(frozen=True)
class ObliqueContactModel:
    """Coupled model with curved face, face/hosel modes, and contact laws."""

    geometry: NonSphericalContactGeometry
    modes: FaceHoselModalSystem
    normal_law: KelvinVoigtContactLaw
    friction_coefficient: float
    tangential_stiffness_n_per_m: float
    head_com_material_offset_m: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, NonSphericalContactGeometry):
            raise TypeError("geometry must be NonSphericalContactGeometry")
        if not isinstance(self.modes, FaceHoselModalSystem):
            raise TypeError("modes must be FaceHoselModalSystem")
        if not isinstance(self.normal_law, KelvinVoigtContactLaw):
            raise TypeError("normal_law must be KelvinVoigtContactLaw")

        mu = require_finite_float(self.friction_coefficient, "friction_coefficient")
        if mu < 0.0:
            raise ValueError("friction_coefficient must be nonnegative")
        object.__setattr__(self, "friction_coefficient", mu)

        kt = require_finite_float(
            self.tangential_stiffness_n_per_m,
            "tangential_stiffness_n_per_m",
            positive=True,
        )
        object.__setattr__(self, "tangential_stiffness_n_per_m", kt)

        offset = _vector(self.head_com_material_offset_m, "head_com_material_offset_m")
        object.__setattr__(self, "head_com_material_offset_m", offset)


@dataclass(frozen=True)
class ObliqueContactState:
    """Instantaneous state of the coupled system."""

    face: ContactBodyState
    ball: ContactBodyState
    modes: FaceHoselModalState
    tangential_deflection_m: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.face, ContactBodyState):
            raise TypeError("face must be ContactBodyState")
        if not isinstance(self.ball, ContactBodyState):
            raise TypeError("ball must be ContactBodyState")
        if not isinstance(self.modes, FaceHoselModalState):
            raise TypeError("modes must be FaceHoselModalState")
        defl = _vector(self.tangential_deflection_m, "tangential_deflection_m")
        object.__setattr__(self, "tangential_deflection_m", defl)


@dataclass(frozen=True)
class ObliqueContactResponse:
    """Instantaneous mechanical and constitutive response of oblique impact."""

    kinematics: MovingCOPKinematics
    normal_force_n: float
    friction_force_n: Vector3
    total_contact_force_on_ball_n: Vector3
    modal_accelerations: np.ndarray
    modal_deflection_m: float
    effective_gap_m: float
    load_pair: ContactPairLoad
    power_residual_w: float


def evaluate_oblique_contact(
    model: ObliqueContactModel, state: ObliqueContactState
) -> ObliqueContactResponse:
    """Compute instantaneous contact forces, modal dynamics, and energy rates."""
    kinematics = MovingCOPKinematics(
        face=state.face,
        ball=state.ball,
        geometry=model.geometry,
        head_com_material_offset_m=model.head_com_material_offset_m,
    )

    cop_xy = (
        float(kinematics.cop_material_m[0]),
        float(kinematics.cop_material_m[1]),
    )
    modal_defl = model.modes.modal_deflection_m(state.modes, cop_xy)
    modal_vel = model.modes.modal_velocity_mps(state.modes, cop_xy)

    # Effective gap accounts for flexible face trampoline / bending deformation
    effective_gap = kinematics.gap_m - modal_defl
    effective_gap_rate = kinematics.gap_rate_mps - modal_vel

    normal_unit = np.asarray(kinematics.normal)

    # Normal compliance: Kelvin-Voigt contact law
    if effective_gap < 0.0:
        compression = -effective_gap
        compression_rate = -effective_gap_rate
        raw_force = (
            model.normal_law.stiffness_n_per_m * compression
            + model.normal_law.damping_n_s_per_m * compression_rate
        )
        normal_force = max(0.0, float(raw_force))
    else:
        normal_force = 0.0

    # Tangential friction: stick-slip law with Coulomb cone
    tangential_defl = np.asarray(state.tangential_deflection_m)
    # Project tangential deflection orthogonal to normal
    tangential_defl = (
        tangential_defl - np.dot(tangential_defl, normal_unit) * normal_unit
    )

    trial_friction = -model.tangential_stiffness_n_per_m * tangential_defl
    coulomb_limit = model.friction_coefficient * normal_force
    trial_mag = float(np.linalg.norm(trial_friction))

    if trial_mag > coulomb_limit and trial_mag > 1e-12:
        friction_force = (coulomb_limit / trial_mag) * trial_friction
    else:
        friction_force = trial_friction

    total_force = normal_force * normal_unit + friction_force
    load_pair = kinematics.load_pair(total_force)

    # Modal dynamics
    gen_forces = model.modes.generalized_forces(cop_xy, normal_force)
    modal_accels = model.modes.modal_accelerations(state.modes, gen_forces)

    # Energy balance and power accounting
    # Contact power delivered to rigid bodies:
    p_mech = load_pair.power_w

    # Power channel decomposition:
    # P_mech = F_contact . v_rel_mat
    #        = (F_n * n + F_t) . (v_n * n + v_t)
    #        = F_n * v_n + F_t . v_t
    # Note that v_n = kinematics.gap_rate_mps = effective_gap_rate + modal_vel
    # Thus: F_n * v_n = F_n * modal_vel + F_n * effective_gap_rate
    # where F_n * modal_vel is the modal transfer power (Q . v_modal),
    # and F_n * effective_gap_rate is the normal contact spring/damper power.
    p_modal_transfer = normal_force * modal_vel
    p_normal_contact = normal_force * effective_gap_rate
    p_tangential_contact = float(
        np.dot(friction_force, np.asarray(kinematics.relative_velocity_mps))
    )

    power_residual = p_mech - (
        p_modal_transfer + p_normal_contact + p_tangential_contact
    )

    return ObliqueContactResponse(
        kinematics=kinematics,
        normal_force_n=normal_force,
        friction_force_n=_vector(friction_force, "friction force"),
        total_contact_force_on_ball_n=_vector(total_force, "total force"),
        modal_accelerations=modal_accels,
        modal_deflection_m=modal_defl,
        effective_gap_m=effective_gap,
        load_pair=load_pair,
        power_residual_w=require_finite_float(float(power_residual), "power residual"),
    )


__all__ = (
    "ObliqueContactModel",
    "ObliqueContactState",
    "ObliqueContactResponse",
    "evaluate_oblique_contact",
)
