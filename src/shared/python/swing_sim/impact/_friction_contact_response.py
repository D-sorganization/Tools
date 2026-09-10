"""Shared mechanical and constitutive evaluation without mutating history."""

from dataclasses import dataclass

import numpy as np

from ...golf_club._validation import Vector3, require_vector3
from ._friction_trajectory_contracts import (
    FrictionTrajectoryProblem,
    FrictionTrajectoryState,
)
from ._normal_contact_work import NormalContactWork, normal_contact_work
from ._normal_shaft_contact import NormalShaftContact, ShaftBallContactResponse
from ._spatial_contact_kinematics import PlaneSphereKinematics


@dataclass(frozen=True)
class FrictionContactResponse:
    """Bodies and normal storage; tangential storage stays with owned history."""

    bodies: ShaftBallContactResponse
    normal: NormalContactWork
    tangential_force_n: Vector3

    @property
    def rates(self) -> np.ndarray:
        bodies = self.bodies
        return np.vstack((bodies.shaft.twist_rates, bodies.ball.twist_rate))

    @property
    def mechanical_normal_energy_j(self) -> float:
        bodies = self.bodies
        return float(
            bodies.shaft.total_energy_j
            + bodies.ball.kinetic_energy_j
            + self.normal.elastic_energy_j
        )

    @property
    def work_powers_w(self) -> np.ndarray:
        shaft = self.bodies.shaft
        return np.array(
            [
                self.bodies.external_power_w,
                shaft.anchor_power_w,
                shaft.dissipated_power_w,
                self.normal.viscous_power_w,
                self.normal.cutoff_power_w,
            ]
        )


def friction_response(
    model: NormalShaftContact,
    state: FrictionTrajectoryState,
    contact: PlaneSphereKinematics,
    ball_frame: str,
) -> FrictionContactResponse:
    """Use the accepted/trial elastic effort with canonical common-point loads."""
    normal = normal_contact_work(model.law, -contact.gap_m, -contact.gap_rate_mps)
    history = state.tangential
    tangent = -history.law.stiffness_n_per_m * np.asarray(history.elastic_deflection_m)
    force = normal.force_n * np.asarray(contact.normal) + tangent
    bodies = model._body_response(state.mechanical.shaft, contact, force, ball_frame)
    return FrictionContactResponse(
        bodies, normal, require_vector3(tangent, "friction force")
    )


def initial_response(
    problem: FrictionTrajectoryProblem, state: FrictionTrajectoryState, time_s: float
) -> FrictionContactResponse:
    """Refuse an inadmissible initial history instead of silently erasing storage."""
    model = problem.normal.contact_at(time_s)
    contact = model.kinematics(state.mechanical.shaft, state.mechanical.ball)
    history = state.tangential
    if history.law != problem.tangential_law:
        raise ValueError("initial tangential law must match the problem")
    if not np.allclose(np.asarray(history.normal), contact.normal, rtol=0, atol=1e-12):
        raise ValueError("initial tangential normal must match contact geometry")
    response = friction_response(
        model, state, contact, problem.normal.ball_material_frame_id
    )
    cap = problem.tangential_law.friction_coefficient * response.normal.force_n
    if np.linalg.norm(response.tangential_force_n) > cap:
        raise ValueError("initial tangential history exceeds the Coulomb cap")
    return response


__all__ = ()
