"""Constraint-projected integration for the rotating-base tier."""

from __future__ import annotations

import numpy as np

from ._numeric import N_COORDINATES, FloatArray, _direction_derivative
from .dynamics import (
    control_generalized_force,
    distal_segment_kinetic_energy,
    mass_matrix,
    mechanical_energy,
    solve_constrained_dynamics,
)
from .kinematics import _points, constraint_jacobian, constraint_vector
from .types import (
    ControlLaw,
    RotatingBaseConfig,
    RotatingBaseParams,
    RotatingBaseState,
    RotatingBaseTrace,
    TorsoTwoHandControl,
)


def initial_state(
    params: RotatingBaseParams,
    *,
    torso_angle_rad: float = 0.0,
    torso_rate_rad_s: float = 0.0,
    club_rate_rad_s: float = 0.0,
) -> RotatingBaseState:
    """Return an exactly closed, velocity-consistent reference state."""
    alpha = 0.5 * np.pi
    shoulder_x = abs(params.lead_shoulder_offset_m[0])
    grip_x = abs(params.lead_grip_offset_m)
    horizontal = shoulder_x - grip_x
    if horizontal >= params.arm_length_m:
        raise ValueError("arm length must reach the separated grips")
    center_y = -float(np.sqrt(params.arm_length_m**2 - horizontal**2))
    center = np.array([0.0, center_y])
    q = np.array([torso_angle_rad, 0.0, 0.0, *center, alpha, 0.0])
    points = _points(q, params)
    for index, side in ((1, "lead"), (2, "trail")):
        vector = points[f"{side}_grip"] - points[f"{side}_shoulder"]
        absolute_angle = float(np.arctan2(vector[0], -vector[1]))
        q[index] = absolute_angle - torso_angle_rad
    fixed = np.array([0, 5, 6])
    unknown = np.array([1, 2, 3, 4])
    qdot = np.zeros(N_COORDINATES)
    qdot[fixed] = [torso_rate_rad_s, club_rate_rad_s, 0.0]
    jacobian = constraint_jacobian(q, params)
    qdot[unknown] = np.linalg.solve(
        jacobian[:, unknown], -jacobian[:, fixed] @ qdot[fixed]
    )
    state = RotatingBaseState(q, qdot)
    if np.linalg.norm(constraint_vector(q, params)) > 1e-10:
        raise ValueError("reference configuration did not close")
    return state


def _project_configuration(
    q: FloatArray, params: RotatingBaseParams, config: RotatingBaseConfig
) -> tuple[FloatArray, float]:
    projected = q.copy()
    correction_norm = 0.0
    for _ in range(config.maximum_projection_iterations):
        residual = constraint_vector(projected, params)
        if np.linalg.norm(residual) <= config.projection_tolerance_m:
            return projected, correction_norm
        jacobian = constraint_jacobian(projected, params)
        inverse_mass = np.linalg.inv(mass_matrix(projected, params))
        correction = (
            -inverse_mass
            @ jacobian.T
            @ np.linalg.solve(jacobian @ inverse_mass @ jacobian.T, residual)
        )
        projected += correction
        correction_norm += float(np.linalg.norm(correction))
    raise ValueError("configuration projection did not converge")


def _project_velocity(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    jacobian = constraint_jacobian(q, params)
    inverse_mass = np.linalg.inv(mass_matrix(q, params))
    return qdot - inverse_mass @ jacobian.T @ np.linalg.solve(
        jacobian @ inverse_mass @ jacobian.T, jacobian @ qdot
    )


def _point_velocity_jacobians(
    q: FloatArray, params: RotatingBaseParams
) -> tuple[FloatArray, FloatArray, FloatArray]:
    alpha = q[5]
    lead_grip = np.zeros((2, N_COORDINATES))
    trail_grip = np.zeros((2, N_COORDINATES))
    lead_grip[:, 3:5] = np.eye(2)
    trail_grip[:, 3:5] = np.eye(2)
    lead_grip[:, 5] = params.lead_grip_offset_m * _direction_derivative(alpha)
    trail_grip[:, 5] = params.trail_grip_offset_m * _direction_derivative(alpha)
    center = np.zeros((2, N_COORDINATES))
    center[:, 3:5] = np.eye(2)
    beta = q[6]
    clubhead = center.copy()
    clubhead[:, 5] = params.proximal_club_length_m * _direction_derivative(
        alpha
    ) + params.distal_club_length_m * _direction_derivative(alpha + beta)
    clubhead[:, 6] = params.distal_club_length_m * _direction_derivative(alpha + beta)
    return lead_grip, trail_grip, clubhead


def rollout(
    initial: RotatingBaseState,
    control_law: ControlLaw,
    params: RotatingBaseParams,
    config: RotatingBaseConfig,
) -> RotatingBaseTrace:
    """Integrate and return a constraint- and energy-audited trajectory."""
    samples = config.interval_count + 1
    time = np.linspace(0.0, config.duration_s, samples)
    q = np.empty((samples, N_COORDINATES))
    qdot = np.empty_like(q)
    qddot = np.empty_like(q)
    q[0], qdot[0] = initial.q, initial.qdot
    projection_energy = np.zeros(samples)
    controls: list[TorsoTwoHandControl] = []
    for index in range(samples - 1):
        state = RotatingBaseState(q[index], qdot[index])
        control = control_law(float(time[index]), state)
        solution = solve_constrained_dynamics(state, control, params)
        controls.append(control)
        qddot[index] = solution.qddot
        trial_velocity = qdot[index] + config.step_s * solution.qddot
        trial_q = q[index] + config.step_s * trial_velocity
        before = mechanical_energy(RotatingBaseState(trial_q, trial_velocity), params)
        q[index + 1], _ = _project_configuration(trial_q, params, config)
        projected_velocity = _project_velocity(q[index + 1], trial_velocity, params)
        after = mechanical_energy(
            RotatingBaseState(q[index + 1], projected_velocity), params
        )
        qdot[index + 1] = projected_velocity
        projection_energy[index + 1] = after - before
    final_state = RotatingBaseState(q[-1], qdot[-1])
    controls.append(control_law(float(time[-1]), final_state))
    qddot[-1] = solve_constrained_dynamics(final_state, controls[-1], params).qddot

    forces = np.empty((samples, 2, 2))
    couples = np.empty(samples)
    contact_power = np.empty(samples)
    identity_residual = np.empty(samples)
    clubhead_velocity = np.empty((samples, 2))
    energy = np.empty(samples)
    distal_energy = np.empty(samples)
    control_power = np.empty(samples)
    dissipation_power = np.empty(samples)
    position_residual = np.empty(samples)
    velocity_residual = np.empty(samples)
    for index, control in enumerate(controls):
        state = RotatingBaseState(q[index], qdot[index])
        solution = solve_constrained_dynamics(state, control, params)
        forces[index] = solution.force_on_club_n
        couples[index] = solution.force_generated_couple_nm
        lead_jacobian, trail_jacobian, clubhead_jacobian = _point_velocity_jacobians(
            state.q, params
        )
        grip_velocities = np.stack(
            (lead_jacobian @ state.qdot, trail_jacobian @ state.qdot)
        )
        contact_power[index] = float(np.sum(forces[index] * grip_velocities))
        resultant = np.sum(forces[index], axis=0)
        center_velocity = state.qdot[3:5]
        wrench_power = float(
            resultant @ center_velocity + couples[index] * state.qdot[5]
        )
        identity_residual[index] = contact_power[index] - wrench_power
        clubhead_velocity[index] = clubhead_jacobian @ state.qdot
        energy[index] = mechanical_energy(state, params)
        distal_energy[index] = distal_segment_kinetic_energy(state, params)
        control_power[index] = control_generalized_force(control) @ state.qdot
        dissipation_power[index] = -(
            params.torso_damping_nms_rad * state.qdot[0] ** 2
            + params.arm_damping_nms_rad * np.sum(state.qdot[1:3] ** 2)
            + params.shaft_damping_nms_rad * state.qdot[6] ** 2
        )
        position_residual[index] = np.linalg.norm(constraint_vector(state.q, params))
        velocity_residual[index] = np.linalg.norm(
            constraint_jacobian(state.q, params) @ state.qdot
        )
    expected_change = float(
        np.trapezoid(control_power + dissipation_power, x=time)
        + np.sum(projection_energy)
    )
    closure = float(energy[-1] - energy[0] - expected_change)
    return RotatingBaseTrace(
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        controls=tuple(controls),
        force_on_club_n=forces,
        force_generated_couple_nm=couples,
        contact_power_on_club_w=contact_power,
        contact_power_identity_residual_w=identity_residual,
        clubhead_velocity_m_s=clubhead_velocity,
        clubhead_speed_m_s=np.linalg.norm(clubhead_velocity, axis=1),
        distal_segment_kinetic_energy_j=distal_energy,
        mechanical_energy_j=energy,
        control_power_w=control_power,
        dissipation_power_w=dissipation_power,
        position_constraint_norm_m=position_residual,
        velocity_constraint_norm_m_s=velocity_residual,
        projection_energy_change_j=projection_energy,
        work_energy_closure_j=closure,
    )
