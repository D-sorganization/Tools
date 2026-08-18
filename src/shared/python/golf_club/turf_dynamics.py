"""Reduced effective-mass integration for the compliant-turf contact law."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._validation import Vector3, require_finite_float, require_vector3
from .turf_contact import (
    TurfContactKinematics,
    TurfContactProfile,
    TurfContactStatus,
    evaluate_turf_contact,
)

_UNIT_TOLERANCE = 1e-9
_CONTACT_TOLERANCE_M = 1e-12
_DEFAULT_MAX_TIME_S = 0.1
_DEFAULT_MAX_STEPS = 200_000
_LIMITATIONS = (
    "Reduced single-point effective-mass diagnostic only; the returned wrench "
    "is the integration seam for a full rigid-body/turf solver."
)


@dataclass(frozen=True)
class ReducedTurfContactResult:
    """Converged reduced effective-mass contact diagnostic."""

    status: TurfContactStatus
    duration_s: float
    step_count: int
    peak_penetration_m: float
    impulse_world_n_s: Vector3
    normal_impulse_n_s: float
    final_contact_velocity_mps: Vector3
    initial_kinetic_energy_j: float
    final_kinetic_energy_j: float
    dissipated_energy_j: float
    separation_loss_energy_j: float
    energy_balance_residual_j: float
    limitations: str = _LIMITATIONS


def _tuple3(values: np.ndarray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _result(
    status: TurfContactStatus,
    *,
    duration_s: float,
    step_count: int,
    peak_penetration_m: float,
    impulse: np.ndarray,
    normal: np.ndarray,
    velocity: np.ndarray,
    mass_kg: float,
    initial_energy_j: float,
    dissipated_energy_j: float,
    separation_loss_energy_j: float = 0.0,
) -> ReducedTurfContactResult:
    final_energy = 0.5 * mass_kg * float(velocity @ velocity)
    return ReducedTurfContactResult(
        status=status,
        duration_s=duration_s,
        step_count=step_count,
        peak_penetration_m=peak_penetration_m,
        impulse_world_n_s=_tuple3(impulse),
        normal_impulse_n_s=float(impulse @ normal),
        final_contact_velocity_mps=_tuple3(velocity),
        initial_kinetic_energy_j=initial_energy_j,
        final_kinetic_energy_j=final_energy,
        dissipated_energy_j=dissipated_energy_j + separation_loss_energy_j,
        separation_loss_energy_j=separation_loss_energy_j,
        energy_balance_residual_j=(
            initial_energy_j
            - final_energy
            - dissipated_energy_j
            - separation_loss_energy_j
        ),
    )


def _early_result(
    status: TurfContactStatus,
    velocity: np.ndarray,
    normal: np.ndarray,
    mass_kg: float,
) -> ReducedTurfContactResult:
    energy = 0.5 * mass_kg * float(velocity @ velocity)
    return _result(
        status,
        duration_s=0.0,
        step_count=0,
        peak_penetration_m=0.0,
        impulse=np.zeros(3),
        normal=normal,
        velocity=velocity,
        mass_kg=mass_kg,
        initial_energy_j=energy,
        dissipated_energy_j=0.0,
    )


def simulate_reduced_turf_contact(
    profile: TurfContactProfile,
    *,
    initial_contact_velocity_mps: object,
    surface_normal_unit: object,
    effective_mass_kg: float,
    time_step_s: float = 5e-6,
    max_time_s: float = _DEFAULT_MAX_TIME_S,
    cancel_check: Callable[[], bool] | None = None,
) -> ReducedTurfContactResult:
    """Integrate one effective contact mass until separation or a typed limit."""
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    velocity: np.ndarray = np.asarray(
        require_vector3(initial_contact_velocity_mps, "initial_contact_velocity_mps")
    )
    normal: np.ndarray = np.asarray(
        require_vector3(surface_normal_unit, "surface_normal_unit")
    )
    if not math.isclose(
        float(np.linalg.norm(normal)), 1.0, abs_tol=_UNIT_TOLERANCE, rel_tol=0.0
    ):
        raise ValueError("surface_normal_unit must be unit length")
    mass = require_finite_float(effective_mass_kg, "effective_mass_kg", positive=True)
    time_step = require_finite_float(time_step_s, "time_step_s", positive=True)
    max_time = require_finite_float(max_time_s, "max_time_s", positive=True)
    step_limit = min(math.ceil(max_time / time_step), _DEFAULT_MAX_STEPS)
    if profile.normal_stiffness_n_m == profile.normal_damping_n_s_m == 0.0:
        return _early_result(TurfContactStatus.NO_RESPONSE, velocity, normal, mass)
    if float(velocity @ normal) >= 0.0:
        return _early_result(TurfContactStatus.NO_CONTACT, velocity, normal, mass)
    initial_energy = 0.5 * mass * float(velocity @ velocity)
    impulse = np.zeros(3)
    penetration = 0.0
    peak_penetration = 0.0
    dissipated_energy = 0.0
    for step in range(1, step_limit + 1):
        if cancel_check is not None and cancel_check():
            return _result(
                TurfContactStatus.CANCELLED,
                duration_s=(step - 1) * time_step,
                step_count=step - 1,
                peak_penetration_m=peak_penetration,
                impulse=impulse,
                normal=normal,
                velocity=velocity,
                mass_kg=mass,
                initial_energy_j=initial_energy,
                dissipated_energy_j=dissipated_energy,
            )
        response = evaluate_turf_contact(
            profile,
            TurfContactKinematics(
                frame_id="reduced_contact_frame",
                reference_point_m=(0.0, 0.0, 0.0),
                application_point_m=(0.0, 0.0, 0.0),
                surface_normal_unit=_tuple3(normal),
                surface_velocity_mps=(0.0, 0.0, 0.0),
                contact_point_velocity_mps=_tuple3(velocity),
                penetration_m=penetration,
            ),
        )
        if response.status is TurfContactStatus.OUTSIDE_CALIBRATED_DOMAIN:
            return _result(
                response.status,
                duration_s=(step - 1) * time_step,
                step_count=step - 1,
                peak_penetration_m=peak_penetration,
                impulse=impulse,
                normal=normal,
                velocity=velocity,
                mass_kg=mass,
                initial_energy_j=initial_energy,
                dissipated_energy_j=dissipated_energy,
            )
        if (
            response.status is TurfContactStatus.NO_CONTACT
            and penetration > _CONTACT_TOLERANCE_M
            and float(velocity @ normal) >= 0.0
        ):
            # The tensile branch is clipped by the unilateral law. Any spring
            # energy remaining at that instant cannot return to the effective
            # contact mass and is therefore reported as an explicit snap-off
            # loss rather than disappearing from the energy balance.
            separation_loss = 0.5 * profile.normal_stiffness_n_m * penetration**2
            return _result(
                TurfContactStatus.SEPARATED,
                duration_s=(step - 1) * time_step,
                step_count=step - 1,
                peak_penetration_m=peak_penetration,
                impulse=impulse,
                normal=normal,
                velocity=velocity,
                mass_kg=mass,
                initial_energy_j=initial_energy,
                dissipated_energy_j=dissipated_energy,
                separation_loss_energy_j=separation_loss,
            )
        force = np.asarray(response.force_world_n)
        velocity = velocity + force * (time_step / mass)
        impulse += force * time_step
        dissipated_energy += response.dissipated_power_w * time_step
        penetration = max(0.0, penetration - float(velocity @ normal) * time_step)
        peak_penetration = max(peak_penetration, penetration)
        if penetration <= _CONTACT_TOLERANCE_M and float(velocity @ normal) >= 0.0:
            return _result(
                TurfContactStatus.SEPARATED,
                duration_s=step * time_step,
                step_count=step,
                peak_penetration_m=peak_penetration,
                impulse=impulse,
                normal=normal,
                velocity=velocity,
                mass_kg=mass,
                initial_energy_j=initial_energy,
                dissipated_energy_j=dissipated_energy,
            )
    return _result(
        TurfContactStatus.STEP_LIMIT,
        duration_s=step_limit * time_step,
        step_count=step_limit,
        peak_penetration_m=peak_penetration,
        impulse=impulse,
        normal=normal,
        velocity=velocity,
        mass_kg=mass,
        initial_energy_j=initial_energy,
        dissipated_energy_j=dissipated_energy,
    )


__all__ = ["ReducedTurfContactResult", "simulate_reduced_turf_contact"]
