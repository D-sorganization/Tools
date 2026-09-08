"""Passive local grip-port mechanics and power-preserving frame changes."""

from __future__ import annotations

import numpy as np

from ._grip_contracts import (
    GripPortResponse,
    GripPortState,
    PassiveGripImpedance,
    factor6,
    vector6,
)
from ._validation import require_finite_float
from .types import RigidTransform


def _require_grip(grip: PassiveGripImpedance) -> None:
    if not isinstance(grip, PassiveGripImpedance):
        raise TypeError("grip must be PassiveGripImpedance")


def _require_state(state: GripPortState) -> None:
    if not isinstance(state, GripPortState):
        raise TypeError("state must be GripPortState")


def evaluate_grip_impedance(
    grip: PassiveGripImpedance, state: GripPortState
) -> GripPortResponse:
    """Evaluate reaction and the local power identity without a time integrator.

    Preconditions: validated contracts in the same local frame. Postconditions:
    finite outputs; stored energies and dissipation are squared factor norms,
    hence nonnegative even for rank-deficient impedance coefficients.
    """
    _require_grip(grip)
    _require_state(state)
    if grip.frame_id != state.frame_id:
        raise ValueError("grip and state must use the same frame")
    mass, damping, stiffness = map(
        np.asarray, (grip.inertance_factor, grip.damping_factor, grip.stiffness_factor)
    )
    q, velocity, acceleration = map(
        np.asarray, (state.displacement, state.velocity, state.acceleration)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        mass_velocity, mass_acceleration = mass @ velocity, mass @ acceleration
        damping_velocity = damping @ velocity
        elastic_q, elastic_velocity = stiffness @ q, stiffness @ velocity
        input_wrench = (
            mass.T @ mass_acceleration
            + damping.T @ damping_velocity
            + stiffness.T @ elastic_q
        )
        dissipated = float(damping_velocity @ damping_velocity)
        input_power = float(input_wrench @ velocity)
        energy_rate = float(
            mass_velocity @ mass_acceleration + elastic_q @ elastic_velocity
        )
        values = (
            float(mass_velocity @ mass_velocity / 2),
            float(elastic_q @ elastic_q / 2),
            dissipated,
            input_power,
            energy_rate,
            input_power - energy_rate - dissipated,
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("grip response must be finite")
    return GripPortResponse(
        grip.frame_id,
        grip.source_id,
        vector6(-input_wrench, "reaction_wrench"),
        *values[:5],
    )


def grip_frequency_impedance(
    grip: PassiveGripImpedance, angular_frequency_rad_s: float
) -> np.ndarray:
    """Return wrench/velocity impedance C + i(omega M - K/omega).

    Uses exp(i omega t); frequency must be finite and positive. This is an
    analytic local model, not an empirical FRF or a claim about its valid band.
    The Hermitian part is the nonnegative damping coefficient matrix.
    """
    _require_grip(grip)
    omega = require_finite_float(angular_frequency_rad_s, "frequency", positive=True)
    mass, damping, stiffness = map(
        np.asarray, (grip.inertance_factor, grip.damping_factor, grip.stiffness_factor)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = damping.T @ damping + 1j * (
            omega * (mass.T @ mass) - (stiffness.T @ stiffness) / omega
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("grip frequency impedance must be finite")
    return np.asarray(result)


def _motion_map(frame_id: str, transform: RigidTransform) -> np.ndarray:
    if not isinstance(transform, RigidTransform):
        raise TypeError("transform must be RigidTransform")
    if frame_id != transform.from_frame_id:
        raise ValueError("transform must start at the input frame")
    rotation = np.asarray(transform.rotation)
    # Columns are p cross R[:,j]; twist order is linear then angular.
    offset_cross_rotation = np.cross(transform.translation_m, rotation.T).T
    return np.block([[rotation, offset_cross_rotation], [np.zeros((3, 3)), rotation]])


def transform_grip_impedance(
    grip: PassiveGripImpedance, transform: RigidTransform
) -> PassiveGripImpedance:
    """Map constant port coefficients by power-preserving congruence.

    If q_to = A q_from, F_to = F_from A^-1 and H_to = A^-T H_from A^-1.
    The supplied transform is fixed; time-varying frames need transport terms
    and are outside this local constitutive operation.
    """
    _require_grip(grip)
    motion = _motion_map(grip.frame_id, transform)
    factors = [
        factor6(np.linalg.solve(motion.T, np.asarray(factor).T).T, "mapped factor")
        for factor in (
            grip.inertance_factor,
            grip.damping_factor,
            grip.stiffness_factor,
        )
    ]
    return PassiveGripImpedance(
        frame_id=transform.to_frame_id,
        inertance_factor=factors[0],
        damping_factor=factors[1],
        stiffness_factor=factors[2],
        source_id=grip.source_id,
    )


def transform_grip_state(
    state: GripPortState, transform: RigidTransform
) -> GripPortState:
    """Map infinitesimal local state through a fixed rigid-frame adjoint.

    This changes coordinates/reference point; it does not integrate finite
    orientation or transform acceleration through a moving frame.
    """
    _require_state(state)
    motion = _motion_map(state.frame_id, transform)
    mapped = [
        vector6(motion @ value, "mapped state")
        for value in (state.displacement, state.velocity, state.acceleration)
    ]
    return GripPortState(transform.to_frame_id, *mapped)


__all__ = [
    "PassiveGripImpedance",
    "GripPortState",
    "GripPortResponse",
    "evaluate_grip_impedance",
    "grip_frequency_impedance",
    "transform_grip_impedance",
    "transform_grip_state",
]
