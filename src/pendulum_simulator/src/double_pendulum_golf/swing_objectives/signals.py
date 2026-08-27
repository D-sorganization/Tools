"""Vectorized downswing signals shared by every swing objective.

:mod:`double_pendulum_golf.physics` is written one state at a time so it reads
like the equations of motion. The downswing optimizer evaluates objectives at
every collocation node inside a finite-difference Jacobian, which makes a
per-sample Python loop the dominant cost and violates the vectorization standard
in ``AGENTS.md`` 6a.

This module computes the same quantities for a whole ``(N, 4)`` block of states
with array operations. It follows the precedent already set by
:mod:`double_pendulum_golf.native_backend`, where a second implementation of the
same formulas is kept honest by tests rather than by inspection: every signal
here is pinned against its scalar counterpart in ``tests/test_swing_signals.py``.

Frames and units are those of ``physics``: SI throughout, world frame with the
hub at the origin, ``theta1`` measured from the downward vertical and ``phi`` the
wrist cock angle of the club relative to the arms.

Closes #4768.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.swing_objectives.velocity_terms import coupling_constant

__all__ = ["SwingSignals", "build_swing_signals", "generalized_accelerations"]

FloatArray = npt.NDArray[np.float64]

_STATE_WIDTH = 4
_TORQUE_WIDTH = 2
_PLANE_DIMENSIONS = 2


def _validated_trajectory(
    time_s: FloatArray, states: FloatArray, torques: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Coerce and validate a trajectory triple.

    Pre: ``time_s`` is strictly increasing with at least two finite samples;
    ``states`` is ``(N, 4)`` and ``torques`` is ``(N, 2)``, both finite.
    Post: returns float64 copies with those properties guaranteed.
    """
    time_array = np.asarray(time_s, dtype=np.float64)
    state_array = np.asarray(states, dtype=np.float64)
    torque_array = np.asarray(torques, dtype=np.float64)

    sample_count = time_array.shape[0] if time_array.ndim == 1 else -1
    expected_states = (sample_count, _STATE_WIDTH)
    expected_torques = (sample_count, _TORQUE_WIDTH)
    if sample_count < 2:
        raise ValueError("time_s must have shape (N,) with at least two samples")
    if state_array.shape != expected_states:
        raise ValueError(f"states must have shape {expected_states}")
    if torque_array.shape != expected_torques:
        raise ValueError(f"torques must have shape {expected_torques}")

    for name, array in (
        ("time_s", time_array),
        ("states", state_array),
        ("torques", torque_array),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")

    if not np.all(np.diff(time_array) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    return time_array, state_array, torque_array


def _mass_matrix_entries(
    phi: FloatArray, params: PendulumParams
) -> tuple[FloatArray, FloatArray, float]:
    """Return ``(M11, M12, M22)`` for a batch of wrist angles, matching physics."""
    effective_distal_mass = params.m2 + params.mClub
    coupling = coupling_constant(params)
    cos_phi = np.cos(phi)
    distal_inertia = effective_distal_mass * params.L2**2
    mass_11 = (
        (params.m1 + effective_distal_mass) * params.L1**2
        + distal_inertia
        + 2.0 * coupling * cos_phi
    )
    return mass_11, distal_inertia + coupling * cos_phi, distal_inertia


def _velocity_product(
    states: FloatArray, params: PendulumParams
) -> tuple[FloatArray, FloatArray]:
    """Return the combined ``C(q, q̇) q̇`` entries for a batch of states."""
    phi, dtheta1, dphi = states[:, 1], states[:, 2], states[:, 3]
    coupling_sine = -coupling_constant(params) * np.sin(phi)
    return (
        coupling_sine * (2.0 * dtheta1 * dphi + dphi**2),
        -coupling_sine * dtheta1**2,
    )


def _gravity_terms(
    states: FloatArray, params: PendulumParams
) -> tuple[FloatArray, FloatArray]:
    """Return the gravitational generalized forces for a batch of states."""
    theta1, phi = states[:, 0], states[:, 1]
    effective_distal_mass = params.m2 + params.mClub
    wrist = effective_distal_mass * params.g * params.L2 * np.sin(theta1 + phi)
    hub = (params.m1 + effective_distal_mass) * params.g * params.L1 * np.sin(theta1) + wrist
    return hub, wrist


def generalized_accelerations(
    states: FloatArray, torques: FloatArray, params: PendulumParams
) -> FloatArray:
    """Solve ``M(q) q̈ = tau − C(q, q̇) q̇ − G(q)`` for a block of samples.

    Args:
        states: ``(N, 4)`` array of ``[theta1, phi, dtheta1, dphi]``.
        torques: ``(N, 2)`` array of applied joint torques in N·m.
        params: Double pendulum physical parameters.

    Returns:
        ``(N, 2)`` array of ``[ddtheta1, ddphi]`` in rad/s².

    Pre: arrays are finite and correctly shaped.
    Post: result is finite.
    """
    mass_11, mass_12, distal_inertia = _mass_matrix_entries(states[:, 1], params)
    velocity_hub, velocity_wrist = _velocity_product(states, params)
    gravity_hub, gravity_wrist = _gravity_terms(states, params)

    rhs_hub = torques[:, 0] - velocity_hub - gravity_hub
    rhs_wrist = torques[:, 1] - velocity_wrist - gravity_wrist

    determinant = mass_11 * distal_inertia - mass_12**2
    hub_acceleration = (distal_inertia * rhs_hub - mass_12 * rhs_wrist) / determinant
    wrist_acceleration = (-mass_12 * rhs_hub + mass_11 * rhs_wrist) / determinant

    result: FloatArray = np.column_stack([hub_acceleration, wrist_acceleration])
    if not np.all(np.isfinite(result)):
        raise ValueError("Generalized accelerations are non-finite")
    return result


@dataclass(frozen=True, slots=True)
class SwingSignals:
    """Per-sample downswing quantities every objective is scored from.

    Attributes:
        time_s: ``(N,)`` strictly increasing sample times in s.
        states: ``(N, 4)`` array of ``[theta1, phi, dtheta1, dphi]``.
        torques: ``(N, 2)`` applied joint torques in N·m.
        generalized_acceleration: ``(N, 2)`` ``[ddtheta1, ddphi]`` in rad/s².
        clubhead_speed: ``(N,)`` tip speed in m/s — the headline golf metric.
        clubhead_acceleration: ``(N, 2)`` tip acceleration in m/s².
        grip_velocity: ``(N, 2)`` wrist-joint velocity in m/s.
        grip_force: ``(N, 2)`` proximal-on-distal force at the wrist in N.
        centrifugal_wrist_moment: ``(N,)`` left-hand-side centrifugal drive on the
            wrist row in N·m. Positive means the release is being driven.
        coriolis_hub_power: ``(N,)`` power the Coriolis coupling delivers to the
            hub in W. Negative means the kinetic chain is draining the arms.
    """

    time_s: FloatArray
    states: FloatArray
    torques: FloatArray
    generalized_acceleration: FloatArray
    clubhead_speed: FloatArray
    clubhead_acceleration: FloatArray
    grip_velocity: FloatArray
    grip_force: FloatArray
    centrifugal_wrist_moment: FloatArray
    coriolis_hub_power: FloatArray

    @property
    def sample_count(self) -> int:
        """Number of samples along the trajectory."""
        return int(self.time_s.shape[0])

    @property
    def grip_force_magnitude(self) -> FloatArray:
        """``(N,)`` magnitude of the grip force in N."""
        magnitude: FloatArray = np.linalg.norm(self.grip_force, axis=1)
        return magnitude

    @property
    def grip_force_power(self) -> FloatArray:
        """``(N,)`` power delivered into the club by the grip force in W."""
        power: FloatArray = np.sum(self.grip_force * self.grip_velocity, axis=1)
        return power

    def integrate(self, series: FloatArray) -> float:
        """Trapezoidally integrate a per-sample series over the trajectory time.

        Args:
            series: ``(N,)`` array aligned to :attr:`time_s`.

        Returns:
            The time integral.

        Pre: ``series`` has shape ``(N,)``.
        """
        if series.shape != (self.sample_count,):
            raise ValueError(f"series must have shape ({self.sample_count},)")
        return float(np.trapezoid(series, self.time_s))


def _planar_kinematics(
    states: FloatArray, qddot: FloatArray, params: PendulumParams
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return grip velocity, tip velocity, grip acceleration and tip acceleration."""
    theta1, phi = states[:, 0], states[:, 1]
    dtheta1, dphi = states[:, 2], states[:, 3]
    absolute_club_angle = theta1 + phi
    club_rate = dtheta1 + dphi
    club_acceleration = qddot[:, 0] + qddot[:, 1]

    cos_arm, sin_arm = np.cos(theta1), np.sin(theta1)
    cos_club, sin_club = np.cos(absolute_club_angle), np.sin(absolute_club_angle)

    grip_velocity = params.L1 * np.column_stack([cos_arm * dtheta1, sin_arm * dtheta1])
    tip_velocity = grip_velocity + params.L2 * np.column_stack(
        [cos_club * club_rate, sin_club * club_rate]
    )

    grip_acceleration = params.L1 * np.column_stack(
        [
            -sin_arm * dtheta1**2 + cos_arm * qddot[:, 0],
            cos_arm * dtheta1**2 + sin_arm * qddot[:, 0],
        ]
    )
    tip_acceleration = grip_acceleration + params.L2 * np.column_stack(
        [
            -sin_club * club_rate**2 + cos_club * club_acceleration,
            cos_club * club_rate**2 + sin_club * club_acceleration,
        ]
    )
    return grip_velocity, tip_velocity, grip_acceleration, tip_acceleration


def build_swing_signals(
    time_s: FloatArray,
    states: FloatArray,
    torques: FloatArray,
    params: PendulumParams,
) -> SwingSignals:
    """Compute every per-sample quantity the swing objectives are scored from.

    Args:
        time_s: ``(N,)`` strictly increasing sample times in s.
        states: ``(N, 4)`` array of ``[theta1, phi, dtheta1, dphi]``.
        torques: ``(N, 2)`` applied joint torques in N·m.
        params: Double pendulum physical parameters.

    Returns:
        The immutable signal bundle.

    Pre: shapes and finiteness as validated by ``_validated_trajectory``.
    Post: every array is finite and aligned to the sample axis.
    """
    time_array, state_array, torque_array = _validated_trajectory(time_s, states, torques)
    qddot = generalized_accelerations(state_array, torque_array, params)
    grip_velocity, tip_velocity, _, tip_acceleration = _planar_kinematics(
        state_array, qddot, params
    )

    effective_distal_mass = params.m2 + params.mClub
    gravity_vector = np.array([0.0, -params.g], dtype=np.float64)
    grip_force = effective_distal_mass * (tip_acceleration - gravity_vector)

    coupling = coupling_constant(params)
    sin_phi = np.sin(state_array[:, 1])
    arm_rate, uncock_rate = state_array[:, 2], state_array[:, 3]
    centrifugal_wrist_moment = coupling * sin_phi * arm_rate**2
    # Right-hand-side Coriolis force on the hub is +2*mu*sin(phi)*dtheta1*dphi,
    # so its power is that force times the arm rate.
    coriolis_hub_power = 2.0 * coupling * sin_phi * arm_rate**2 * uncock_rate

    return SwingSignals(
        time_s=time_array,
        states=state_array,
        torques=torque_array,
        generalized_acceleration=qddot,
        clubhead_speed=np.linalg.norm(tip_velocity, axis=1),
        clubhead_acceleration=tip_acceleration,
        grip_velocity=grip_velocity,
        grip_force=grip_force,
        centrifugal_wrist_moment=centrifugal_wrist_moment,
        coriolis_hub_power=coriolis_hub_power,
    )
