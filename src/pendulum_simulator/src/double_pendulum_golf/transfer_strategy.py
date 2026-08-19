"""Phase-resolved metrics for falsifiable proximal-to-distal transfer studies.

The module is deliberately model-neutral. Physics adapters supply achieved-state
signals and an exact pointwise drift/control force split; this layer only validates,
integrates, and ranks them. A proximal angular velocity is therefore never relabeled
as anatomical shoulder or torso velocity by this API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_CLOSURE_ATOL = 1e-9


def _finite_vector(name: str, value: object, sample_count: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (sample_count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({sample_count},)")
    return array


def _finite_planar(name: str, value: object, sample_count: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    expected = (sample_count, 2)
    if array.shape != expected or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {expected}")
    return array


@dataclass(frozen=True, slots=True)
class TransferSignals:
    """Achieved-state signals required for transfer-strategy evaluation.

    Preconditions
    -------------
    Time is strictly increasing with at least two samples. All arrays are finite,
    use SI units, and total grip force equals drift plus control pointwise.
    """

    time_s: FloatArray
    proximal_angular_velocity_rad_s: FloatArray
    distal_speed_m_s: FloatArray
    distal_kinetic_energy_j: FloatArray
    grip_velocity_m_s: FloatArray
    grip_force_total_n: FloatArray
    grip_force_drift_n: FloatArray
    grip_force_control_n: FloatArray
    wrist_control_couple_nm: FloatArray
    club_angular_velocity_rad_s: FloatArray
    model_tier: str

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float).reshape(-1)
        if time.size < 2 or not np.all(np.isfinite(time)):
            raise ValueError("time_s must contain at least two finite samples")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        if not isinstance(self.model_tier, str) or not self.model_tier.strip():
            raise ValueError("model_tier must be a non-empty string")
        count = time.size
        vector_names = (
            "proximal_angular_velocity_rad_s",
            "distal_speed_m_s",
            "distal_kinetic_energy_j",
            "wrist_control_couple_nm",
            "club_angular_velocity_rad_s",
        )
        planar_names = (
            "grip_velocity_m_s",
            "grip_force_total_n",
            "grip_force_drift_n",
            "grip_force_control_n",
        )
        object.__setattr__(self, "time_s", time)
        for name in vector_names:
            object.__setattr__(self, name, _finite_vector(name, getattr(self, name), count))
        for name in planar_names:
            object.__setattr__(self, name, _finite_planar(name, getattr(self, name), count))
        residual = self.grip_force_total_n - self.grip_force_drift_n
        residual -= self.grip_force_control_n
        if not np.allclose(residual, 0.0, atol=_CLOSURE_ATOL, rtol=0.0):
            raise ValueError("grip-force drift/control closure failed")
        if np.any(self.distal_speed_m_s < 0.0) or np.any(self.distal_kinetic_energy_j < 0.0):
            raise ValueError("distal speed and kinetic energy must be non-negative")


@dataclass(frozen=True, slots=True)
class TransferSummary:
    """Integrated transfer outcomes for one declared time window."""

    start_s: float
    end_s: float
    proximal_velocity_at_start_rad_s: float
    proximal_velocity_at_end_rad_s: float
    peak_distal_speed_m_s: float
    distal_energy_gain_j: float
    total_grip_work_j: float
    drift_grip_work_j: float
    control_grip_work_j: float
    work_closure_residual_j: float
    negative_grip_work_j: float
    negative_along_path_impulse_n_s: float
    wrist_control_work_j: float
    peak_grip_force_n: float
    model_tier: str


def _negative_linear_integral(time: FloatArray, values: FloatArray) -> float:
    """Integrate only the negative portion of a piecewise-linear signal."""
    total = 0.0
    for left, right, t_left, t_right in zip(
        values[:-1], values[1:], time[:-1], time[1:], strict=True
    ):
        width = float(t_right - t_left)
        if left <= 0.0 and right <= 0.0:
            total += 0.5 * float(left + right) * width
        elif left < 0.0 < right:
            fraction = float(-left / (right - left))
            total += 0.5 * float(left) * width * fraction
        elif right < 0.0 < left:
            fraction = float(left / (left - right))
            total += 0.5 * float(right) * width * (1.0 - fraction)
    return total


def _window_indices(signals: TransferSignals, start_s: float, end_s: float) -> FloatArray:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or start_s >= end_s:
        raise ValueError("start_s must be less than end_s and both must be finite")
    time = signals.time_s
    if start_s < time[0] or end_s > time[-1]:
        raise ValueError("analysis window must lie inside the trajectory")
    mask = (time >= start_s) & (time <= end_s)
    if np.count_nonzero(mask) < 2:
        raise ValueError("analysis window must contain at least two samples")
    return np.flatnonzero(mask)


def _force_power_terms(
    velocity: FloatArray,
    total_force: FloatArray,
    drift_force: FloatArray,
    control_force: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return total, drift, control, and along-path force signals."""
    total_power = np.einsum("ij,ij->i", total_force, velocity)
    drift_power = np.einsum("ij,ij->i", drift_force, velocity)
    control_power = np.einsum("ij,ij->i", control_force, velocity)
    speed = np.linalg.norm(velocity, axis=1)
    tangent = np.divide(
        velocity,
        speed[:, None],
        out=np.zeros_like(velocity),
        where=speed[:, None] > 1e-12,
    )
    along_path_force = np.einsum("ij,ij->i", total_force, tangent)
    return total_power, drift_power, control_power, along_path_force


def summarize_transfer(
    signals: TransferSignals, *, start_s: float, end_s: float
) -> TransferSummary:
    """Integrate drift/control work and braking metrics over one phase window."""
    if not isinstance(signals, TransferSignals):
        raise TypeError("signals must be a TransferSignals instance")
    indices = _window_indices(signals, start_s, end_s).astype(np.int64)
    time = signals.time_s[indices]
    velocity = signals.grip_velocity_m_s[indices]
    total_force = signals.grip_force_total_n[indices]
    drift_force = signals.grip_force_drift_n[indices]
    control_force = signals.grip_force_control_n[indices]
    total_power, drift_power, control_power, along_path_force = _force_power_terms(
        velocity, total_force, drift_force, control_force
    )
    total_work = float(np.trapezoid(total_power, time))
    drift_work = float(np.trapezoid(drift_power, time))
    control_work = float(np.trapezoid(control_power, time))
    wrist_power = (
        signals.wrist_control_couple_nm[indices] * signals.club_angular_velocity_rad_s[indices]
    )
    energy = signals.distal_kinetic_energy_j[indices]
    return TransferSummary(
        start_s=float(start_s),
        end_s=float(end_s),
        proximal_velocity_at_start_rad_s=float(
            signals.proximal_angular_velocity_rad_s[indices[0]]
        ),
        proximal_velocity_at_end_rad_s=float(
            signals.proximal_angular_velocity_rad_s[indices[-1]]
        ),
        peak_distal_speed_m_s=float(np.max(signals.distal_speed_m_s[indices])),
        distal_energy_gain_j=float(energy[-1] - energy[0]),
        total_grip_work_j=total_work,
        drift_grip_work_j=drift_work,
        control_grip_work_j=control_work,
        work_closure_residual_j=total_work - drift_work - control_work,
        negative_grip_work_j=_negative_linear_integral(time, total_power),
        negative_along_path_impulse_n_s=_negative_linear_integral(time, along_path_force),
        wrist_control_work_j=float(np.trapezoid(wrist_power, time)),
        peak_grip_force_n=float(np.max(np.linalg.norm(total_force, axis=1))),
        model_tier=signals.model_tier,
    )


def _club_kinetic_energy(states: FloatArray, params: object) -> FloatArray:
    """Return shaft-plus-clubhead kinetic energy for the planar model."""
    theta = states[:, 0]
    phi = states[:, 1]
    proximal_rate = states[:, 2]
    club_rate = proximal_rate + states[:, 3]
    wrist_velocity = np.column_stack(
        (
            params.L1 * np.cos(theta) * proximal_rate,
            params.L1 * np.sin(theta) * proximal_rate,
        )
    )
    club_angle = theta + phi
    club_tangent = np.column_stack((np.cos(club_angle), np.sin(club_angle)))
    shaft_com_velocity = wrist_velocity + 0.5 * params.L2 * club_rate[:, None] * club_tangent
    tip_velocity = wrist_velocity + params.L2 * club_rate[:, None] * club_tangent
    shaft_inertia = params.m2 * params.L2**2 / 12.0
    shaft_energy = 0.5 * params.m2 * np.sum(shaft_com_velocity**2, axis=1)
    shaft_energy += 0.5 * shaft_inertia * club_rate**2
    head_energy = 0.5 * params.mClub * np.sum(tip_velocity**2, axis=1)
    return np.asarray(shaft_energy + head_energy, dtype=float)


def double_pendulum_transfer_signals(result: object) -> TransferSignals:
    """Adapt a double-pendulum result to the model-neutral transfer contract.

    The proximal signal is the model's first-link angular velocity. It is not an
    anatomical shoulder or torso measurement. The grip is the wrist joint, and
    force direction is proximal-on-distal as declared by ``net_joint_forces``.
    """
    from .counterfactual import zero_torque_joint_forces_double

    required = ("t", "states", "params", "n_steps")
    if any(not hasattr(result, name) for name in required):
        raise TypeError("result must provide the double-pendulum result contract")
    states = np.asarray(result.states, dtype=float)
    if states.shape != (result.n_steps, 4) or not np.all(np.isfinite(states)):
        raise ValueError("double-pendulum result states must be finite with width four")
    total_force = np.empty((result.n_steps, 2))
    drift_force = np.empty_like(total_force)
    grip_velocity = np.empty_like(total_force)
    wrist_couple = np.empty(result.n_steps)
    distal_speed = np.empty(result.n_steps)
    for index in range(result.n_steps):
        total_force[index] = result.joint_forces_at(index)["wrist"]
        drift_force[index] = zero_torque_joint_forces_double(states[index], result.params)[
            "wrist"
        ]
        velocities = result.joint_velocities_at(index)
        grip_velocity[index] = velocities["wrist_vel"]
        distal_speed[index] = velocities["tip_speed"]
        wrist_couple[index] = result.torques_at(index)[1]
    return TransferSignals(
        time_s=np.asarray(result.t, dtype=float),
        proximal_angular_velocity_rad_s=states[:, 2],
        distal_speed_m_s=distal_speed,
        distal_kinetic_energy_j=_club_kinetic_energy(states, result.params),
        grip_velocity_m_s=grip_velocity,
        grip_force_total_n=total_force,
        grip_force_drift_n=drift_force,
        grip_force_control_n=total_force - drift_force,
        wrist_control_couple_nm=wrist_couple,
        club_angular_velocity_rad_s=states[:, 2] + states[:, 3],
        model_tier="exact_planar_double_pendulum",
    )


def pareto_front(values: object, *, maximize: tuple[bool, ...]) -> npt.NDArray[np.int64]:
    """Return nondominated row indices for mixed max/min objectives."""
    objectives = np.asarray(values, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if objectives.shape[1] != len(maximize):
        raise ValueError("maximize length must match the objective count")
    if objectives.shape[0] == 0 or not np.all(np.isfinite(objectives)):
        raise ValueError("values must contain at least one finite row")
    oriented = objectives * np.where(np.asarray(maximize), -1.0, 1.0)
    keep = np.ones(oriented.shape[0], dtype=bool)
    for candidate in range(oriented.shape[0]):
        no_worse = np.all(oriented <= oriented[candidate], axis=1)
        strictly_better = np.any(oriented < oriented[candidate], axis=1)
        if np.any(no_worse & strictly_better):
            keep[candidate] = False
    return np.flatnonzero(keep)


__all__ = [
    "TransferSignals",
    "TransferSummary",
    "double_pendulum_transfer_signals",
    "pareto_front",
    "summarize_transfer",
]
