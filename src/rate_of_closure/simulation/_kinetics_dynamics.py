"""Pure double-pendulum dynamics used by the swing-kinetics façade."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np

from rate_of_closure._contracts import ensure, require
from shared.python.swing_sim import reference
from shared.python.swing_sim.types import PendulumParameters, PendulumState

__all__ = [
    "inverse_dynamics",
    "reaction_forces",
    "simulate_forced",
    "zero_torque_counterfactual",
]


def _eom_terms(
    p: PendulumParameters,
    theta: np.ndarray,
    omega: np.ndarray,
    g_inplane: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample Coriolis, gravity, and damping forces."""
    n = theta.shape[0]
    coriolis = np.empty((n, 2))
    gravity = np.empty((n, 2))
    damping = np.empty((n, 2))
    for i in range(n):
        theta1_i = float(theta[i, 0])
        theta2_i = float(theta[i, 1])
        omega1_i = float(omega[i, 0])
        omega2_i = float(omega[i, 1])
        coriolis[i] = reference.coriolis_vector(p, theta2_i, omega1_i, omega2_i)
        gravity[i] = reference.gravity_vector(p, theta1_i, theta2_i, g_inplane)
        damping[i] = reference.damping_vector(p, omega1_i, omega2_i)
    return coriolis, gravity, damping


def inverse_dynamics(
    p: PendulumParameters,
    states: np.ndarray,
    g_inplane: tuple[float, float],
    dt: float,
) -> dict[str, np.ndarray]:
    """Return inverse dynamics over a uniformly sampled joint trajectory.

    Generalized forces map one-to-one to physical shoulder and wrist actuator
    torques by virtual work. Accelerations use central differences, with
    one-sided differences at the ends.
    """
    states = np.asarray(states, dtype=float)
    require(
        states.ndim == 2 and states.shape[1] == 4 and states.shape[0] >= 3,
        "states must be an (N>=3, 4) array",
        states.shape,
    )
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    require(bool(np.all(np.isfinite(states))), "states must be finite", None)
    theta = states[:, :2]
    omega = states[:, 2:]

    alpha = np.gradient(omega, dt, axis=0)
    coriolis, gravity, damping = _eom_terms(p, theta, omega, g_inplane)

    inertial = np.empty((states.shape[0], 2))
    for i in range(states.shape[0]):
        inertial[i] = (
            reference.mass_matrix(p, float(theta[i, 1])) @ alpha[i] + coriolis[i]
        )
    applied = inertial + gravity + damping
    ensure(bool(np.all(np.isfinite(applied))), "inverse dynamics must be finite")
    return {
        "applied": applied,
        "gravity": -gravity,
        "damping": -damping,
        "inertial": inertial,
        "alpha": alpha,
    }


def zero_torque_counterfactual(
    p: PendulumParameters,
    states: np.ndarray,
    g_inplane: tuple[float, float],
    *,
    locked: tuple[bool, bool] = (False, False),
) -> dict[str, np.ndarray]:
    """Evaluate the pointwise state-matched zero-torque counterfactual."""
    states = np.asarray(states, dtype=float)
    require(
        states.ndim == 2 and states.shape[1] == 4 and states.shape[0] >= 1,
        "states must be an (N>=1, 4) array",
        states.shape,
    )
    require(bool(np.all(np.isfinite(states))), "states must be finite", None)
    require(
        len(locked) == 2 and all(type(value) is bool for value in locked),
        "locked must contain two boolean flags",
        locked,
    )

    acceleration = np.empty((states.shape[0], 2))
    inertial_torque = np.empty_like(acceleration)
    for index, row in enumerate(states):
        state = PendulumState(
            theta1=float(row[0]),
            theta2=float(row[1]),
            omega1=float(row[2]),
            omega2=float(row[3]),
        )
        derivative = (
            reference.derivatives_locked(p, state, g_inplane, (0.0, 0.0), locked)
            if any(locked)
            else reference.derivatives(p, state, g_inplane)
        )
        acceleration[index] = derivative[2:]
        coriolis = np.asarray(
            reference.coriolis_vector(p, state.theta2, state.omega1, state.omega2)
        )
        inertial_torque[index] = (
            reference.mass_matrix(p, state.theta2) @ acceleration[index] + coriolis
        )
    ensure(
        bool(np.all(np.isfinite(acceleration)))
        and bool(np.all(np.isfinite(inertial_torque))),
        "ZTCF outputs must be finite",
    )
    return {
        "acceleration": acceleration,
        "inertial_torque": inertial_torque,
    }


def _point_kinematics(
    radius: float, phi: np.ndarray, phid: np.ndarray, phidd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return in-plane position and acceleration along one link."""
    e = np.stack([np.sin(phi), -np.cos(phi)], axis=1)
    e_t = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    pos = radius * e
    acc = radius * (phidd[:, None] * e_t - (phid**2)[:, None] * e)
    return pos, acc


def reaction_forces(
    p: PendulumParameters,
    theta: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    g_inplane: tuple[float, float],
    clubhead_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shoulder, wrist, and estimated clubhead forces in-plane."""
    g_vec = np.asarray(g_inplane, dtype=float)
    phi1, phi12 = theta[:, 0], theta[:, 0] + theta[:, 1]
    phid1, phid12 = omega[:, 0], omega[:, 0] + omega[:, 1]
    phidd1, phidd12 = alpha[:, 0], alpha[:, 0] + alpha[:, 1]

    _, a_arm_com = _point_kinematics(p.lc1, phi1, phid1, phidd1)
    _, a_elbow = _point_kinematics(p.l1, phi1, phid1, phidd1)
    _, a_club_rel = _point_kinematics(p.lc2, phi12, phid12, phidd12)
    _, a_tip_rel = _point_kinematics(p.l2, phi12, phid12, phidd12)

    a_club_com = a_elbow + a_club_rel
    a_tip = a_elbow + a_tip_rel
    f_wrist = p.m2 * (a_club_com - g_vec)
    f_shoulder = p.m1 * (a_arm_com - g_vec) + f_wrist
    f_head = clubhead_mass_kg * (a_tip - g_vec)
    return f_shoulder, f_wrist, f_head


def simulate_forced(
    p: PendulumParameters,
    initial: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
    n_steps: int,
    torque_fn: Callable[[float], tuple[float, float]],
) -> np.ndarray:
    """RK4-simulate the pendulum with an applied joint-torque profile."""
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    require(n_steps >= 0, "n_steps must be >= 0", n_steps)
    require(callable(torque_fn), "torque_fn must be callable")

    def f(t: float, y: np.ndarray) -> np.ndarray:
        tau = np.asarray(torque_fn(t), dtype=float)
        theta1, theta2 = float(y[0]), float(y[1])
        omega1, omega2 = float(y[2]), float(y[3])
        c = reference.coriolis_vector(p, theta2, omega1, omega2)
        g = reference.gravity_vector(p, theta1, theta2, g_inplane)
        d = reference.damping_vector(p, omega1, omega2)
        m = reference.mass_matrix(p, theta2)
        rhs = tau - np.asarray(c) - np.asarray(g) - np.asarray(d)
        acc = np.linalg.solve(m, rhs)
        return cast(np.ndarray, np.concatenate([y[2:], acc]))

    out = np.empty((n_steps + 1, 4))
    out[0] = (initial.theta1, initial.theta2, initial.omega1, initial.omega2)
    for i in range(n_steps):
        t, y = i * dt, out[i]
        k1 = f(t, y)
        k2 = f(t + dt / 2.0, y + dt / 2.0 * k1)
        k3 = f(t + dt / 2.0, y + dt / 2.0 * k2)
        k4 = f(t + dt, y + dt * k3)
        out[i + 1] = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out
