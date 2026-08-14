"""Pure-Python reference implementation of the swing dynamics.

Mirrors ``rust_core/swing-core/src/swing`` operation-for-operation (same
formula grouping and evaluation order) so it can serve as:

1. the parity oracle for the Rust kernel (``parity``-marked tests), and
2. the fallback backend for one-shot calls on machines without the wheel.

Hot integration loops must use the Rust backend via
:mod:`shared.python.swing_sim._rust_facade` — this module is deliberately
unoptimised Python.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from shared.python.contracts import require

from .types import PendulumParameters, PendulumState

MASS_MATRIX_SINGULAR_TOLERANCE = 1e-12
"""Numerical tolerance for detecting singular mass matrices."""


def plane_rotation(yaw: float, side_tilt: float, fwd_tilt: float) -> np.ndarray:
    """World-from-plane rotation matrix ``Rz(yaw) @ Rx(side) @ Ry(fwd)``.

    Columns are the plane's local axes in world coordinates: column 0 =
    in-plane horizontal, column 1 = plane normal, column 2 = in-plane up.
    All angles in radians.
    """
    for name, value in (("yaw", yaw), ("side_tilt", side_tilt), ("fwd_tilt", fwd_tilt)):
        require(math.isfinite(value), f"{name} must be finite", value)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cs, ss = math.cos(side_tilt), math.sin(side_tilt)
    cf, sf = math.cos(fwd_tilt), math.sin(fwd_tilt)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cs, -ss], [0.0, ss, cs]])
    ry = np.array([[cf, 0.0, sf], [0.0, 1.0, 0.0], [-sf, 0.0, cf]])
    rotation: np.ndarray = np.asarray(rz @ rx @ ry)
    return rotation


def in_plane_gravity(plane_r: np.ndarray, g: float) -> tuple[float, float]:
    """Project world gravity ``(0, 0, -g)`` into the swing plane.

    Returns ``(g_x_inplane, g_y_inplane)`` along the plane's local
    horizontal and local up axes. The EOM consumes this 2-vector directly.
    """
    require(math.isfinite(g) and g >= 0.0, "g must be finite and >= 0", g)
    r = np.asarray(plane_r, dtype=np.float64)
    require(r.shape == (3, 3), "plane_r must be a 3x3 matrix", r.shape)
    g_world = np.array([0.0, 0.0, -g])
    gx = float(g_world @ r[:, 0])
    gy = float(g_world @ r[:, 2])
    return gx, gy


def in_plane_gravity_from_tilts(
    yaw: float, side_tilt: float, fwd_tilt: float, g: float
) -> tuple[float, float]:
    """In-plane gravity straight from the three tilt angles (radians)."""
    return in_plane_gravity(plane_rotation(yaw, side_tilt, fwd_tilt), g)


def mass_matrix(p: PendulumParameters, theta2: float) -> np.ndarray:
    """Symmetric 2x2 mass matrix for the given relative angle."""
    cos_theta2 = math.cos(theta2)
    m11 = p.i1 + p.i2 + p.m2 * p.l1 * p.l1 + 2.0 * p.m2 * p.l1 * p.lc2 * cos_theta2
    m12 = p.i2 + p.m2 * p.l1 * p.lc2 * cos_theta2
    m22 = p.i2
    mass: np.ndarray = np.array([[m11, m12], [m12, m22]])
    return mass


def coriolis_vector(
    p: PendulumParameters, theta2: float, omega1: float, omega2: float
) -> tuple[float, float]:
    """Coriolis and centripetal generalized-force vector."""
    h = -p.m2 * p.l1 * p.lc2 * math.sin(theta2)
    c1 = h * (2.0 * omega1 * omega2 + omega2 * omega2)
    c2 = -h * omega1 * omega1
    return c1, c2


def gravity_vector(
    p: PendulumParameters,
    theta1: float,
    theta2: float,
    g_inplane: tuple[float, float],
) -> tuple[float, float]:
    """Gravity generalized-force vector for an in-plane gravity 2-vector.

    For the flat plane ``(0, -g)`` this reduces exactly to the classic
    scalar form used by the UpstreamDrift reference.
    """
    gx, gy = g_inplane
    t12 = theta1 + theta2
    a1 = p.m1 * p.lc1 + p.m2 * p.l1
    a2 = p.m2 * p.lc2
    g1 = -a1 * (gx * math.cos(theta1) + gy * math.sin(theta1)) - a2 * (
        gx * math.cos(t12) + gy * math.sin(t12)
    )
    g2 = -a2 * (gx * math.cos(t12) + gy * math.sin(t12))
    return g1, g2


def damping_vector(
    p: PendulumParameters, omega1: float, omega2: float
) -> tuple[float, float]:
    """Viscous damping torques."""
    return p.d1 * omega1, p.d2 * omega2


def _invert_mass_matrix(p: PendulumParameters, theta2: float) -> np.ndarray:
    m = mass_matrix(p, theta2)
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    require(
        abs(det) > MASS_MATRIX_SINGULAR_TOLERANCE,
        "mass matrix determinant too close to zero; check pendulum parameters",
        det,
    )
    inverse: np.ndarray = np.array(
        [[m[1, 1] / det, -m[0, 1] / det], [-m[1, 0] / det, m[0, 0] / det]]
    )
    return inverse


def derivatives(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
) -> tuple[float, float, float, float]:
    """State derivatives ``(dθ1, dθ2, dω1, dω2)`` for unforced dynamics."""
    c1, c2 = coriolis_vector(p, state.theta2, state.omega1, state.omega2)
    g1, g2 = gravity_vector(p, state.theta1, state.theta2, g_inplane)
    d1, d2 = damping_vector(p, state.omega1, state.omega2)
    inv_m = _invert_mass_matrix(p, state.theta2)
    rhs1 = -(c1 + g1 + d1)
    rhs2 = -(c2 + g2 + d2)
    acc1 = inv_m[0, 0] * rhs1 + inv_m[0, 1] * rhs2
    acc2 = inv_m[1, 0] * rhs1 + inv_m[1, 1] * rhs2
    return state.omega1, state.omega2, acc1, acc2


def derivatives_forced(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    torques_nm: tuple[float, float],
) -> tuple[float, float, float, float]:
    """State derivatives with shoulder/wrist generalized torques in N*m."""
    require(len(torques_nm) == 2, "torques_nm must contain two joint torques")
    tau1, tau2 = float(torques_nm[0]), float(torques_nm[1])
    require(
        math.isfinite(tau1) and math.isfinite(tau2),
        "joint torques must be finite",
        torques_nm,
    )
    passive = derivatives(p, state, g_inplane)
    inv_m = _invert_mass_matrix(p, state.theta2)
    delta1 = inv_m[0, 0] * tau1 + inv_m[0, 1] * tau2
    delta2 = inv_m[1, 0] * tau1 + inv_m[1, 1] * tau2
    return passive[0], passive[1], passive[2] + delta1, passive[3] + delta2


def derivatives_locked(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    torques_nm: tuple[float, float],
    locked: tuple[bool, bool],
) -> tuple[float, float, float, float]:
    """Derivatives under ideal shoulder/wrist coordinate constraints.

    ``theta1`` is the shoulder coordinate relative to ground and ``theta2``
    is the wrist coordinate relative to the upper segment. A locked coordinate
    has zero velocity/acceleration; the free coordinate is solved from the
    reduced mass-matrix equation so constraint reactions remain truthful.
    """
    require(
        len(locked) == 2 and all(type(value) is bool for value in locked),
        "locked must contain two boolean flags",
        locked,
    )
    require(any(locked), "derivatives_locked requires at least one lock")
    tau1, tau2 = _validated_torques(torques_nm)
    require(
        not locked[0] or state.omega1 == 0.0,
        "locked shoulder requires zero relative velocity",
        state.omega1,
    )
    require(
        not locked[1] or state.omega2 == 0.0,
        "locked wrist requires zero relative velocity",
        state.omega2,
    )
    if locked == (True, True):
        return 0.0, 0.0, 0.0, 0.0
    rhs1, rhs2 = _forced_rhs(p, state, g_inplane, (tau1, tau2))
    mass = mass_matrix(p, state.theta2)
    if locked[0]:
        return 0.0, state.omega2, 0.0, rhs2 / mass[1, 1]
    return state.omega1, 0.0, rhs1 / mass[0, 0], 0.0


def _validated_torques(torques_nm: tuple[float, float]) -> tuple[float, float]:
    """Validate one shoulder/wrist applied-torque pair."""
    require(len(torques_nm) == 2, "torques_nm must contain two joint torques")
    torques = float(torques_nm[0]), float(torques_nm[1])
    require(
        all(math.isfinite(value) for value in torques),
        "joint torques must be finite",
        torques_nm,
    )
    return torques


def _forced_rhs(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    torques_nm: tuple[float, float],
) -> tuple[float, float]:
    """Generalized-force right-hand side before the mass-matrix solve."""
    c1, c2 = coriolis_vector(p, state.theta2, state.omega1, state.omega2)
    g1, g2 = gravity_vector(p, state.theta1, state.theta2, g_inplane)
    d1, d2 = damping_vector(p, state.omega1, state.omega2)
    return torques_nm[0] - c1 - g1 - d1, torques_nm[1] - c2 - g2 - d2


def rk4_step(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
) -> PendulumState:
    """Advance the state by one classical RK4 step of size ``dt``."""
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    y = (state.theta1, state.theta2, state.omega1, state.omega2)

    def f(v: tuple[float, ...]) -> tuple[float, float, float, float]:
        return derivatives(
            p,
            PendulumState(theta1=v[0], theta2=v[1], omega1=v[2], omega2=v[3]),
            g_inplane,
        )

    def add(a: tuple[float, ...], s: float, b: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(ai + s * bi for ai, bi in zip(a, b, strict=True))

    k1 = f(y)
    k2 = f(add(y, dt / 2.0, k1))
    k3 = f(add(y, dt / 2.0, k2))
    k4 = f(add(y, dt, k3))
    out = [
        y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)
    ]
    return PendulumState(theta1=out[0], theta2=out[1], omega1=out[2], omega2=out[3])


def rk4_step_forced(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    time_s: float,
    dt: float,
    torque_at: Callable[[float], tuple[float, float]],
) -> PendulumState:
    """Advance a non-autonomous prescribed-torque state by one RK4 step."""
    require(math.isfinite(time_s), "time_s must be finite", time_s)
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    y = (state.theta1, state.theta2, state.omega1, state.omega2)

    def f(
        sample_time: float, values: tuple[float, ...]
    ) -> tuple[float, float, float, float]:
        sample_state = PendulumState(
            theta1=values[0],
            theta2=values[1],
            omega1=values[2],
            omega2=values[3],
        )
        return derivatives_forced(p, sample_state, g_inplane, torque_at(sample_time))

    def add(
        a: tuple[float, ...], scale: float, b: tuple[float, ...]
    ) -> tuple[float, ...]:
        return tuple(ai + scale * bi for ai, bi in zip(a, b, strict=True))

    k1 = f(time_s, y)
    k2 = f(time_s + dt / 2.0, add(y, dt / 2.0, k1))
    k3 = f(time_s + dt / 2.0, add(y, dt / 2.0, k2))
    k4 = f(time_s + dt, add(y, dt, k3))
    out = [
        y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)
    ]
    return PendulumState(theta1=out[0], theta2=out[1], omega1=out[2], omega2=out[3])


def rk4_step_locked(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
    time_s: float,
    dt: float,
    torque_at: Callable[[float], tuple[float, float]],
    locked: tuple[bool, bool],
) -> PendulumState:
    """Advance an ideal coordinate-constrained state by one RK4 step."""
    require(math.isfinite(time_s), "time_s must be finite", time_s)
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    y = (state.theta1, state.theta2, state.omega1, state.omega2)

    def f(
        sample_time: float, values: tuple[float, ...]
    ) -> tuple[float, float, float, float]:
        sample_state = PendulumState(*values)
        return derivatives_locked(
            p, sample_state, g_inplane, torque_at(sample_time), locked
        )

    def add(
        values: tuple[float, ...], scale: float, rates: tuple[float, ...]
    ) -> tuple[float, ...]:
        return tuple(
            value + scale * rate for value, rate in zip(values, rates, strict=True)
        )

    k1 = f(time_s, y)
    k2 = f(time_s + dt / 2.0, add(y, dt / 2.0, k1))
    k3 = f(time_s + dt / 2.0, add(y, dt / 2.0, k2))
    k4 = f(time_s + dt, add(y, dt, k3))
    values = tuple(
        y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(4)
    )
    projected = (
        state.theta1 if locked[0] else values[0],
        state.theta2 if locked[1] else values[1],
        0.0 if locked[0] else values[2],
        0.0 if locked[1] else values[3],
    )
    return PendulumState(*projected)


def total_energy(
    p: PendulumParameters,
    state: PendulumState,
    g_inplane: tuple[float, float],
) -> float:
    """Total mechanical energy (kinetic + gravitational potential) [J]."""
    m = mass_matrix(p, state.theta2)
    q0, q1 = state.omega1, state.omega2
    kinetic = 0.5 * (m[0, 0] * q0 * q0 + 2.0 * m[0, 1] * q0 * q1 + m[1, 1] * q1 * q1)

    gx, gy = g_inplane
    e1x, e1y = math.sin(state.theta1), -math.cos(state.theta1)
    t12 = state.theta1 + state.theta2
    e2x, e2y = math.sin(t12), -math.cos(t12)
    p1 = (p.lc1 * e1x, p.lc1 * e1y)
    p2 = (p.l1 * e1x + p.lc2 * e2x, p.l1 * e1y + p.lc2 * e2y)
    potential = -(p.m1 * (gx * p1[0] + gy * p1[1]) + p.m2 * (gx * p2[0] + gy * p2[1]))
    return float(kinetic + potential)


def simulate(
    p: PendulumParameters,
    initial: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Simulate ``n_steps`` RK4 steps.

    Returns an ``(n_steps + 1, 4)`` array of rows
    ``[theta1, theta2, omega1, omega2]`` including the initial state.
    """
    require(n_steps >= 0, "n_steps must be >= 0", n_steps)
    out: np.ndarray = np.empty((n_steps + 1, 4), dtype=np.float64)
    out[0] = (initial.theta1, initial.theta2, initial.omega1, initial.omega2)
    current = initial
    for i in range(n_steps):
        current = rk4_step(p, current, g_inplane, dt)
        out[i + 1] = (current.theta1, current.theta2, current.omega1, current.omega2)
    return out


def simulate_forced(
    p: PendulumParameters,
    initial: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
    n_steps: int,
    torque_at: Callable[[float], tuple[float, float]],
) -> np.ndarray:
    """Simulate prescribed generalized torques on the existing RK4 grid."""
    require(n_steps >= 0, "n_steps must be >= 0", n_steps)
    require(callable(torque_at), "torque_at must be callable", torque_at)
    out: np.ndarray = np.empty((n_steps + 1, 4), dtype=np.float64)
    out[0] = (initial.theta1, initial.theta2, initial.omega1, initial.omega2)
    current = initial
    for i in range(n_steps):
        current = rk4_step_forced(p, current, g_inplane, i * dt, dt, torque_at)
        out[i + 1] = (current.theta1, current.theta2, current.omega1, current.omega2)
    return out


def simulate_locked(
    p: PendulumParameters,
    initial: PendulumState,
    g_inplane: tuple[float, float],
    dt: float,
    n_steps: int,
    torque_at: Callable[[float], tuple[float, float]],
    locked: tuple[bool, bool],
) -> np.ndarray:
    """Simulate ideal locked coordinates with optional prescribed torques."""
    require(n_steps >= 0, "n_steps must be >= 0", n_steps)
    require(callable(torque_at), "torque_at must be callable", torque_at)
    derivatives_locked(p, initial, g_inplane, torque_at(0.0), locked)
    out: np.ndarray = np.empty((n_steps + 1, 4), dtype=np.float64)
    out[0] = (initial.theta1, initial.theta2, initial.omega1, initial.omega2)
    current = initial
    for i in range(n_steps):
        current = rk4_step_locked(p, current, g_inplane, i * dt, dt, torque_at, locked)
        out[i + 1] = (current.theta1, current.theta2, current.omega1, current.omega2)
    return out
