"""Planar triple-pendulum dynamics and swing-frame sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import require
from shared.python.swing_sim import reference
from shared.python.swing_sim.types import PlaneOrientation, SwingSample

__all__ = [
    "TRIPLE_PENDULUM_JOINT_IDS",
    "TriplePendulumParameters",
    "TriplePendulumSwing",
    "triple_derivatives",
    "triple_total_energy",
]

TRIPLE_PENDULUM_JOINT_IDS = (
    "joint.shoulder",
    "joint.elbow",
    "joint.wrist",
)


@dataclass(frozen=True)
class TriplePendulumParameters:
    """Planar three-link pendulum parameters, absolute-angle formulation.

    Per link i: mass ``m`` [kg], length ``l`` [m], COM distance from the
    proximal joint ``lc`` [m], and inertia about the COM ``i_com``
    [kg·m²]. ``damping`` are viscous joint coefficients on the RELATIVE
    joint rates (shoulder, elbow, wrist).
    """

    m: tuple[float, float, float]
    l: tuple[float, float, float]  # noqa: E741 — matches the double-pendulum naming
    lc: tuple[float, float, float]
    i_com: tuple[float, float, float]
    damping: tuple[float, float, float] = (0.4, 0.3, 0.25)

    def __post_init__(self) -> None:
        for name in ("m", "l", "lc", "i_com"):
            for value in getattr(self, name):
                require(
                    math.isfinite(value) and value > 0.0,
                    f"{name} entries must be finite and > 0",
                    value,
                )
        for value in self.damping:
            require(
                math.isfinite(value) and value >= 0.0,
                "damping entries must be finite and >= 0",
                value,
            )
        for lc_i, l_i in zip(self.lc, self.l, strict=True):
            require(lc_i <= l_i, "lc must not exceed l", lc_i)

    @classmethod
    def golf_default(cls) -> TriplePendulumParameters:
        """Upper arm + forearm + club, splitting the shared golf defaults.

        The 0.75 m / 7.5 kg arm of the double-pendulum default becomes an
        upper arm and a forearm; the club link reuses the shared shaft +
        clubhead composition so total reach and club inertia match the
        double model.
        """
        upper = (4.5, 0.40, 0.40 * 0.45, (1.0 / 12.0) * 4.5 * 0.40**2)
        fore = (3.0, 0.35, 0.35 * 0.45, (1.0 / 12.0) * 3.0 * 0.35**2)
        # Club: 1.0 m shaft (0.15 kg, COM at 43%) + 0.20 kg head at the tip.
        length, ms, mh = 1.0, 0.15, 0.20
        m_club = ms + mh
        shaft_com = length * 0.43
        lc_club = (shaft_com * ms + length * mh) / m_club
        i_shaft_com = (1.0 / 12.0) * ms * length * length
        parallel = ms * (shaft_com - lc_club) ** 2 + mh * (length - lc_club) ** 2
        club = (m_club, length, lc_club, i_shaft_com + parallel)
        return cls(
            m=(upper[0], fore[0], club[0]),
            l=(upper[1], fore[1], club[1]),
            lc=(upper[2], fore[2], club[2]),
            i_com=(upper[3], fore[3], club[3]),
        )


def _triple_a_matrix(p: TriplePendulumParameters) -> np.ndarray:
    """Coefficient matrix ``a[i, k]``: dr_ci/dphi_k magnitude factors."""
    a = np.zeros((3, 3))
    for i in range(3):
        for k in range(3):
            if k < i:
                a[i, k] = p.l[k]
            elif k == i:
                a[i, k] = p.lc[i]
    return a


def triple_derivatives(
    p: TriplePendulumParameters,
    state: np.ndarray,
    g_inplane: tuple[float, float],
) -> np.ndarray:
    """State derivatives for the unforced triple pendulum.

    ``state`` is ``[phi1, phi2, phi3, dphi1, dphi2, dphi3]`` with
    absolute angles from the in-plane downward vertical. Standard n-link
    absolute-angle equations: ``M(phi) ddphi + C(phi, dphi) + G(phi) +
    D(dphi) = 0`` with ``beta[k, j] = sum_i m_i a_ik a_ij``.
    """
    phi = state[:3]
    dphi = state[3:]
    a = _triple_a_matrix(p)
    m_arr = np.array(p.m)
    beta = np.einsum("i,ik,ij->kj", m_arr, a, a)

    diff = phi[:, None] - phi[None, :]
    mass = beta * np.cos(diff) + np.diag(p.i_com)
    coriolis = (beta * np.sin(diff)) @ (dphi**2)

    gx, gy = g_inplane
    weights = m_arr @ a  # sum_i m_i a_ik per coordinate k
    gravity = -weights * (gx * np.cos(phi) + gy * np.sin(phi))

    # Damping on relative joint rates: psi_dot = J dphi.
    j = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    damping = j.T @ (np.array(p.damping) * (j @ dphi))

    ddphi = np.linalg.solve(mass, -(coriolis + gravity + damping))
    derivative: np.ndarray = np.concatenate([dphi, ddphi])
    return derivative


def triple_total_energy(
    p: TriplePendulumParameters,
    state: np.ndarray,
    g_inplane: tuple[float, float],
) -> float:
    """Total mechanical energy [J] (kinetic + in-plane gravitational)."""
    phi = state[:3]
    dphi = state[3:]
    a = _triple_a_matrix(p)
    m_arr = np.array(p.m)
    beta = np.einsum("i,ik,ij->kj", m_arr, a, a)
    mass = beta * np.cos(phi[:, None] - phi[None, :]) + np.diag(p.i_com)
    kinetic = 0.5 * float(dphi @ mass @ dphi)

    gx, gy = g_inplane
    e_x, e_y = np.sin(phi), -np.cos(phi)
    potential = 0.0
    for i in range(3):
        cx = sum(p.l[j] * e_x[j] for j in range(i)) + p.lc[i] * e_x[i]
        cy = sum(p.l[j] * e_y[j] for j in range(i)) + p.lc[i] * e_y[i]
        potential -= p.m[i] * (gx * cx + gy * cy)
    return kinetic + potential


class TriplePendulumSwing:
    """Planar triple pendulum on an oriented plane (swing frame).

    Same posture as :class:`DoublePendulumSwing`: integrate once at
    construction with classical RK4 on a uniform grid, then sample by
    linear interpolation. Wrap in ``AppFrameSwing`` for the app.
    """

    def __init__(
        self,
        parameters: TriplePendulumParameters | None = None,
        plane: PlaneOrientation | None = None,
        initial_state: np.ndarray | None = None,
        duration: float = 1.5,
        dt: float = 1e-3,
        gravity_m_s2: float = 9.80665,
    ) -> None:
        require(
            math.isfinite(duration) and duration > 0.0,
            "duration must be finite and > 0",
            duration,
        )
        require(math.isfinite(dt) and 0.0 < dt <= duration, "invalid dt", dt)
        self._p = parameters or TriplePendulumParameters.golf_default()
        self._plane = plane or PlaneOrientation()
        self._dt = float(dt)
        self._n_steps = int(round(duration / dt))
        self._duration = self._n_steps * self._dt

        self._plane_r = reference.plane_rotation(
            self._plane.yaw_rad, self._plane.side_tilt_rad, self._plane.forward_tilt_rad
        )
        self._g_inplane = reference.in_plane_gravity(self._plane_r, gravity_m_s2)

        state = (
            np.asarray(initial_state, dtype=float)
            if initial_state is not None
            else np.array(
                [-math.pi / 2.0, -math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0, 0.0]
            )
        )
        require(state.shape == (6,), "initial_state must be a 6-vector", state.shape)
        self._states = np.empty((self._n_steps + 1, 6))
        self._states[0] = state
        for i in range(self._n_steps):
            self._states[i + 1] = self._rk4_step(self._states[i])

    def _rk4_step(self, y: np.ndarray) -> np.ndarray:
        dt = self._dt
        f = lambda s: triple_derivatives(self._p, s, self._g_inplane)  # noqa: E731
        k1 = f(y)
        k2 = f(y + dt / 2.0 * k1)
        k3 = f(y + dt / 2.0 * k2)
        k4 = f(y + dt * k3)
        next_state: np.ndarray = np.asarray(
            y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )
        return next_state

    def state_at(self, t: float) -> np.ndarray:
        """Interpolated absolute-angle joint state at ``t``."""
        require(math.isfinite(t), "t must be finite", t)
        require(-1e-9 <= t <= self._duration + 1e-9, "t within duration", t)
        t = min(max(t, 0.0), self._duration)
        idx = t / self._dt
        i0 = min(int(idx), self._n_steps)
        i1 = min(i0 + 1, self._n_steps)
        frac = idx - i0
        state: np.ndarray = np.asarray(
            (1.0 - frac) * self._states[i0] + frac * self._states[i1]
        )
        return state

    def joint_positions(self, t: float) -> np.ndarray:
        """Pivot and each link endpoint in the oriented swing frame."""
        phi = self.state_at(t)[:3]
        x_axis, up_axis = self._plane_r[:, 0], self._plane_r[:, 2]
        points = [np.zeros(3)]
        current = np.zeros(3)
        for length, angle in zip(self._p.l, phi, strict=True):
            current = current + length * (
                math.sin(float(angle)) * x_axis - math.cos(float(angle)) * up_axis
            )
            points.append(current.copy())
        positions: np.ndarray = np.vstack(points)
        return positions

    @property
    def parameters(self) -> TriplePendulumParameters:
        """Pendulum parameters used for the integration."""
        return self._p

    @property
    def joint_ids(self) -> tuple[str, ...]:
        """Stable ordering for the three generalized torque coordinates."""
        return TRIPLE_PENDULUM_JOINT_IDS

    @property
    def generalized_state_ids(self) -> tuple[str, ...]:
        """Stable component IDs for absolute angles and angular rates."""
        return (
            "joint.shoulder.absolute_angle_rad",
            "joint.elbow.absolute_angle_rad",
            "joint.wrist.absolute_angle_rad",
            "joint.shoulder.absolute_rate_rad_s",
            "joint.elbow.absolute_rate_rad_s",
            "joint.wrist.absolute_rate_rad_s",
        )

    @property
    def generalized_state_units(self) -> tuple[str, ...]:
        """SI units aligned with :attr:`generalized_state_ids`."""
        return ("rad", "rad", "rad", "rad/s", "rad/s", "rad/s")

    def generalized_state_at(self, t: float) -> np.ndarray:
        """Return the integrated absolute-angle state at ``t``."""
        return self.state_at(t)

    def joint_torques_at(self, t: float) -> dict[str, float]:
        """Return the current passive model's zero commanded torque."""
        self.state_at(t)
        return {joint_id: 0.0 for joint_id in TRIPLE_PENDULUM_JOINT_IDS}

    @property
    def plane(self) -> PlaneOrientation:
        """Swing-plane orientation."""
        return self._plane

    @property
    def duration(self) -> float:
        """Total duration [s] of the integrated swing."""
        return self._duration

    def sample(self, t: float) -> SwingSample:
        """Clubhead (tip of link 3) sample at ``t``, swing frame."""
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        t = min(max(t, 0.0), self._duration)
        row = self.state_at(t)
        phi, dphi = row[:3], row[3:]

        lengths: np.ndarray = np.array(self._p.l)
        x = float(np.sum(lengths * np.sin(phi)))
        y = -float(np.sum(lengths * np.cos(phi)))
        vx = float(np.sum(lengths * np.cos(phi) * dphi))
        vy = float(np.sum(lengths * np.sin(phi) * dphi))

        r_plane = self._plane_r
        x_axis, normal, up_axis = r_plane[:, 0], r_plane[:, 1], r_plane[:, 2]
        position = x * x_axis + y * up_axis
        c3, s3 = math.cos(phi[2]), math.sin(phi[2])
        local = np.array([[c3, 0.0, s3], [0.0, 1.0, 0.0], [-s3, 0.0, c3]])
        pose = np.eye(4)
        pose[:3, :3] = r_plane @ local
        pose[:3, 3] = position
        twist = np.concatenate([float(dphi[2]) * normal, vx * x_axis + vy * up_axis])
        return SwingSample(t=t, pose=pose, twist=twist)
