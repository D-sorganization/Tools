"""App-frame swing sources for the simulation session (epic #4103).

Every source satisfies the :class:`shared.python.swing_sim.swing_source.
SwingSource` protocol (``duration`` + ``sample(t) -> SwingSample``) and
returns samples in the APP frame — the AffineDrift launch-monitor
convention (x target, y up, z right) used by the rate-of-closure model
and the 3D scene. The shared ``swing_sim`` dynamics run in the swing
frame (x forward, y left, z up — identical to the flight frame), so
pendulum sources are wrapped in :class:`AppFrameSwing`.

Sources:

* :class:`ManualSwingSource` — wraps an existing
  :class:`~rate_of_closure.model.ImpactScenario` as a trivial
  constant-twist source: straight-line reference-point travel plus the
  scenario's constant angular velocity, square at mid-window.
* ``DoublePendulumSwing`` (from ``swing_sim``) via :func:`make_source`.
* :class:`TriplePendulumSwing` — planar three-link pendulum on the same
  oriented plane, integrated with the same classical RK4. New model
  (the shared package only ships the double pendulum, recon #4105).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.model import ImpactScenario, solve
from shared.python.swing_sim import reference
from shared.python.swing_sim.run_config import DoublePendulumRunConfig
from shared.python.swing_sim.swing_source import DoublePendulumSwing, SwingSource
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.types import (
    PendulumState,
    PlaneOrientation,
    SwingSample,
)

__all__ = [
    "SOURCE_KINDS",
    "APP_FROM_SWING",
    "AppFrameSwing",
    "ManualSwingSource",
    "TriplePendulumParameters",
    "TriplePendulumSwing",
    "make_source",
]

#: Swing-source kinds accepted by :func:`make_source`, in UI order.
SOURCE_KINDS: tuple[str, ...] = ("manual", "double_pendulum", "triple_pendulum")

#: Rotation taking swing/flight-frame vectors (x fwd, y left, z up) into
#: app-frame vectors (x target, y up, z right): app y = swing z,
#: app z = -swing y.
APP_FROM_SWING = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rodrigues(axis_omega: np.ndarray, dt: float) -> np.ndarray:
    """Rotation matrix for spinning at ``axis_omega`` [rad/s] for ``dt`` s."""
    theta = float(np.linalg.norm(axis_omega)) * dt
    if abs(theta) < 1e-15:
        return np.eye(3)
    axis = axis_omega / np.linalg.norm(axis_omega)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.asarray(
        np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    )


class AppFrameSwing:
    """Adapter re-expressing a swing-frame :class:`SwingSource` in the app frame.

    Positions, rotations, and twists are rotated by
    :data:`APP_FROM_SWING`; timing is passed through unchanged.
    """

    def __init__(self, inner: SwingSource) -> None:
        require(
            isinstance(inner, SwingSource),
            "inner must satisfy the SwingSource protocol",
            inner,
        )
        self._inner = inner

    @property
    def inner(self) -> SwingSource:
        """The wrapped swing-frame source."""
        return self._inner

    @property
    def duration(self) -> float:
        """Total duration [s] of the wrapped swing."""
        return float(self._inner.duration)

    def sample(self, t: float) -> SwingSample:
        """Sample the wrapped source and rotate the result into the app frame."""
        s = self._inner.sample(t)
        c = APP_FROM_SWING
        pose = np.eye(4)
        pose[:3, :3] = c @ s.pose[:3, :3]
        pose[:3, 3] = c @ s.pose[:3, 3]
        twist = np.concatenate([c @ s.twist[:3], c @ s.twist[3:]])
        return SwingSample(t=s.t, pose=pose, twist=twist)

    def joint_positions(self, t: float) -> np.ndarray:
        """Articulated joints in the app frame, when the source exposes them."""
        joint_positions = getattr(self._inner, "joint_positions", None)
        require(callable(joint_positions), "inner source has no joint geometry")
        sampler = cast(Callable[[float], np.ndarray], joint_positions)
        positions: np.ndarray = np.asarray(sampler(t), dtype=float) @ APP_FROM_SWING.T
        return positions

    @property
    def joint_ids(self) -> tuple[str, ...]:
        """Stable joint IDs exposed by the wrapped source, if supported."""
        identifiers = getattr(self._inner, "joint_ids", ())
        return tuple(cast(tuple[str, ...], identifiers))

    def joint_torques_at(self, t: float) -> dict[str, float]:
        """Forward generalized torques; scalar joint values are frame-invariant."""
        torque_sampler = getattr(self._inner, "joint_torques_at", None)
        require(callable(torque_sampler), "inner source has no joint torque history")
        sampler = cast(Callable[[float], dict[str, float]], torque_sampler)
        return sampler(t)


class ManualSwingSource:
    """Constant-twist source built from an :class:`ImpactScenario` (app frame).

    The clubhead reference point travels dead down the target line at the
    scenario speed while the head spins at the scenario's constant angular
    velocity — exactly the twist model of the explorer, extended over a
    short time window. The head is square (identity rotation, reference at
    the origin) at the window midpoint, matching the explorer's "instant
    of maximum compression" convention.
    """

    def __init__(self, scenario: ImpactScenario, duration: float = 0.06) -> None:
        require(
            isinstance(scenario, ImpactScenario),
            "scenario must be an ImpactScenario",
            scenario,
        )
        require(
            math.isfinite(duration) and duration > 0.0,
            "duration must be finite and > 0",
            duration,
        )
        self._scenario = scenario
        self._duration = float(duration)
        result = solve(scenario)
        self._omega = np.radians(np.array(result.omega_dps))
        speed_mps = result.reference_speed_mph * 0.44704
        self._velocity = np.array([speed_mps, 0.0, 0.0])

    @property
    def scenario(self) -> ImpactScenario:
        """The wrapped scenario."""
        return self._scenario

    @property
    def duration(self) -> float:
        """Window length [s]."""
        return self._duration

    def sample(self, t: float) -> SwingSample:
        """Clubhead sample at ``t``; square at the window midpoint."""
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        t = min(max(t, 0.0), self._duration)
        dt = t - self._duration / 2.0
        pose = np.eye(4)
        pose[:3, :3] = _rodrigues(self._omega, dt)
        pose[:3, 3] = self._velocity * dt
        twist = np.concatenate([self._omega, self._velocity])
        return SwingSample(t=t, pose=pose, twist=twist)


# ── Triple pendulum ─────────────────────────────────────────────────


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
    return np.concatenate([dphi, ddphi])


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
    linear interpolation. Wrap in :class:`AppFrameSwing` for the app.
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
        return np.asarray(y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def state_at(self, t: float) -> np.ndarray:
        """Interpolated absolute-angle joint state at ``t``."""
        require(math.isfinite(t), "t must be finite", t)
        require(-1e-9 <= t <= self._duration + 1e-9, "t within duration", t)
        t = min(max(t, 0.0), self._duration)
        idx = t / self._dt
        i0 = min(int(idx), self._n_steps)
        i1 = min(i0 + 1, self._n_steps)
        frac = idx - i0
        return np.asarray((1.0 - frac) * self._states[i0] + frac * self._states[i1])

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
        return np.vstack(points)

    @property
    def parameters(self) -> TriplePendulumParameters:
        """Pendulum parameters used for the integration."""
        return self._p

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

        lengths = np.array(self._p.l)
        x = float(np.sum(lengths * np.sin(phi)))
        y = float(-np.sum(lengths * np.cos(phi)))
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


def make_source(
    kind: str,
    scenario: ImpactScenario,
    plane: PlaneOrientation | None = None,
    duration: float = 1.5,
    run_config: DoublePendulumRunConfig | None = None,
    torque_library: TorqueProfileLibrary | None = None,
) -> SwingSource:
    """Build an app-frame swing source by kind.

    Args:
        kind: One of :data:`SOURCE_KINDS`.
        scenario: The manual scenario (used directly by ``"manual"``;
            pendulum kinds use it only for impact offsets downstream).
        plane: Swing-plane orientation for the pendulum kinds.
        duration: Pendulum integration length [s].
        run_config: Passive or prescribed double-pendulum execution policy.
        torque_library: Canonical profiles used by prescribed execution.

    Returns:
        A source whose samples are in the app frame.
    """
    require(kind in SOURCE_KINDS, f"unknown swing source kind {kind!r}", kind)
    execution = run_config or DoublePendulumRunConfig()
    require(
        kind == "double_pendulum" or not execution.joint_locks.has_locks,
        "joint locks are unsupported outside the double-pendulum source",
        kind,
    )
    if kind == "manual":
        return ManualSwingSource(scenario)
    if kind == "double_pendulum":
        # Start on the target side (theta1 = -pi/2, arm horizontal) so
        # the gravity-driven downswing carries the clubhead TOWARD the
        # target (+x) at the bottom of the arc.
        start = PendulumState(theta1=-math.pi / 2.0, theta2=0.0, omega1=0.0, omega2=0.0)
        return AppFrameSwing(
            DoublePendulumSwing(
                plane=plane,
                initial_state=start,
                duration=duration,
                backend="auto",
                run_config=execution,
                torque_library=torque_library,
            )
        )
    return AppFrameSwing(TriplePendulumSwing(plane=plane, duration=duration))
