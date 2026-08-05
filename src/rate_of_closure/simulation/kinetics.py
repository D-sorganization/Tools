"""Swing kinetics: inverse dynamics over the pendulum swing (#4125 H2).

Per-sample joint torques, joint reaction forces, and joint powers for
the double-pendulum swing source, computed from the ``swing_sim``
pendulum EOM surfaces (:func:`shared.python.swing_sim.reference.
mass_matrix` / ``coriolis_vector`` / ``gravity_vector`` /
``damping_vector``).

Presentation conventions mirror the movement optimizer
(``src/movement_optimizer/gui/plot_renderer.py`` and
``models/swingset_forces.py``): SI arrays with unit-suffixed field
names (``*_nm``, ``*_w``, ``*_n``), joint order proximal-to-distal,
and "Time (s)" / "Torque (N·m)" / "Power (W)" / "Force (N)" axis
labels downstream.

Sign conventions (documented here because the movement optimizer does
not state one for torque):

* Generalized coordinates: ``theta1`` is the upper-segment (arm)
  absolute angle from the in-plane downward vertical; ``theta2`` is
  the lower segment (club) angle relative to the upper segment.
* Positive torque acts counter-clockwise about the swing-plane normal
  (the plane's local +y axis, right-hand rule) — the direction of
  increasing ``theta``. The generalized torques map one-to-one onto
  the physical joints: coordinate 1 is the shoulder (pivot) torque,
  coordinate 2 the wrist torque (virtual-work argument in the
  inverse-dynamics docstring).
* Reaction forces are the force the PROXIMAL side exerts on the
  DISTAL side at the joint, expressed in the app frame (x target,
  y up, z right). Gravity is included (a static hang shows the
  supporting force, not zero).

The swing sources are passive (gravity-driven), so the net applied
torque recovered by inverse dynamics is ~0 up to integration error;
the physically interesting series are the gravity / damping / inertial
breakdown, the reaction forces, and the joint powers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.simulation.session import SimulationRun
from rate_of_closure.simulation.sources import (
    APP_FROM_SWING,
    AppFrameSwing,
    make_source,
)
from shared.python.swing_sim import reference
from shared.python.swing_sim.swing_source import DoublePendulumSwing
from shared.python.swing_sim.types import PendulumParameters, PendulumState

__all__ = [
    "CLUBHEAD_MASS_KG",
    "KINETIC_JOINT_NAMES",
    "KineticsSeries",
    "compute_kinetics",
    "inverse_dynamics",
    "kinetics_for_run",
    "simulate_forced",
]

#: Joint names in coordinate order (proximal to distal), mirroring the
#: movement optimizer's lowercase joint-name legend convention.
KINETIC_JOINT_NAMES: tuple[str, ...] = ("shoulder", "wrist")

#: Clubhead point mass [kg] used for the clubhead-force estimate — the
#: shared golf-default head mass (swing_sim.types _CLUBHEAD_MASS_KG).
#: The double-pendulum lumps shaft + head into segment 2, so the head
#: force is a point-mass estimate at the segment tip (documented
#: approximation).
CLUBHEAD_MASS_KG = 0.20


@dataclass(frozen=True)
class KineticsSeries:
    """Per-sample swing kinetics of one double-pendulum run.

    All arrays share the sample count N of the run's swing grid.
    Torque columns follow :data:`KINETIC_JOINT_NAMES` order and the
    sign convention documented in the module docstring. The torque
    breakdown satisfies (per sample, per joint)::

        torque_applied = torque_inertial - torque_gravity - torque_damping

    with ``torque_inertial = M(q)·q̈ + C(q, q̇)`` (what actually
    accelerates the segments), ``torque_gravity = -G(q)`` (gravity's
    generalized torque — the driver of the passive swing) and
    ``torque_damping = -D(q̇)`` (always resistive).

    Attributes:
        t: (N,) sample times [s].
        joint_names: Joint labels in column order.
        torque_applied_nm: (N, 2) net applied joint torque from inverse
            dynamics (~0 for the passive swing sources).
        torque_gravity_nm: (N, 2) gravity torque per joint.
        torque_damping_nm: (N, 2) viscous damping torque per joint.
        torque_inertial_nm: (N, 2) net inertial (intersegmental)
            torque per joint — the plotted "net torque".
        power_w: (N, 2) joint power ``τ_net · ω`` per coordinate; the
            row sum equals d(kinetic energy)/dt.
        shoulder_force_n: (N, 3) app-frame reaction force at the pivot.
        wrist_force_n: (N, 3) app-frame reaction force at the wrist.
        clubhead_force_n: (N, 3) app-frame point-mass force estimate on
            the clubhead (mass :data:`CLUBHEAD_MASS_KG` at the tip).
        pivot_position_m: (3,) app-frame, ball-aligned pivot position.
        wrist_positions_m: (N, 3) app-frame, ball-aligned wrist path.
        clubhead_positions_m: (N, 3) app-frame, ball-aligned head path.
        plane_x_app: (3,) app-frame in-plane horizontal unit axis.
        plane_up_app: (3,) app-frame in-plane up unit axis.
        impact_time_s: Impact instant tau of the source run [s].
    """

    t: np.ndarray = field(repr=False)
    joint_names: tuple[str, ...]
    torque_applied_nm: np.ndarray = field(repr=False)
    torque_gravity_nm: np.ndarray = field(repr=False)
    torque_damping_nm: np.ndarray = field(repr=False)
    torque_inertial_nm: np.ndarray = field(repr=False)
    power_w: np.ndarray = field(repr=False)
    shoulder_force_n: np.ndarray = field(repr=False)
    wrist_force_n: np.ndarray = field(repr=False)
    clubhead_force_n: np.ndarray = field(repr=False)
    pivot_position_m: np.ndarray = field(repr=False)
    wrist_positions_m: np.ndarray = field(repr=False)
    clubhead_positions_m: np.ndarray = field(repr=False)
    plane_x_app: np.ndarray = field(repr=False)
    plane_up_app: np.ndarray = field(repr=False)
    impact_time_s: float

    def __post_init__(self) -> None:
        n = self.t.shape[0]
        j = len(self.joint_names)
        require(n >= 3, "kinetics needs at least 3 samples", n)
        require(j >= 2, "kinetics needs at least 2 joints", j)
        for name in (
            "torque_applied_nm",
            "torque_gravity_nm",
            "torque_damping_nm",
            "torque_inertial_nm",
            "power_w",
        ):
            require(
                getattr(self, name).shape == (n, j),
                f"{name} must be (N, {j})",
                getattr(self, name).shape,
            )
        for name in (
            "shoulder_force_n",
            "wrist_force_n",
            "clubhead_force_n",
            "wrist_positions_m",
            "clubhead_positions_m",
        ):
            require(
                getattr(self, name).shape == (n, 3),
                f"{name} must be (N, 3)",
                getattr(self, name).shape,
            )
        require(
            self.pivot_position_m.shape == (3,),
            "pivot_position_m must be a 3-vector",
            self.pivot_position_m.shape,
        )
        require(
            math.isfinite(self.impact_time_s) and self.impact_time_s >= 0.0,
            "impact_time_s must be finite and >= 0",
            self.impact_time_s,
        )

    def force_magnitude_n(self, which: str) -> np.ndarray:
        """(N,) magnitude of one force series.

        Args:
            which: ``"shoulder"``, ``"wrist"``, or ``"clubhead"``.
        """
        require(
            which in ("shoulder", "wrist", "clubhead"),
            "unknown force series",
            which,
        )
        vectors = getattr(self, f"{which}_force_n")
        return np.asarray(np.linalg.norm(vectors, axis=1), dtype=float)


def _eom_terms(
    p: PendulumParameters,
    theta: np.ndarray,
    omega: np.ndarray,
    g_inplane: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample Coriolis, gravity, and damping generalized forces."""
    n = theta.shape[0]
    coriolis = np.empty((n, 2))
    gravity = np.empty((n, 2))
    damping = np.empty((n, 2))
    for i in range(n):
        coriolis[i] = reference.coriolis_vector(
            p, theta[i, 1], omega[i, 0], omega[i, 1]
        )
        gravity[i] = reference.gravity_vector(p, theta[i, 0], theta[i, 1], g_inplane)
        damping[i] = reference.damping_vector(p, omega[i, 0], omega[i, 1])
    return coriolis, gravity, damping


def inverse_dynamics(
    p: PendulumParameters,
    states: np.ndarray,
    g_inplane: tuple[float, float],
    dt: float,
) -> dict[str, np.ndarray]:
    """Per-sample inverse dynamics over a sampled joint trajectory.

    Joint accelerations come from central differences of the sampled
    rates (one-sided at the ends), so a trajectory produced by a FORCED
    forward simulation recovers its driving torque profile up to
    O(dt²) differencing error (test-pinned round trip).

    Virtual-work mapping (why generalized = physical joint torques):
    a shoulder actuator torque τ_s on the arm does work ``τ_s·δθ1``; a
    wrist actuator applies +τ_w to the club and -τ_w to the arm, doing
    work ``τ_w·δ(θ1+θ2) - τ_w·δθ1 = τ_w·δθ2``. Hence the generalized
    force vector is exactly ``(τ_shoulder, τ_wrist)``.

    Args:
        p: Pendulum parameters.
        states: (N, 4) rows ``[theta1, theta2, omega1, omega2]`` on a
            uniform grid.
        g_inplane: In-plane gravity 2-vector (see ``reference``).
        dt: Uniform sample step [s].

    Returns:
        Dict with (N, 2) arrays ``applied``, ``gravity``, ``damping``,
        ``inertial`` (see :class:`KineticsSeries` for the identity
        relating them) and the (N, 2) accelerations ``alpha``.
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

    alpha = np.gradient(omega, dt, axis=0)  # central; one-sided ends
    coriolis, gravity, damping = _eom_terms(p, theta, omega, g_inplane)

    n = states.shape[0]
    inertial = np.empty((n, 2))
    for i in range(n):
        m = reference.mass_matrix(p, theta[i, 1])
        inertial[i] = m @ alpha[i] + coriolis[i]
    applied = inertial + gravity + damping
    ensure(bool(np.all(np.isfinite(applied))), "inverse dynamics must be finite")
    return {
        "applied": applied,
        "gravity": -gravity,
        "damping": -damping,
        "inertial": inertial,
        "alpha": alpha,
    }


def _point_kinematics(
    radius: float, phi: np.ndarray, phid: np.ndarray, phidd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """In-plane position and acceleration of a point at ``radius`` along
    a link with absolute angle ``phi`` from the downward vertical."""
    e = np.stack([np.sin(phi), -np.cos(phi)], axis=1)
    e_t = np.stack([np.cos(phi), np.sin(phi)], axis=1)  # tangential
    pos = radius * e
    acc = radius * (phidd[:, None] * e_t - (phid**2)[:, None] * e)
    return pos, acc


def _reaction_forces(
    p: PendulumParameters,
    theta: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    g_inplane: tuple[float, float],
    clubhead_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton–Euler reaction forces per segment (in-plane 2-vectors).

    Lower segment (club): ``F_wrist = m2·(a_c2 - g)`` — the force the
    arm exerts on the club at the wrist. Upper segment (arm):
    ``F_shoulder = m1·(a_c1 - g) + F_wrist`` (the wrist reaction from
    the club enters with opposite sign on the arm). Clubhead:
    point-mass estimate ``m_head·(a_tip - g)`` from the lower-segment
    kinematics at the tip.
    """
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
    """RK4-simulate the pendulum with an applied joint-torque profile.

    Same classical RK4 and evaluation order as ``reference.simulate``;
    the applied torque enters the RHS as ``M·q̈ = τ - C - G - D``.
    Used by the inverse-dynamics round-trip and energy tests, and by
    any future driven-swing source.

    Args:
        p: Pendulum parameters.
        initial: Initial joint state.
        g_inplane: In-plane gravity 2-vector.
        dt: Step size [s].
        n_steps: Number of RK4 steps (>= 0).
        torque_fn: ``t -> (tau_shoulder, tau_wrist)`` [N·m].

    Returns:
        ``(n_steps + 1, 4)`` state rows including the initial state.
    """
    require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
    require(n_steps >= 0, "n_steps must be >= 0", n_steps)
    require(callable(torque_fn), "torque_fn must be callable")

    def f(t: float, y: np.ndarray) -> np.ndarray:
        tau = np.asarray(torque_fn(t), dtype=float)
        c = reference.coriolis_vector(p, y[1], y[2], y[3])
        g = reference.gravity_vector(p, y[0], y[1], g_inplane)
        d = reference.damping_vector(p, y[2], y[3])
        m = reference.mass_matrix(p, y[1])
        rhs = tau - np.asarray(c) - np.asarray(g) - np.asarray(d)
        acc = np.linalg.solve(m, rhs)
        return np.concatenate([y[2:], acc])

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


def _to_app(plane_r: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Map in-plane 2-vectors (local x, local up) into the app frame."""
    world = vectors[:, :1] * plane_r[:, 0] + vectors[:, 1:2] * plane_r[:, 2]
    return np.asarray(world @ APP_FROM_SWING.T)


def compute_kinetics(
    run: SimulationRun, clubhead_mass_kg: float = CLUBHEAD_MASS_KG
) -> KineticsSeries | None:
    """Swing kinetics of a run, or ``None`` when unsupported.

    Only the ``"double_pendulum"`` source exposes the joint states the
    EOM surfaces need; manual (no joint model) and triple-pendulum
    (separate absolute-angle formulation, deferred — SPEC.md H2 note)
    runs return ``None``.

    The source is rebuilt from the run's config (construction is
    deterministic), and the joint trajectory is read through the public
    ``DoublePendulumSwing.state_at`` accessor on the run's own sample
    grid, so the kinetics align sample-for-sample with the stored
    clubhead series.

    Args:
        run: A completed simulation run.
        clubhead_mass_kg: Point mass for the clubhead-force estimate.

    Returns:
        The kinetics series, or ``None`` for unsupported sources.
    """
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    require(
        math.isfinite(clubhead_mass_kg) and clubhead_mass_kg > 0.0,
        "clubhead_mass_kg must be finite and > 0",
        clubhead_mass_kg,
    )
    if run.config.source_kind != "double_pendulum":
        return None

    # Rebuild the deterministic source (session does not retain it).
    source = make_source(
        "double_pendulum",
        run.config.scenario,
        plane=run.config.plane,
        duration=run.config.swing_duration_s,
    )
    assert isinstance(source, AppFrameSwing)
    pendulum = source.inner
    assert isinstance(pendulum, DoublePendulumSwing)

    times = np.asarray(run.swing_times, dtype=float)
    dt = float(times[1] - times[0])
    duration = float(pendulum.duration)
    states = np.array(
        [
            (s.theta1, s.theta2, s.omega1, s.omega2)
            for s in (pendulum.state_at(min(float(t), duration)) for t in times)
        ]
    )

    p = pendulum.parameters
    plane_r = reference.plane_rotation(
        pendulum.plane.yaw_rad,
        pendulum.plane.side_tilt_rad,
        pendulum.plane.forward_tilt_rad,
    )
    g_inplane = reference.in_plane_gravity(plane_r, reference_g())

    torques = inverse_dynamics(p, states, g_inplane, dt)
    theta, omega, alpha = states[:, :2], states[:, 2:], torques["alpha"]
    f_shoulder, f_wrist, f_head = _reaction_forces(
        p, theta, omega, alpha, g_inplane, clubhead_mass_kg
    )
    net = torques["inertial"]
    power = net * omega  # τ_net · ω per coordinate; sums to dKE/dt

    # Ball-aligned app-frame geometry: the run's stored positions carry
    # the scrubber offset; the pivot sits at the swing origin + offset.
    offset = (
        run.swing_positions[0]
        - APP_FROM_SWING @ (source.inner.sample(float(times[0])).pose[:3, 3])
    )
    wrist_local = np.stack(
        [p.l1 * np.sin(theta[:, 0]), -p.l1 * np.cos(theta[:, 0])], axis=1
    )
    return KineticsSeries(
        t=times,
        joint_names=KINETIC_JOINT_NAMES,
        torque_applied_nm=torques["applied"],
        torque_gravity_nm=torques["gravity"],
        torque_damping_nm=torques["damping"],
        torque_inertial_nm=net,
        power_w=power,
        shoulder_force_n=_to_app(plane_r, f_shoulder),
        wrist_force_n=_to_app(plane_r, f_wrist),
        clubhead_force_n=_to_app(plane_r, f_head),
        pivot_position_m=np.asarray(offset, dtype=float),
        wrist_positions_m=_to_app(plane_r, wrist_local) + offset,
        clubhead_positions_m=np.asarray(run.swing_positions, dtype=float),
        plane_x_app=np.asarray(APP_FROM_SWING @ plane_r[:, 0]),
        plane_up_app=np.asarray(APP_FROM_SWING @ plane_r[:, 2]),
        impact_time_s=float(run.impact_time_s),
    )


def reference_g() -> float:
    """Gravity magnitude used by the swing sources [m/s²]."""
    from shared.python.swing_sim.types import DEFAULT_GRAVITY_M_S2

    return float(DEFAULT_GRAVITY_M_S2)


# Single-slot cache: the UI recomputes kinetics for the same run from
# several widgets (plots, overlay, panel); runs are immutable.
_CACHE: dict[str, object] = {"id": None, "series": None}


def kinetics_for_run(run: SimulationRun) -> KineticsSeries | None:
    """Cached :func:`compute_kinetics` (single-slot, keyed by identity)."""
    require(isinstance(run, SimulationRun), "run must be a SimulationRun", run)
    if _CACHE["id"] == id(run):
        return _CACHE["series"]  # type: ignore[return-value]
    series = compute_kinetics(run)
    _CACHE["id"] = id(run)
    _CACHE["series"] = series
    return series
