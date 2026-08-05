"""Swing sources: ``t -> clubhead pose + twist`` abstraction (epic #4103 P1).

:class:`SwingSource` is the protocol every swing generator implements —
manual deliveries, pendulum models, and (later) the movement optimizer.
:class:`DoublePendulumSwing` is the first concrete source: a double pendulum
swinging on an oriented plane, integrated once at construction (one-shot,
so backend "auto" may fall back to the pure-Python reference when the Rust
wheel is absent) and sampled by linear interpolation afterwards.
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from shared.python.contracts import require

from . import _rust_facade, reference
from .run_config import (
    DOUBLE_PENDULUM_JOINT_IDS,
    DOUBLE_PENDULUM_MODEL_ID,
    SHOULDER_JOINT_ID,
    WRIST_JOINT_ID,
    DoublePendulumRunConfig,
    SwingRunMode,
)
from .torque_library import TorqueProfileLibrary
from .torque_profiles import PrescribedTorqueProfile
from .types import (
    DEFAULT_GRAVITY_M_S2,
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
    SwingSample,
)

Backend = Literal["auto", "rust", "python"]


def _resolve_prescribed_profile(
    config: DoublePendulumRunConfig,
    library: TorqueProfileLibrary | None,
    duration: float,
) -> PrescribedTorqueProfile | None:
    """Resolve and validate the exact profile contract for this kernel."""
    if config.mode is SwingRunMode.PASSIVE:
        return None
    if library is None:
        require(False, "prescribed mode requires a torque profile library")
        raise AssertionError("unreachable")
    require(
        isinstance(library, TorqueProfileLibrary),
        "torque_library must be a TorqueProfileLibrary",
        library,
    )
    profile = library.get(config.prescribed_profile_id or "")
    require(
        profile.model_id == DOUBLE_PENDULUM_MODEL_ID,
        "prescribed profile model_id is incompatible with double pendulum",
        profile.model_id,
    )
    joint_ids = tuple(assignment.joint_id for assignment in profile.assignments)
    require(
        set(joint_ids) == set(DOUBLE_PENDULUM_JOINT_IDS),
        "prescribed profile joint IDs must match the double-pendulum joints",
        joint_ids,
    )
    start_s, end_s = profile.time_domain_s
    require(
        start_s <= 0.0 and end_s >= duration,
        "prescribed profile time domain must cover the complete run",
        profile.time_domain_s,
    )
    return profile


@runtime_checkable
class SwingSource(Protocol):
    """Anything that can be sampled for a clubhead pose + twist over time."""

    @property
    def duration(self) -> float:
        """Total duration [s] of the swing."""
        ...

    def sample(self, t: float) -> SwingSample:
        """Return the clubhead sample at time ``t`` in ``[0, duration]``."""
        ...


def _local_axis_rotation(angle: float) -> np.ndarray:
    """Rotation about the plane-local normal axis (local y) by ``angle``."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class DoublePendulumSwing:
    """Double-pendulum swing on an oriented plane, as a :class:`SwingSource`.

    The trajectory is integrated once at construction on a uniform grid of
    step ``dt`` and sampled by linear interpolation of the joint state.

    Backend policy (strict-Rust posture for hot loops):
    - ``"rust"``: require the ``swing_core`` wheel; raise ``ImportError``
      when absent.
    - ``"python"``: always use the pure-Python reference (parity oracle).
    - ``"auto"`` (default): use Rust when available, else fall back — this
      constructor is a one-shot call, so the explicit fallback is allowed.
    """

    def __init__(
        self,
        parameters: PendulumParameters | None = None,
        plane: PlaneOrientation | None = None,
        initial_state: PendulumState | None = None,
        duration: float = 1.5,
        dt: float = 1e-3,
        gravity_m_s2: float = DEFAULT_GRAVITY_M_S2,
        backend: Backend = "auto",
        run_config: DoublePendulumRunConfig | None = None,
        torque_library: TorqueProfileLibrary | None = None,
    ) -> None:
        require(
            math.isfinite(duration) and duration > 0.0,
            "duration must be finite and > 0",
            duration,
        )
        require(math.isfinite(dt) and dt > 0.0, "dt must be finite and > 0", dt)
        require(dt <= duration, "dt must not exceed duration", dt)
        require(
            math.isfinite(gravity_m_s2) and gravity_m_s2 >= 0.0,
            "gravity_m_s2 must be finite and >= 0",
            gravity_m_s2,
        )
        require(backend in ("auto", "rust", "python"), "invalid backend", backend)
        config = run_config or DoublePendulumRunConfig()
        require(
            isinstance(config, DoublePendulumRunConfig),
            "run_config must be a DoublePendulumRunConfig",
            config,
        )
        require(
            torque_library is None or isinstance(torque_library, TorqueProfileLibrary),
            "torque_library must be a TorqueProfileLibrary",
            torque_library,
        )

        self._parameters = parameters or PendulumParameters.golf_default()
        self._plane = plane or PlaneOrientation()
        self._initial_state = initial_state or PendulumState(
            theta1=math.pi / 2.0, theta2=0.0, omega1=0.0, omega2=0.0
        )
        self._dt = float(dt)
        self._n_steps = int(round(duration / dt))
        self._duration = self._n_steps * self._dt
        self._run_config = config
        self._torque_profile = _resolve_prescribed_profile(
            config, torque_library, self._duration
        )

        self._plane_r = reference.plane_rotation(
            self._plane.yaw_rad, self._plane.side_tilt_rad, self._plane.forward_tilt_rad
        )
        self._g_inplane = reference.in_plane_gravity(self._plane_r, gravity_m_s2)

        require(
            not (self._torque_profile is not None and backend == "rust"),
            "prescribed torque integration is not supported by the Rust backend",
        )
        if self._torque_profile is not None:
            self._states = reference.simulate_forced(
                self._parameters,
                self._initial_state,
                self._g_inplane,
                self._dt,
                self._n_steps,
                self._prescribed_torque_tuple,
            )
            self._backend: Backend = "python"
        elif backend == "rust" or (backend == "auto" and _rust_facade.rust_available()):
            self._states = _rust_facade.simulate_rust(
                self._parameters,
                self._initial_state,
                self._g_inplane,
                self._dt,
                self._n_steps,
            )
            self._backend = "rust"
        else:
            self._states = reference.simulate(
                self._parameters,
                self._initial_state,
                self._g_inplane,
                self._dt,
                self._n_steps,
            )
            self._backend = "python"

    @property
    def backend(self) -> Backend:
        """Backend that produced the trajectory (``"rust"`` or ``"python"``)."""
        return self._backend

    @property
    def run_mode(self) -> SwingRunMode:
        """Validated dynamics execution mode for this trajectory."""
        return self._run_config.mode

    @property
    def joint_ids(self) -> tuple[str, str]:
        """Stable kernel ordering for generalized joint torques."""
        return DOUBLE_PENDULUM_JOINT_IDS

    @property
    def parameters(self) -> PendulumParameters:
        """Pendulum parameters used for the integration."""
        return self._parameters

    @property
    def plane(self) -> PlaneOrientation:
        """Swing-plane orientation."""
        return self._plane

    @property
    def duration(self) -> float:
        """Total duration [s] of the integrated swing."""
        return self._duration

    def _prescribed_torque_tuple(self, time_s: float) -> tuple[float, float]:
        """Return kernel-ordered torques for an already-resolved profile."""
        profile = self._torque_profile
        if profile is None:
            require(False, "prescribed torque profile is not configured")
            raise AssertionError("unreachable")
        require(
            -1e-9 <= time_s <= self._duration + 1e-9,
            "integration torque time must be within the run",
            time_s,
        )
        sample_time = min(max(time_s, 0.0), self._duration)
        values = profile.evaluate(sample_time)
        return values[SHOULDER_JOINT_ID], values[WRIST_JOINT_ID]

    def joint_torques_at(self, t: float) -> dict[str, float]:
        """Return stable-ID joint torques at a time within the run."""
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        sample_time = min(max(t, 0.0), self._duration)
        if self._torque_profile is None:
            return {SHOULDER_JOINT_ID: 0.0, WRIST_JOINT_ID: 0.0}
        return self._torque_profile.evaluate(sample_time)

    def _state_at(self, t: float) -> tuple[float, float, float, float]:
        """Linearly interpolate the joint state at time ``t``."""
        idx = t / self._dt
        i0 = min(int(idx), self._n_steps)
        i1 = min(i0 + 1, self._n_steps)
        frac = idx - i0
        row = (1.0 - frac) * self._states[i0] + frac * self._states[i1]
        return float(row[0]), float(row[1]), float(row[2]), float(row[3])

    def state_at(self, t: float) -> PendulumState:
        """Interpolated joint state at time ``t`` (public, for kinetics).

        Exposes the integrated joint trajectory so downstream inverse
        dynamics (rate_of_closure swing kinetics, epic #4125 H2) can
        evaluate the EOM surfaces without re-integrating.

        Preconditions: ``0 <= t <= duration`` (within float tolerance).
        """
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        t = min(max(t, 0.0), self._duration)
        theta1, theta2, omega1, omega2 = self._state_at(t)
        return PendulumState(theta1=theta1, theta2=theta2, omega1=omega1, omega2=omega2)

    def joint_positions(self, t: float) -> np.ndarray:
        """Pivot, wrist, and clubhead positions in the swing world frame."""
        state = self.state_at(t)
        p = self._parameters
        t12 = state.theta1 + state.theta2
        x_axis = self._plane_r[:, 0]
        up_axis = self._plane_r[:, 2]
        wrist = p.l1 * (
            math.sin(state.theta1) * x_axis - math.cos(state.theta1) * up_axis
        )
        tip = wrist + p.l2 * (math.sin(t12) * x_axis - math.cos(t12) * up_axis)
        return np.vstack([np.zeros(3), wrist, tip])

    def sample(self, t: float) -> SwingSample:
        """Return the clubhead :class:`SwingSample` at time ``t``.

        Preconditions: ``0 <= t <= duration`` (within float tolerance).
        """
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        t = min(max(t, 0.0), self._duration)
        theta1, theta2, omega1, omega2 = self._state_at(t)

        p = self._parameters
        t12 = theta1 + theta2
        # In-plane clubhead position (local x horizontal, local y up).
        x = p.l1 * math.sin(theta1) + p.l2 * math.sin(t12)
        y = -(p.l1 * math.cos(theta1) + p.l2 * math.cos(t12))
        # In-plane clubhead velocity.
        vx = p.l1 * math.cos(theta1) * omega1 + p.l2 * math.cos(t12) * (omega1 + omega2)
        vy = p.l1 * math.sin(theta1) * omega1 + p.l2 * math.sin(t12) * (omega1 + omega2)

        r_plane = self._plane_r
        x_axis = r_plane[:, 0]
        normal = r_plane[:, 1]
        up_axis = r_plane[:, 2]

        position = x * x_axis + y * up_axis
        rotation = r_plane @ _local_axis_rotation(t12)

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position

        angular = (omega1 + omega2) * normal
        linear = vx * x_axis + vy * up_axis
        twist = np.concatenate([angular, linear])

        return SwingSample(t=t, pose=pose, twist=twist)
