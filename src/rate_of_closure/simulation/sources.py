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
  :class:`~rate_of_closure.model.ImpactScenario` as a declared
  constant-twist delivery: straight-line reference travel at a selected
  attack angle/path plus a rigid forward-lean head pose at mid-window.
* ``DoublePendulumSwing`` (from ``swing_sim``) via :func:`make_source`.
* :class:`TriplePendulumSwing` — planar three-link pendulum on the same
  oriented plane, integrated with the same classical RK4. New model
  (the shared package only ships the double pendulum, recon #4105).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.simulation.manual_delivery import (
    ManualDeliveryConfig,
    manual_head_rotation,
    manual_reference_velocity,
)
from rate_of_closure.simulation.triple_pendulum import (
    TRIPLE_PENDULUM_JOINT_IDS,
    TriplePendulumParameters,
    TriplePendulumSwing,
)
from rate_of_closure.simulation.triple_pendulum import (
    triple_derivatives as triple_derivatives,
)
from rate_of_closure.simulation.triple_pendulum import (
    triple_total_energy as triple_total_energy,
)
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_JOINT_IDS,
    DoublePendulumRunConfig,
    SwingRunMode,
)
from shared.python.swing_sim.swing_source import DoublePendulumSwing, SwingSource
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.types import (
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
    SwingSample,
)

__all__ = [
    "SOURCE_KINDS",
    "MANUAL_SWING_DURATION_S",
    "APP_FROM_SWING",
    "AppFrameSwing",
    "ManualSwingSource",
    "TriplePendulumParameters",
    "TriplePendulumSwing",
    "commanded_torque_joint_ids",
    "generalized_state_layout",
    "make_source",
]

#: Swing-source kinds accepted by :func:`make_source`, in UI order.
SOURCE_KINDS: tuple[str, ...] = ("manual", "double_pendulum", "triple_pendulum")

_STATE_LAYOUTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "manual": ((), ()),
    "double_pendulum": (
        (
            "joint.shoulder.angle_rad",
            "joint.wrist.relative_angle_rad",
            "joint.shoulder.rate_rad_s",
            "joint.wrist.relative_rate_rad_s",
        ),
        ("rad", "rad", "rad/s", "rad/s"),
    ),
    "triple_pendulum": (
        (
            "joint.shoulder.absolute_angle_rad",
            "joint.elbow.absolute_angle_rad",
            "joint.wrist.absolute_angle_rad",
            "joint.shoulder.absolute_rate_rad_s",
            "joint.elbow.absolute_rate_rad_s",
            "joint.wrist.absolute_rate_rad_s",
        ),
        ("rad", "rad", "rad", "rad/s", "rad/s", "rad/s"),
    ),
}


def generalized_state_layout(kind: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stable generalized-state IDs and units for a source kind."""
    require(kind in _STATE_LAYOUTS, "unknown swing source kind", kind)
    return _STATE_LAYOUTS[kind]


def commanded_torque_joint_ids(kind: str) -> tuple[str, ...]:
    """Return stable externally applied generalized-torque joint IDs."""
    if kind == "double_pendulum":
        return cast(tuple[str, ...], DOUBLE_PENDULUM_JOINT_IDS)
    if kind == "triple_pendulum":
        return cast(tuple[str, ...], TRIPLE_PENDULUM_JOINT_IDS)
    require(kind == "manual", "unknown swing source kind", kind)
    return ()


#: Canonical duration of the manual source's centered inspection window [s].
MANUAL_SWING_DURATION_S = 0.06

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
        identity: np.ndarray = np.eye(3)
        return identity
    axis = axis_omega / np.linalg.norm(axis_omega)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation: np.ndarray = np.asarray(
        np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)
    )
    return rotation


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

    @property
    def generalized_state_ids(self) -> tuple[str, ...]:
        """Forward frame-invariant generalized-state component IDs."""
        if isinstance(self._inner, DoublePendulumSwing):
            return generalized_state_layout("double_pendulum")[0]
        identifiers = getattr(self._inner, "generalized_state_ids", ())
        return tuple(cast(tuple[str, ...], identifiers))

    @property
    def generalized_state_units(self) -> tuple[str, ...]:
        """Forward units aligned with generalized-state component IDs."""
        if isinstance(self._inner, DoublePendulumSwing):
            return generalized_state_layout("double_pendulum")[1]
        units = getattr(self._inner, "generalized_state_units", ())
        return tuple(cast(tuple[str, ...], units))

    def generalized_state_at(self, t: float) -> np.ndarray:
        """Forward generalized coordinates, which are frame invariant."""
        if isinstance(self._inner, DoublePendulumSwing):
            state = self._inner.state_at(t)
            return cast(
                np.ndarray,
                np.asarray(
                    (state.theta1, state.theta2, state.omega1, state.omega2),
                    dtype=np.float64,
                ),
            )
        state_sampler = getattr(self._inner, "generalized_state_at", None)
        require(
            callable(state_sampler), "inner source has no generalized-state history"
        )
        sampler = cast(Callable[[float], object], state_sampler)
        values: np.ndarray = np.asarray(sampler(t), dtype=np.float64)
        return values


class ManualSwingSource:
    """Constant-twist source built from an :class:`ImpactScenario` (app frame).

    The clubhead reference point follows the declared attack angle and path at
    the scenario speed while the head spins at the scenario's constant angular
    velocity. The rigid head reaches the declared forward-lean pose with its
    reference at the origin at the window midpoint, matching the explorer's
    "instant of maximum compression" convention. Zero-valued declarations
    preserve the historical target-line/identity-pose source exactly.
    """

    def __init__(
        self,
        scenario: ImpactScenario,
        duration: float = MANUAL_SWING_DURATION_S,
        *,
        delivery: ManualDeliveryConfig | None = None,
    ) -> None:
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
        self._delivery = delivery or ManualDeliveryConfig()
        result = solve(scenario)
        self._impact_rotation = manual_head_rotation(self._delivery)
        self._omega = self._impact_rotation @ np.radians(np.array(result.omega_dps))
        speed_mps = result.reference_speed_mph * 0.44704
        self._velocity = manual_reference_velocity(speed_mps, self._delivery)

    @property
    def scenario(self) -> ImpactScenario:
        """The wrapped scenario."""
        return self._scenario

    @property
    def duration(self) -> float:
        """Window length [s]."""
        return self._duration

    @property
    def uses_declared_head_pose(self) -> bool:
        """Declare that delivery extraction must honor this source's pose."""
        return True

    @property
    def generalized_state_ids(self) -> tuple[str, ...]:
        """Manual rigid motion has no additional generalized coordinates."""
        return ()

    @property
    def generalized_state_units(self) -> tuple[str, ...]:
        """Return the empty units schema paired with the empty state."""
        return ()

    def generalized_state_at(self, t: float) -> np.ndarray:
        """Validate ``t`` and return the empty manual generalized state."""
        self.sample(t)
        return np.empty((0,), dtype=np.float64)

    def sample(self, t: float) -> SwingSample:
        """Clubhead sample at ``t``; at the declared pose at window midpoint."""
        require(math.isfinite(t), "t must be finite", t)
        require(
            -1e-9 <= t <= self._duration + 1e-9,
            "t must be within [0, duration]",
            t,
        )
        t = min(max(t, 0.0), self._duration)
        dt = t - self._duration / 2.0
        pose = np.eye(4)
        pose[:3, :3] = _rodrigues(self._omega, dt) @ self._impact_rotation
        pose[:3, 3] = self._velocity * dt
        twist = np.concatenate([self._omega, self._velocity])
        return SwingSample(t=t, pose=pose, twist=twist)


def make_source(
    kind: str,
    scenario: ImpactScenario,
    plane: PlaneOrientation | None = None,
    duration: float = 1.5,
    run_config: DoublePendulumRunConfig | None = None,
    torque_library: TorqueProfileLibrary | None = None,
    pendulum_parameters: PendulumParameters | None = None,
    manual_delivery: ManualDeliveryConfig | None = None,
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
    require(
        run_config is None or isinstance(run_config, DoublePendulumRunConfig),
        "run_config must be a DoublePendulumRunConfig or None",
        run_config,
    )
    require(kind in SOURCE_KINDS, f"unknown swing source kind {kind!r}", kind)
    execution = DoublePendulumRunConfig() if run_config is None else run_config
    uses_default_mode = (
        execution.mode is SwingRunMode.PASSIVE
        and execution.prescribed_profile_id is None
    )
    require(
        kind == "double_pendulum" or uses_default_mode,
        "non-default execution policy is unsupported outside the "
        "double-pendulum source",
        kind,
    )
    require(
        kind == "double_pendulum" or not execution.joint_locks.has_locks,
        "joint locks are unsupported outside the double-pendulum source",
        kind,
    )
    require(
        kind == "double_pendulum" or not execution.commanded_torque_offsets,
        "localized torque offsets are unsupported outside the double-pendulum source",
        kind,
    )
    if kind == "manual":
        return ManualSwingSource(scenario, delivery=manual_delivery)
    if kind == "double_pendulum":
        # Start on the target side (theta1 = -pi/2, arm horizontal) so
        # the gravity-driven downswing carries the clubhead TOWARD the
        # target (+x) at the bottom of the arc.
        start = PendulumState(theta1=-math.pi / 2.0, theta2=0.0, omega1=0.0, omega2=0.0)
        return AppFrameSwing(
            DoublePendulumSwing(
                plane=plane,
                parameters=pendulum_parameters,
                initial_state=start,
                duration=duration,
                backend="auto",
                run_config=execution,
                torque_library=torque_library,
            )
        )
    return AppFrameSwing(TriplePendulumSwing(plane=plane, duration=duration))
