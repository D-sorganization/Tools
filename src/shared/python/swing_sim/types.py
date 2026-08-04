"""Core swing-simulation value types (frozen dataclasses with DbC validation).

All angles on public dataclasses are degrees (UI-friendly); radians are used
internally by the dynamics. SI units throughout (kg, m, s).

Conventions (mirrors ``rust_core/swing-core/src/swing``):
- World frame: x forward, y left, z up.
- Plane pose: three sequential intrinsic tilts — yaw about world-up, then
  side tilt about the rotated in-plane horizontal axis, then forward/back
  tilt about the resulting in-plane up axis.
- ``theta1`` measured from the in-plane downward vertical; ``theta2`` is the
  lower segment's angle relative to the upper segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from shared.python.contracts import require

DEFAULT_GRAVITY_M_S2 = 9.80665
"""Standard gravitational acceleration [m/s²]."""

# Defaults ported from UpstreamDrift double_pendulum.py (documented there).
_ARM_LENGTH_M = 0.75
_ARM_MASS_KG = 7.5
_ARM_COM_RATIO = 0.45
_ARM_INERTIA_SCALING = 1.0 / 12.0
_SHAFT_LENGTH_M = 1.0
_SHAFT_MASS_KG = 0.15
_CLUBHEAD_MASS_KG = 0.20
_SHAFT_COM_RATIO = 0.43
_DAMPING_SHOULDER = 0.4
_DAMPING_WRIST = 0.25


def _require_finite(name: str, value: float) -> None:
    require(math.isfinite(value), f"{name} must be finite", value)


@dataclass(frozen=True)
class PlaneOrientation:
    """Swing-plane pose as three sequential intrinsic tilts (degrees).

    Order (documented, epic #4103): yaw about world-up, then side tilt about
    the rotated in-plane horizontal axis, then forward/back tilt about the
    resulting in-plane up axis. The zero pose is a vertical plane feeling
    full gravity in-plane.
    """

    yaw_deg: float = 0.0
    side_tilt_deg: float = 0.0
    forward_tilt_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("yaw_deg", "side_tilt_deg", "forward_tilt_deg"):
            _require_finite(name, getattr(self, name))

    @property
    def yaw_rad(self) -> float:
        """Yaw angle in radians."""
        return math.radians(self.yaw_deg)

    @property
    def side_tilt_rad(self) -> float:
        """Side tilt angle in radians."""
        return math.radians(self.side_tilt_deg)

    @property
    def forward_tilt_rad(self) -> float:
        """Forward/back tilt angle in radians."""
        return math.radians(self.forward_tilt_deg)


@dataclass(frozen=True)
class PendulumParameters:
    """Double-pendulum physical parameters (SI units).

    Inertias ``i1``/``i2`` are about the proximal joint (parallel-axis
    already applied), matching the Rust kernel and the UpstreamDrift
    reference implementation.
    """

    m1: float
    l1: float
    lc1: float
    i1: float
    m2: float
    l2: float
    lc2: float
    i2: float
    d1: float = _DAMPING_SHOULDER
    d2: float = _DAMPING_WRIST

    def __post_init__(self) -> None:
        for name in ("m1", "l1", "lc1", "i1", "m2", "l2", "lc2", "i2"):
            value = getattr(self, name)
            _require_finite(name, value)
            require(value > 0.0, f"{name} must be > 0", value)
        for name in ("d1", "d2"):
            value = getattr(self, name)
            _require_finite(name, value)
            require(value >= 0.0, f"{name} must be >= 0", value)
        require(self.lc1 <= self.l1, "lc1 must not exceed l1", self.lc1)
        require(self.lc2 <= self.l2, "lc2 must not exceed l2", self.lc2)

    @classmethod
    def golf_default(cls) -> PendulumParameters:
        """Default golf-swing parameters (UpstreamDrift reference constants).

        Computed from the same segment formulas as the Rust kernel so the
        two backends agree bit-for-bit.
        """
        m1 = _ARM_MASS_KG
        l1 = _ARM_LENGTH_M
        lc1 = l1 * _ARM_COM_RATIO
        i1_com = _ARM_INERTIA_SCALING * m1 * l1 * l1
        i1 = i1_com + m1 * lc1 * lc1

        l2 = _SHAFT_LENGTH_M
        ms = _SHAFT_MASS_KG
        mh = _CLUBHEAD_MASS_KG
        m2 = ms + mh
        shaft_com = l2 * _SHAFT_COM_RATIO
        lc2 = (shaft_com * ms + l2 * mh) / m2
        shaft_inertia_com = (1.0 / 12.0) * ms * l2 * l2
        parallel_axis = ms * (shaft_com - lc2) * (shaft_com - lc2) + mh * (l2 - lc2) * (
            l2 - lc2
        )
        i2_com = shaft_inertia_com + parallel_axis
        i2 = i2_com + m2 * lc2 * lc2

        return cls(m1=m1, l1=l1, lc1=lc1, i1=i1, m2=m2, l2=l2, lc2=lc2, i2=i2)


@dataclass(frozen=True)
class PendulumState:
    """Planar dynamic state (radians, rad/s)."""

    theta1: float
    theta2: float
    omega1: float
    omega2: float

    def __post_init__(self) -> None:
        for name in ("theta1", "theta2", "omega1", "omega2"):
            _require_finite(name, getattr(self, name))


@dataclass(frozen=True)
class SwingSample:
    """One clubhead sample: time, SE(3) pose, and spatial twist.

    - ``pose``: 4x4 homogeneous transform (world-from-clubhead), rotation in
      the upper-left 3x3 block, position in the last column.
    - ``twist``: 6-vector ``[wx, wy, wz, vx, vy, vz]`` — world-frame angular
      velocity followed by clubhead linear velocity.
    """

    t: float
    pose: np.ndarray = field(repr=False)
    twist: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        _require_finite("t", self.t)
        require(self.t >= 0.0, "t must be >= 0", self.t)
        pose = np.asarray(self.pose, dtype=np.float64)
        twist = np.asarray(self.twist, dtype=np.float64)
        require(pose.shape == (4, 4), "pose must be a 4x4 matrix", pose.shape)
        require(twist.shape == (6,), "twist must be a 6-vector", twist.shape)
        require(bool(np.all(np.isfinite(pose))), "pose must be finite", None)
        require(bool(np.all(np.isfinite(twist))), "twist must be finite", None)
        require(
            bool(np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12)),
            "pose bottom row must be [0, 0, 0, 1]",
            None,
        )
        rotation = pose[:3, :3]
        require(
            bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)),
            "pose rotation block must be orthonormal",
            None,
        )
        object.__setattr__(self, "pose", pose)
        object.__setattr__(self, "twist", twist)


@dataclass(frozen=True)
class SwingTrajectory:
    """Immutable, time-ordered sequence of :class:`SwingSample`."""

    samples: tuple[SwingSample, ...]

    def __post_init__(self) -> None:
        require(len(self.samples) > 0, "trajectory must contain samples", None)
        times = [s.t for s in self.samples]
        require(
            all(b > a for a, b in zip(times, times[1:], strict=False)),
            "sample times must be strictly increasing",
            None,
        )

    @property
    def duration(self) -> float:
        """Trajectory duration [s] (time of last sample minus first)."""
        return self.samples[-1].t - self.samples[0].t

    def __len__(self) -> int:
        return len(self.samples)
