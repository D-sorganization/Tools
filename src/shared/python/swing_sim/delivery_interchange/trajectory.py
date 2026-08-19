"""Neutral delivery-trajectory contract (club-tester C5, #4554).

The seam between full-body biomechanics models and the impact + flight
pipeline. Any motion source — Drake, MuJoCo, OpenSim, a launch-monitor
rig, or an OEM tool — that can write this wire feeds the same fitting
machinery; the engine adapters in :mod:`.adapters` produce it from each
engine's documented export.

Frames (declared, not assumed):

- **World** is the AffineDrift frame the impact package documents:
  ``x`` = target line, ``y`` = up, ``z`` = right (right-handed).
- **Grip frame**: origin at the butt, ``+z`` along the shaft toward the
  head, ``+x`` toward the face normal of a square face. Orientation is a
  unit quaternion ``(w, x, y, z)`` mapping grip coordinates into world.

Wire ``swing_sim.delivery_trajectory/1``: strictly increasing timestamps,
at least two samples, finite SI values, deterministic sorted-keys JSON,
unknown fields refused — the same fail-closed posture as the golf_club
wires.

Derivations (:func:`head_state_at`, :func:`grip_kinematics_at`,
:func:`delivery_view_at`) extend the rigid grip frame down the shaft:
``p_head = p + R·(0, 0, ℓ)`` and ``v_head = v + ω × (R·(0, 0, ℓ))`` for a
declared grip-to-head distance ``ℓ`` — exact for a rigid club; the C2
shaft model adds the compliance corrections on top. Angular acceleration
is a central difference of ``|ω|(t)``, one-sided at the ends.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.python.contracts import ensure, require

DELIVERY_TRAJECTORY_FORMAT = "swing_sim.delivery_trajectory/1"

_SAMPLE_FIELDS = frozenset(
    {
        "time_s",
        "position_m",
        "quaternion_wxyz",
        "linear_velocity_mps",
        "angular_velocity_rad_s",
    }
)
_TRAJECTORY_FIELDS = frozenset({"format", "source_id", "frame_id", "samples"})

__all__ = [
    "DELIVERY_TRAJECTORY_FORMAT",
    "DeliveryTrajectory",
    "DeliveryView",
    "TrajectorySample",
    "delivery_trajectory_from_json",
    "delivery_trajectory_to_json",
    "delivery_view_at",
    "grip_kinematics_at",
    "head_state_at",
]


def _finite_triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a 3-vector")
    require(len(value) == 3, f"{name} must be a 3-vector")
    items = tuple(float(item) for item in value)
    require(all(math.isfinite(item) for item in items), f"{name} must be finite")
    return (items[0], items[1], items[2])


@dataclass(frozen=True)
class TrajectorySample:
    """One time-stamped grip-frame state in the declared world frame."""

    time_s: float
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        require(
            isinstance(self.time_s, (float, int)) and math.isfinite(self.time_s),
            "time_s must be finite",
        )
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(
            self, "position_m", _finite_triplet(self.position_m, "position_m")
        )
        object.__setattr__(
            self,
            "linear_velocity_mps",
            _finite_triplet(self.linear_velocity_mps, "linear_velocity_mps"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_triplet(self.angular_velocity_rad_s, "angular_velocity_rad_s"),
        )
        quaternion = self.quaternion_wxyz
        require(
            isinstance(quaternion, (tuple, list)) and len(quaternion) == 4,
            "quaternion_wxyz must have four components",
        )
        values = tuple(float(item) for item in quaternion)
        require(
            all(math.isfinite(item) for item in values),
            "quaternion_wxyz must be finite",
        )
        norm = math.sqrt(sum(item * item for item in values))
        require(
            abs(norm - 1.0) < 1e-6,
            "quaternion_wxyz must be unit length",
            norm,
        )
        object.__setattr__(self, "quaternion_wxyz", values)

    def rotation_matrix(self) -> np.ndarray:
        """World-from-grip rotation matrix for this sample."""
        w, x, y, z = self.quaternion_wxyz
        rotation: np.ndarray = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return rotation


@dataclass(frozen=True)
class DeliveryTrajectory:
    """A validated grip-frame trajectory in a declared world frame."""

    source_id: str
    frame_id: str
    samples: tuple[TrajectorySample, ...]

    def __post_init__(self) -> None:
        require(
            isinstance(self.source_id, str)
            and self.source_id.strip() == self.source_id
            and self.source_id != "",
            "source_id must be a trimmed nonempty string",
        )
        require(
            isinstance(self.frame_id, str)
            and self.frame_id.strip() == self.frame_id
            and self.frame_id != "",
            "frame_id must be a trimmed nonempty string",
        )
        require(
            isinstance(self.samples, tuple)
            and len(self.samples) >= 2
            and all(isinstance(item, TrajectorySample) for item in self.samples),
            "samples must be a tuple of at least two TrajectorySample records",
        )
        times = [sample.time_s for sample in self.samples]
        pairs = zip(times, times[1:], strict=False)
        require(
            all(later > earlier for earlier, later in pairs),
            "sample times must be strictly increasing",
        )

    def index_at_time(self, time_s: float) -> int:
        """Index of the sample closest to ``time_s``."""
        require(
            isinstance(time_s, (float, int)) and math.isfinite(time_s),
            "time_s must be finite",
        )
        deltas = [abs(sample.time_s - float(time_s)) for sample in self.samples]
        return deltas.index(min(deltas))


def delivery_trajectory_to_json(trajectory: DeliveryTrajectory) -> str:
    """Serialize with deterministic key ordering and no non-finite values."""
    require(
        isinstance(trajectory, DeliveryTrajectory),
        "trajectory must be DeliveryTrajectory",
    )
    payload: dict[str, Any] = {
        "format": DELIVERY_TRAJECTORY_FORMAT,
        "source_id": trajectory.source_id,
        "frame_id": trajectory.frame_id,
        "samples": [
            {
                "time_s": sample.time_s,
                "position_m": list(sample.position_m),
                "quaternion_wxyz": list(sample.quaternion_wxyz),
                "linear_velocity_mps": list(sample.linear_velocity_mps),
                "angular_velocity_rad_s": list(sample.angular_velocity_rad_s),
            }
            for sample in trajectory.samples
        ],
    }
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def delivery_trajectory_from_json(text: str) -> DeliveryTrajectory:
    """Parse and validate; unknown fields and wrong formats are refused."""
    require(isinstance(text, str), "text must be str")
    data = json.loads(text)
    require(isinstance(data, dict), "delivery trajectory must be an object")
    unknown = set(data) - _TRAJECTORY_FIELDS
    require(not unknown, f"unknown delivery-trajectory fields: {sorted(unknown)}")
    require(
        data.get("format") == DELIVERY_TRAJECTORY_FORMAT,
        f"format must be {DELIVERY_TRAJECTORY_FORMAT!r}",
    )
    raw_samples = data.get("samples")
    require(isinstance(raw_samples, list), "samples must be a list")
    samples = []
    for raw in raw_samples:
        require(isinstance(raw, dict), "each sample must be an object")
        unknown_sample = set(raw) - _SAMPLE_FIELDS
        require(not unknown_sample, f"unknown sample fields: {sorted(unknown_sample)}")
        samples.append(
            TrajectorySample(
                time_s=raw.get("time_s"),
                position_m=tuple(raw.get("position_m", ())),
                quaternion_wxyz=tuple(raw.get("quaternion_wxyz", ())),
                linear_velocity_mps=tuple(raw.get("linear_velocity_mps", ())),
                angular_velocity_rad_s=tuple(raw.get("angular_velocity_rad_s", ())),
            )
        )
    return DeliveryTrajectory(
        source_id=data.get("source_id"),
        frame_id=data.get("frame_id"),
        samples=tuple(samples),
    )


def head_state_at(
    trajectory: DeliveryTrajectory,
    index: int,
    *,
    grip_to_head_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-extension head position and velocity at one sample.

    ``p_head = p + R·(0, 0, ℓ)``; ``v_head = v + ω × (R·(0, 0, ℓ))``.
    """
    require(isinstance(trajectory, DeliveryTrajectory), "trajectory required")
    require(
        isinstance(index, int) and 0 <= index < len(trajectory.samples),
        "index out of range",
    )
    require(
        isinstance(grip_to_head_m, (float, int))
        and math.isfinite(grip_to_head_m)
        and grip_to_head_m > 0.0,
        "grip_to_head_m must be positive and finite",
    )
    sample = trajectory.samples[index]
    arm = sample.rotation_matrix() @ np.array([0.0, 0.0, float(grip_to_head_m)])
    position = np.asarray(sample.position_m) + arm
    velocity = np.asarray(sample.linear_velocity_mps) + np.cross(
        np.asarray(sample.angular_velocity_rad_s), arm
    )
    ensure(
        bool(np.isfinite(position).all() and np.isfinite(velocity).all()),
        "head state must be finite",
    )
    return position, velocity


def grip_kinematics_at(
    trajectory: DeliveryTrajectory,
    index: int,
    *,
    grip_to_head_m: float,
) -> dict[str, float]:
    """Scalar swing state for the shaft-delivery model at one sample.

    Returns ``omega_rad_s`` (angular speed), ``alpha_rad_s2`` (central
    difference of angular speed, one-sided at the ends), and
    ``swing_radius_m`` (instantaneous center distance ``|v_head|/ω``,
    exact for circular motion). Returned as a plain dict so this shared
    package does not import golf_club; callers construct
    ``golf_club.GripKinematics(**result)``.
    """
    _, head_velocity = head_state_at(trajectory, index, grip_to_head_m=grip_to_head_m)
    sample = trajectory.samples[index]
    omega = float(np.linalg.norm(sample.angular_velocity_rad_s))
    require(omega > 0.0, "angular speed must be positive at the requested sample")

    speeds = [
        float(np.linalg.norm(item.angular_velocity_rad_s))
        for item in trajectory.samples
    ]
    times = [item.time_s for item in trajectory.samples]
    if index == 0:
        alpha = (speeds[1] - speeds[0]) / (times[1] - times[0])
    elif index == len(speeds) - 1:
        alpha = (speeds[-1] - speeds[-2]) / (times[-1] - times[-2])
    else:
        alpha = (speeds[index + 1] - speeds[index - 1]) / (
            times[index + 1] - times[index - 1]
        )
    swing_radius = float(np.linalg.norm(head_velocity)) / omega
    ensure(swing_radius > 0.0, "swing radius must be positive")
    return {
        "omega_rad_s": omega,
        "alpha_rad_s2": float(alpha),
        "swing_radius_m": swing_radius,
    }


@dataclass(frozen=True)
class DeliveryView:
    """Launch-monitor-style view of one trajectory sample (rigid club)."""

    clubhead_speed_mps: float
    attack_angle_deg: float
    club_path_deg: float
    face_angle_deg: float


def delivery_view_at(
    trajectory: DeliveryTrajectory,
    index: int,
    *,
    grip_to_head_m: float,
) -> DeliveryView:
    """Derive the rigid-club delivery numbers at one sample.

    Attack angle from the head velocity's vertical component; club path
    from its lateral component (+ = in-to-out, +z); face angle from the
    grip frame's ``+x`` (face-normal) axis projected on the horizontal
    plane (+ = open, +z) — the AffineDrift sign conventions the impact
    package documents. The C2 shaft deltas apply on top of these.
    """
    _, head_velocity = head_state_at(trajectory, index, grip_to_head_m=grip_to_head_m)
    speed = float(np.linalg.norm(head_velocity))
    require(speed > 0.0, "head speed must be positive at the requested sample")
    vx, vy, vz = (float(item) for item in head_velocity)
    horizontal = math.hypot(vx, vz)
    require(horizontal > 0.0, "head velocity must have a horizontal component")
    attack_deg = math.degrees(math.atan2(vy, horizontal))
    path_deg = math.degrees(math.atan2(vz, vx))
    face_normal = trajectory.samples[index].rotation_matrix() @ np.array(
        [1.0, 0.0, 0.0]
    )
    face_deg = math.degrees(math.atan2(float(face_normal[2]), float(face_normal[0])))
    return DeliveryView(
        clubhead_speed_mps=speed,
        attack_angle_deg=attack_deg,
        club_path_deg=path_deg,
        face_angle_deg=face_deg,
    )
