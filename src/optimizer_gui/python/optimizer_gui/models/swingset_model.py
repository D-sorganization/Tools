"""Side-view swingset kinematics and control-policy rollouts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .chain_model import GRAVITY_M_S2, ChainConfig, ChainState

FloatArray: TypeAlias = NDArray[np.float64]
Policy = Callable[["SwingSetState", float], "SwingControlAction"]

DEFAULT_CHAIN_SEGMENTS: Final[int] = 14
DEFAULT_CHAIN_LENGTH_M: Final[float] = 2.4
DEFAULT_SEAT_MASS_KG: Final[float] = 4.5
DEFAULT_LINK_MASS_KG: Final[float] = 0.16
DEFAULT_DAMPING: Final[float] = 0.04
DEFAULT_PUMP_GAIN: Final[float] = 0.65
MAX_BODY_RATE_RAD_S: Final[float] = 2.4
MAX_EXTERNAL_TORQUE_NM: Final[float] = 70.0
CONTROL_DIMENSION: Final[int] = 5


def _require_positive(name: str, value: float) -> None:
    """Enforce a positive scalar precondition."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` into the closed interval ``[lower, upper]``."""
    return min(max(value, lower), upper)


def _segment_vector(length_m: float, angle_rad: float) -> FloatArray:
    """Return a side-view vector for an angle measured from vertical-down."""
    return np.asarray(
        [length_m * np.sin(angle_rad), length_m * np.cos(angle_rad)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class HumanSegmentSpec:
    """Length and mass for one body segment.

    Preconditions:
        ``length_m`` and ``mass_kg`` must be positive finite values.
    """

    length_m: float
    mass_kg: float

    def __post_init__(self) -> None:
        _require_positive("length_m", self.length_m)
        _require_positive("mass_kg", self.mass_kg)


@dataclass(frozen=True)
class SwingSetConfig:
    """Physical configuration for a trainable swingset model."""

    chain_segments: int = DEFAULT_CHAIN_SEGMENTS
    chain_length_m: float = DEFAULT_CHAIN_LENGTH_M
    chain_link_mass_kg: float = DEFAULT_LINK_MASS_KG
    seat_mass_kg: float = DEFAULT_SEAT_MASS_KG
    torso: HumanSegmentSpec = HumanSegmentSpec(0.62, 28.0)
    thigh: HumanSegmentSpec = HumanSegmentSpec(0.46, 8.0)
    shank: HumanSegmentSpec = HumanSegmentSpec(0.45, 5.5)
    upper_arm: HumanSegmentSpec = HumanSegmentSpec(0.30, 2.0)
    forearm: HumanSegmentSpec = HumanSegmentSpec(0.28, 1.6)
    gravity_m_s2: float = GRAVITY_M_S2
    damping: float = DEFAULT_DAMPING
    pump_gain: float = DEFAULT_PUMP_GAIN

    def __post_init__(self) -> None:
        if self.chain_segments < 1:
            raise ValueError("chain_segments must be at least 1")
        _require_positive("chain_length_m", self.chain_length_m)
        _require_positive("chain_link_mass_kg", self.chain_link_mass_kg)
        _require_positive("seat_mass_kg", self.seat_mass_kg)
        _require_positive("gravity_m_s2", self.gravity_m_s2)
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.pump_gain < 0.0:
            raise ValueError("pump_gain must be non-negative")

    def chain_config(self) -> ChainConfig:
        """Return the shared chain configuration for this swingset."""
        return ChainConfig(
            segment_count=self.chain_segments,
            segment_length_m=self.chain_length_m / self.chain_segments,
            link_mass_kg=self.chain_link_mass_kg,
            gravity_m_s2=self.gravity_m_s2,
            damping=self.damping,
        )

    @property
    def rider_mass_kg(self) -> float:
        """Return total modelled rider mass in kilograms."""
        paired_arms = 2.0 * (self.upper_arm.mass_kg + self.forearm.mass_kg)
        paired_legs = 2.0 * (self.thigh.mass_kg + self.shank.mass_kg)
        return self.torso.mass_kg + paired_arms + paired_legs


@dataclass(frozen=True)
class SwingPose:
    """Generalized coordinates for the side-view human and swing."""

    swing_angle_rad: float = 0.0
    torso_lean_rad: float = 0.0
    hip_angle_rad: float = 0.0
    knee_angle_rad: float = 0.0
    shoulder_angle_rad: float = 0.0
    elbow_angle_rad: float = 0.0


@dataclass(frozen=True)
class SwingControlAction:
    """Control rates and external torque applied during one rollout step."""

    torso_lean_rate_rad_s: float = 0.0
    hip_rate_rad_s: float = 0.0
    knee_rate_rad_s: float = 0.0
    shoulder_rate_rad_s: float = 0.0
    elbow_rate_rad_s: float = 0.0
    external_torque_nm: float = 0.0

    def clipped(self) -> SwingControlAction:
        """Return a control action constrained to the model limits."""
        rate = MAX_BODY_RATE_RAD_S
        torque = MAX_EXTERNAL_TORQUE_NM
        return SwingControlAction(
            torso_lean_rate_rad_s=_clamp(self.torso_lean_rate_rad_s, -rate, rate),
            hip_rate_rad_s=_clamp(self.hip_rate_rad_s, -rate, rate),
            knee_rate_rad_s=_clamp(self.knee_rate_rad_s, -rate, rate),
            shoulder_rate_rad_s=_clamp(self.shoulder_rate_rad_s, -rate, rate),
            elbow_rate_rad_s=_clamp(self.elbow_rate_rad_s, -rate, rate),
            external_torque_nm=_clamp(self.external_torque_nm, -torque, torque),
        )

    def vector(self) -> FloatArray:
        """Return the trainable control-rate vector used by policy search."""
        return np.asarray(
            [
                self.torso_lean_rate_rad_s,
                self.hip_rate_rad_s,
                self.knee_rate_rad_s,
                self.shoulder_rate_rad_s,
                self.elbow_rate_rad_s,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SwingSetState:
    """Dynamic state for the swingset rollout."""

    pose: SwingPose
    swing_angular_velocity_rad_s: float = 0.0

    @classmethod
    def rest(cls) -> SwingSetState:
        """Return a vertical static swingset state."""
        return cls(pose=SwingPose(), swing_angular_velocity_rad_s=0.0)


@dataclass(frozen=True)
class SwingSetSnapshot:
    """Kinematic geometry derived from a swingset pose."""

    points: dict[str, FloatArray]
    chain_nodes: FloatArray
    center_of_mass_m: FloatArray
    hand_chain_error_m: float


@dataclass(frozen=True)
class SwingRolloutMetrics:
    """Summary metrics returned after policy rollout."""

    max_abs_swing_angle_rad: float
    final_energy_proxy_j: float
    mean_hand_chain_error_m: float


@dataclass(frozen=True)
class SwingRollout:
    """Time history for a swingset policy simulation."""

    states: tuple[SwingSetState, ...]
    swing_angles_rad: FloatArray
    controls: FloatArray
    snapshots: tuple[SwingSetSnapshot, ...]
    metrics: SwingRolloutMetrics


def build_swingset_snapshot(
    config: SwingSetConfig,
    pose: SwingPose,
) -> SwingSetSnapshot:
    """Build side-view chain and rider geometry for ``pose``."""
    chain_state = ChainState.stationary(config.chain_config(), pose.swing_angle_rad)
    chain_nodes = chain_state.node_positions(config.chain_config())
    seat = chain_nodes[-1]
    points = _body_points(config, pose, seat)
    center = _center_of_mass(config, points)
    grip_point = chain_nodes[max(config.chain_segments - 2, 0)]
    hand_error = float(np.linalg.norm(points["hand"] - grip_point))
    return SwingSetSnapshot(
        points=points,
        chain_nodes=chain_nodes,
        center_of_mass_m=center,
        hand_chain_error_m=hand_error,
    )


def _body_points(
    config: SwingSetConfig,
    pose: SwingPose,
    seat: FloatArray,
) -> dict[str, FloatArray]:
    """Return rider joint points keyed by anatomical label."""
    hip = seat.copy()
    torso_angle = pose.swing_angle_rad + pose.torso_lean_rad + np.pi
    shoulder = hip + _segment_vector(config.torso.length_m, torso_angle)
    thigh_angle = pose.swing_angle_rad + pose.hip_angle_rad
    knee = hip + _segment_vector(config.thigh.length_m, thigh_angle)
    shank_angle = thigh_angle + pose.knee_angle_rad
    foot = knee + _segment_vector(config.shank.length_m, shank_angle)
    upper_arm_angle = pose.swing_angle_rad + pose.shoulder_angle_rad
    elbow = shoulder + _segment_vector(config.upper_arm.length_m, upper_arm_angle)
    forearm_angle = upper_arm_angle + pose.elbow_angle_rad
    hand = elbow + _segment_vector(config.forearm.length_m, forearm_angle)
    return {
        "seat": seat,
        "hip": hip,
        "shoulder": shoulder,
        "knee": knee,
        "foot": foot,
        "elbow": elbow,
        "hand": hand,
    }


def _center_of_mass(
    config: SwingSetConfig,
    points: dict[str, FloatArray],
) -> FloatArray:
    """Return approximate rider plus seat center of mass."""
    seat = points["seat"]
    torso_mid = 0.5 * (points["hip"] + points["shoulder"])
    thigh_mid = 0.5 * (points["hip"] + points["knee"])
    shank_mid = 0.5 * (points["knee"] + points["foot"])
    upper_arm_mid = 0.5 * (points["shoulder"] + points["elbow"])
    forearm_mid = 0.5 * (points["elbow"] + points["hand"])
    weighted = (
        config.seat_mass_kg * seat
        + config.torso.mass_kg * torso_mid
        + 2.0 * config.thigh.mass_kg * thigh_mid
        + 2.0 * config.shank.mass_kg * shank_mid
        + 2.0 * config.upper_arm.mass_kg * upper_arm_mid
        + 2.0 * config.forearm.mass_kg * forearm_mid
    )
    total_mass = config.seat_mass_kg + config.rider_mass_kg
    return weighted / total_mass


def _step_pose(pose: SwingPose, action: SwingControlAction, dt_s: float) -> SwingPose:
    """Integrate controlled pose coordinates while preserving swing angle."""
    return replace(
        pose,
        torso_lean_rad=pose.torso_lean_rad + action.torso_lean_rate_rad_s * dt_s,
        hip_angle_rad=pose.hip_angle_rad + action.hip_rate_rad_s * dt_s,
        knee_angle_rad=pose.knee_angle_rad + action.knee_rate_rad_s * dt_s,
        shoulder_angle_rad=pose.shoulder_angle_rad + action.shoulder_rate_rad_s * dt_s,
        elbow_angle_rad=pose.elbow_angle_rad + action.elbow_rate_rad_s * dt_s,
    )


def _swing_acceleration(
    config: SwingSetConfig,
    state: SwingSetState,
    action: SwingControlAction,
) -> float:
    """Return angular acceleration for the swing coordinate."""
    length = config.chain_length_m
    gravity = -(config.gravity_m_s2 / length) * np.sin(state.pose.swing_angle_rad)
    damping = -config.damping * state.swing_angular_velocity_rad_s
    inertia = (config.seat_mass_kg + config.rider_mass_kg) * length**2
    control = action.external_torque_nm / inertia
    pump = config.pump_gain * _pumping_projection(state, action) / length
    return float(gravity + damping + control + pump)


def _pumping_projection(state: SwingSetState, action: SwingControlAction) -> float:
    """Project body-rate controls into swing-driving acceleration."""
    direction = 1.0 if state.swing_angular_velocity_rad_s >= 0.0 else -1.0
    leg_drive = action.hip_rate_rad_s - 0.35 * action.knee_rate_rad_s
    torso_drive = -0.5 * action.torso_lean_rate_rad_s
    return direction * (leg_drive + torso_drive)


def step_swingset(
    config: SwingSetConfig,
    state: SwingSetState,
    action: SwingControlAction,
    dt_s: float,
) -> SwingSetState:
    """Advance the swingset by one semi-implicit Euler step."""
    _require_positive("dt_s", dt_s)
    clipped = action.clipped()
    acceleration = _swing_acceleration(config, state, clipped)
    next_velocity = state.swing_angular_velocity_rad_s + acceleration * dt_s
    next_swing = state.pose.swing_angle_rad + next_velocity * dt_s
    pose = replace(_step_pose(state.pose, clipped, dt_s), swing_angle_rad=next_swing)
    return SwingSetState(pose=pose, swing_angular_velocity_rad_s=next_velocity)


def heuristic_pumping_policy(
    state: SwingSetState,
    _time_s: float,
) -> SwingControlAction:
    """Return a deterministic baseline policy that pumps the swing."""
    direction = 1.0 if state.swing_angular_velocity_rad_s >= 0.0 else -1.0
    return SwingControlAction(
        torso_lean_rate_rad_s=-0.7 * direction,
        hip_rate_rad_s=0.9 * direction,
        knee_rate_rad_s=-0.4 * direction,
        shoulder_rate_rad_s=-0.15 * direction,
        elbow_rate_rad_s=0.2 * direction,
    )


def simulate_swingset(
    config: SwingSetConfig,
    initial_state: SwingSetState,
    steps: int,
    dt_s: float,
    policy: Policy,
) -> SwingRollout:
    """Roll out a control policy for a trainable swingset model."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    states = [initial_state]
    controls: list[FloatArray] = []
    snapshots = [build_swingset_snapshot(config, initial_state.pose)]
    for step_index in range(steps):
        time_s = step_index * dt_s
        action = policy(states[-1], time_s).clipped()
        controls.append(action.vector())
        states.append(step_swingset(config, states[-1], action, dt_s))
        snapshots.append(build_swingset_snapshot(config, states[-1].pose))
    angles = np.asarray([state.pose.swing_angle_rad for state in states])
    control_array = np.vstack(controls).reshape(steps, CONTROL_DIMENSION)
    metrics = _rollout_metrics(config, states, snapshots, angles)
    return SwingRollout(
        states=tuple(states),
        swing_angles_rad=angles,
        controls=control_array,
        snapshots=tuple(snapshots),
        metrics=metrics,
    )


def _rollout_metrics(
    config: SwingSetConfig,
    states: list[SwingSetState],
    snapshots: list[SwingSetSnapshot],
    angles: FloatArray,
) -> SwingRolloutMetrics:
    """Build scalar metrics for comparing control policies."""
    final_velocity = states[-1].swing_angular_velocity_rad_s
    total_mass = config.seat_mass_kg + config.rider_mass_kg
    inertia = total_mass * config.chain_length_m**2
    energy = 0.5 * inertia * final_velocity**2
    errors = np.asarray([snapshot.hand_chain_error_m for snapshot in snapshots])
    return SwingRolloutMetrics(
        max_abs_swing_angle_rad=float(np.max(np.abs(angles))),
        final_energy_proxy_j=float(energy),
        mean_hand_chain_error_m=float(np.mean(errors)),
    )
