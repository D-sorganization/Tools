"""Planar chain dynamics used by chain and swingset analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

GRAVITY_M_S2: Final[float] = 9.80665
DEFAULT_SEGMENT_COUNT: Final[int] = 12
DEFAULT_SEGMENT_LENGTH_M: Final[float] = 0.18
DEFAULT_LINK_MASS_KG: Final[float] = 0.12
DEFAULT_DAMPING: Final[float] = 0.08
DEFAULT_COUPLING: Final[float] = 18.0
MIN_SEGMENTS_FOR_CATENARY: Final[int] = 2
INTEGRATION_SUBSTEPS: Final[int] = 32

FloatArray: TypeAlias = NDArray[np.float64]


def _require_positive(name: str, value: float) -> None:
    """Enforce a positive scalar precondition."""
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _as_float_array(name: str, values: FloatArray, expected_size: int) -> FloatArray:
    """Validate a finite 1-D float array with a required length."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if array.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class ChainConfig:
    """Physical configuration for a side-view segmented chain.

    Preconditions:
        ``segment_count`` is at least 1. Length, mass, gravity, damping, and
        coupling are non-negative where physically required.
    """

    segment_count: int = DEFAULT_SEGMENT_COUNT
    segment_length_m: float = DEFAULT_SEGMENT_LENGTH_M
    link_mass_kg: float = DEFAULT_LINK_MASS_KG
    gravity_m_s2: float = GRAVITY_M_S2
    damping: float = DEFAULT_DAMPING
    coupling: float = DEFAULT_COUPLING

    def __post_init__(self) -> None:
        if self.segment_count < 1:
            raise ValueError("segment_count must be at least 1")
        _require_positive("segment_length_m", self.segment_length_m)
        _require_positive("link_mass_kg", self.link_mass_kg)
        _require_positive("gravity_m_s2", self.gravity_m_s2)
        if self.damping < 0.0:
            raise ValueError("damping must be non-negative")
        if self.coupling < 0.0:
            raise ValueError("coupling must be non-negative")

    @property
    def total_length_m(self) -> float:
        """Return the full chain length in metres."""
        return self.segment_count * self.segment_length_m


@dataclass(frozen=True)
class ChainStateMetrics:
    """Derived chain metrics useful for whip-motion analysis."""

    tip_speed_m_s: float
    center_of_mass_m: FloatArray
    max_curvature_rad: float


@dataclass(frozen=True)
class ChainState:
    """Angles and angular velocities for all chain links.

    Angles are measured from vertical-down in radians. Positive angles move
    the link endpoint toward positive x in the side-view plane.
    """

    angles_rad: FloatArray
    angular_velocities_rad_s: FloatArray

    @classmethod
    def stationary(cls, config: ChainConfig, angle_rad: float = 0.0) -> ChainState:
        """Build a stationary chain with every segment at the same angle."""
        angles = np.full(config.segment_count, angle_rad, dtype=np.float64)
        velocities = np.zeros(config.segment_count, dtype=np.float64)
        return cls(angles_rad=angles, angular_velocities_rad_s=velocities)

    def validated(self, config: ChainConfig) -> ChainState:
        """Return a state with validated arrays for ``config``."""
        angles = _as_float_array("angles_rad", self.angles_rad, config.segment_count)
        velocities = _as_float_array(
            "angular_velocities_rad_s",
            self.angular_velocities_rad_s,
            config.segment_count,
        )
        return ChainState(angles_rad=angles, angular_velocities_rad_s=velocities)

    def node_positions(
        self,
        config: ChainConfig,
        anchor_xy_m: tuple[float, float] = (0.0, 0.0),
    ) -> FloatArray:
        """Return anchor plus link endpoint coordinates as an ``(n + 1, 2)`` array."""
        state = self.validated(config)
        offsets = segment_vectors(config, state.angles_rad)
        endpoints = np.vstack(
            (
                np.zeros((1, 2), dtype=np.float64),
                np.cumsum(offsets, axis=0),
            )
        )
        anchor = np.asarray(anchor_xy_m, dtype=np.float64)
        if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
            raise ValueError("anchor_xy_m must contain two finite coordinates")
        return endpoints + anchor

    def metrics(self, config: ChainConfig) -> ChainStateMetrics:
        """Return tip speed, center of mass, and max adjacent-link curvature."""
        positions = self.node_positions(config)
        velocities = node_velocities(config, self.validated(config))
        midpoint_array = link_midpoints(positions)
        curvature = np.diff(self.angles_rad)
        max_curvature = float(np.max(np.abs(curvature))) if curvature.size else 0.0
        return ChainStateMetrics(
            tip_speed_m_s=float(np.linalg.norm(velocities[-1])),
            center_of_mass_m=np.mean(midpoint_array, axis=0),
            max_curvature_rad=max_curvature,
        )


@dataclass(frozen=True)
class ChainRollout:
    """Time history returned by :func:`simulate_chain`."""

    states: tuple[ChainState, ...]
    positions: FloatArray
    energy_j: FloatArray
    tip_speed_m_s: FloatArray


def segment_vectors(config: ChainConfig, angles_rad: FloatArray) -> FloatArray:
    """Return per-link x/y vectors from vertical-down link angles."""
    angles = _as_float_array("angles_rad", angles_rad, config.segment_count)
    return config.segment_length_m * np.column_stack((np.sin(angles), np.cos(angles)))


def link_midpoints(node_positions: FloatArray) -> FloatArray:
    """Return link midpoint coordinates from anchor/end-node coordinates."""
    nodes = np.asarray(node_positions, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 2:
        raise ValueError("node_positions must have shape (n >= 2, 2)")
    return 0.5 * (nodes[:-1] + nodes[1:])


def node_velocities(config: ChainConfig, state: ChainState) -> FloatArray:
    """Return anchor plus endpoint velocities for the chain state."""
    checked = state.validated(config)
    derivatives = config.segment_length_m * np.column_stack(
        (
            np.cos(checked.angles_rad) * checked.angular_velocities_rad_s,
            -np.sin(checked.angles_rad) * checked.angular_velocities_rad_s,
        )
    )
    return np.vstack(
        (
            np.zeros((1, 2), dtype=np.float64),
            np.cumsum(derivatives, axis=0),
        )
    )


def total_energy(config: ChainConfig, state: ChainState) -> float:
    """Return approximate chain mechanical energy in joules."""
    checked = state.validated(config)
    positions = checked.node_positions(config)
    velocities = node_velocities(config, checked)
    mass_positions = positions[1:]
    mass_velocities = velocities[1:]
    kinetic = 0.5 * config.link_mass_kg * np.sum(mass_velocities**2)
    potential_height = config.total_length_m - mass_positions[:, 1]
    potential = config.link_mass_kg * config.gravity_m_s2 * np.sum(potential_height)
    return float(kinetic + potential)


def initial_catenary_angles(segment_count: int, sag_rad: float) -> FloatArray:
    """Return a symmetric initial angle profile for a sagging chain."""
    if segment_count < MIN_SEGMENTS_FOR_CATENARY:
        raise ValueError("segment_count must be at least 2")
    if sag_rad < 0.0:
        raise ValueError("sag_rad must be non-negative")
    return np.linspace(-sag_rad, sag_rad, segment_count, dtype=np.float64)


def _angular_acceleration(
    config: ChainConfig,
    state: ChainState,
    torques_nm: FloatArray,
) -> FloatArray:
    """Compute a stable coupled-pendulum acceleration approximation."""
    checked = state.validated(config)
    torques = _as_float_array("torques_nm", torques_nm, config.segment_count)
    inertia = config.link_mass_kg * config.segment_length_m**2
    effective_lengths = config.segment_length_m * np.arange(
        1,
        config.segment_count + 1,
        dtype=np.float64,
    )
    gravity_term = -(config.gravity_m_s2 / effective_lengths) * np.sin(
        checked.angles_rad
    )
    damping_term = -config.damping * checked.angular_velocities_rad_s
    neighbor_sum = np.zeros(config.segment_count, dtype=np.float64)
    neighbor_sum[:-1] += checked.angles_rad[1:] - checked.angles_rad[:-1]
    neighbor_sum[1:] += checked.angles_rad[:-1] - checked.angles_rad[1:]
    coupling_term = config.coupling * neighbor_sum / config.link_mass_kg
    return gravity_term + damping_term + coupling_term + torques / inertia


def step_chain(
    config: ChainConfig,
    state: ChainState,
    dt_s: float,
    torques_nm: FloatArray | None = None,
) -> ChainState:
    """Advance the chain one semi-implicit Euler step."""
    _require_positive("dt_s", dt_s)
    checked = state.validated(config)
    torques = (
        np.zeros(config.segment_count, dtype=np.float64)
        if torques_nm is None
        else np.asarray(torques_nm, dtype=np.float64)
    )
    sub_dt = dt_s / INTEGRATION_SUBSTEPS
    current = checked
    for _substep in range(INTEGRATION_SUBSTEPS):
        acceleration = _angular_acceleration(config, current, torques)
        next_velocities = current.angular_velocities_rad_s + acceleration * sub_dt
        next_angles = current.angles_rad + next_velocities * sub_dt
        current = ChainState(next_angles, next_velocities)
    return current


def simulate_chain(
    config: ChainConfig,
    initial_state: ChainState,
    steps: int,
    dt_s: float,
    torque_history_nm: FloatArray | None = None,
) -> ChainRollout:
    """Simulate a chain for a fixed number of time steps."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    _require_positive("dt_s", dt_s)
    if torque_history_nm is None:
        torques = np.zeros((steps, config.segment_count), dtype=np.float64)
    else:
        torques = np.asarray(torque_history_nm, dtype=np.float64)
        if torques.shape != (steps, config.segment_count):
            raise ValueError("torque_history_nm has incompatible shape")
    states = [initial_state.validated(config)]
    for step_index in range(steps):
        states.append(step_chain(config, states[-1], dt_s, torques[step_index]))
    positions = np.stack([state.node_positions(config) for state in states])
    energy = np.asarray([total_energy(config, state) for state in states])
    tip_speed = np.asarray([state.metrics(config).tip_speed_m_s for state in states])
    return ChainRollout(
        states=tuple(states),
        positions=positions,
        energy_j=energy,
        tip_speed_m_s=tip_speed,
    )
