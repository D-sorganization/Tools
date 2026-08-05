"""Internal swing/contact/impact/flight orchestration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.club import face_normal_at_offset
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation.contact import (
    ContactMode,
    ImpactOutcome,
    assess_fixed_contact,
    forced_alignment_outcome,
)
from rate_of_closure.simulation.delivery import delivery_at
from rate_of_closure.simulation.records import (
    BALL_POSITION_M,
    SimulationConfig,
    SimulationRun,
)
from shared.python.swing_sim.flight import (
    derive_launch_conditions,
    from_flight_frame,
    to_flight_frame,
)
from shared.python.swing_sim.flight import simulate as flight_simulate
from shared.python.swing_sim.impact import (
    GOLF_BALL_RADIUS_M,
    DeliveryDerived,
    ImpactModelType,
    ImpactParameters,
    ImpactSolverAPI,
    PostImpactState,
)
from shared.python.swing_sim.swing_source import SwingSource

_SAMPLE_DT_S = 1e-3


@dataclass(frozen=True)
class _SwingSeries:
    """Complete sampled swing arrays before or after optional translation."""

    times: np.ndarray
    poses: np.ndarray
    twists: np.ndarray
    joints: np.ndarray
    joint_ids: tuple[str, ...]
    applied_torques_nm: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        """Return clubhead-reference-point positions."""
        return self.poses[:, :3, 3].copy()

    def translated(self, offset_m: np.ndarray) -> _SwingSeries:
        """Return a copy translated by a constant app-frame offset."""
        poses = self.poses.copy()
        joints = self.joints.copy()
        poses[:, :3, 3] += offset_m
        if joints.shape[1] > 0:
            joints += offset_m
        return _SwingSeries(
            self.times,
            poses,
            self.twists,
            joints,
            self.joint_ids,
            self.applied_torques_nm,
        )


@dataclass(frozen=True)
class _ImpactProducts:
    """Optional post-contact phases of one simulation."""

    delivery: DeliveryDerived | None
    post_impact: PostImpactState | None
    launch: dict[str, float] | None
    flight_times: np.ndarray
    flight_positions: np.ndarray
    flight_velocities: np.ndarray


def execute_simulation(config: SimulationConfig) -> SimulationRun:
    """Build a complete run while treating no-contact as a valid result."""
    source = _make_source(config)
    swing = _sample_swing(source)
    outcome, impact_time_s, aligned_swing = _select_contact(config, source, swing)
    products = (
        _solve_hit(config, source, impact_time_s)
        if impact_time_s is not None
        else _empty_impact_products()
    )
    return _assemble_run(config, aligned_swing, outcome, impact_time_s, products)


def _make_source(config: SimulationConfig) -> SwingSource:
    """Construct the configured app-frame swing source."""
    from rate_of_closure.simulation.sources import make_source

    return cast(
        SwingSource,
        make_source(
            config.source_kind,
            config.scenario,
            plane=config.plane,
            duration=config.swing_duration_s,
            run_config=config.swing_run_config,
            torque_library=config.torque_library,
        ),
    )


def _sample_swing(source: SwingSource) -> _SwingSeries:
    """Retain all source samples independently of the contact result."""
    sample_count = int(round(source.duration / _SAMPLE_DT_S))
    times = np.linspace(0.0, source.duration, max(sample_count, 2) + 1)
    samples = [source.sample(float(time_s)) for time_s in times]
    poses = np.stack([sample.pose for sample in samples])
    twists = np.stack([sample.twist for sample in samples])
    joint_sampler = getattr(source, "joint_positions", None)
    joints = (
        np.stack([np.asarray(joint_sampler(float(time_s))) for time_s in times])
        if callable(joint_sampler)
        else np.zeros((len(times), 0, 3))
    )
    joint_ids = tuple(getattr(source, "joint_ids", ()))
    applied_torques_nm = _sample_applied_torques(source, times, joint_ids)
    return _SwingSeries(
        times,
        poses,
        twists,
        joints,
        joint_ids,
        applied_torques_nm,
    )


def _sample_applied_torques(
    source: SwingSource, times: np.ndarray, joint_ids: tuple[str, ...]
) -> np.ndarray:
    """Sample one stable-ID torque mapping per swing time."""
    torque_sampler = getattr(source, "joint_torques_at", None)
    if not joint_ids or not callable(torque_sampler):
        return np.zeros((len(times), 0), dtype=np.float64)
    rows: list[list[float]] = []
    for time_s in times:
        values = torque_sampler(float(time_s))
        require(
            isinstance(values, dict) and set(values) == set(joint_ids),
            "joint torque samples must match stable joint IDs exactly",
            values,
        )
        rows.append([float(values[joint_id]) for joint_id in joint_ids])
    return np.asarray(rows, dtype=np.float64)


def _select_contact(
    config: SimulationConfig, source: SwingSource, swing: _SwingSeries
) -> tuple[ImpactOutcome, float | None, _SwingSeries]:
    """Choose forced inspection alignment or sampled fixed-ball contact."""
    if config.contact_mode is ContactMode.FIXED_BALL_CONTACT:
        outcome = assess_fixed_contact(
            swing.times, swing.positions, BALL_POSITION_M, GOLF_BALL_RADIUS_M
        )
        impact_time_s = outcome.candidate_time_s if outcome.is_hit else None
        return outcome, impact_time_s, swing
    impact_time_s = _inspection_time(config, source, swing)
    offset = BALL_POSITION_M - source.sample(impact_time_s).pose[:3, 3]
    ensure(
        bool(
            np.allclose(
                source.sample(impact_time_s).pose[:3, 3] + offset,
                BALL_POSITION_M,
                atol=1e-9,
            )
        ),
        "clubhead at tau must coincide with the ball position",
    )
    outcome = forced_alignment_outcome(
        impact_time_s, BALL_POSITION_M, GOLF_BALL_RADIUS_M
    )
    return outcome, impact_time_s, swing.translated(offset)


def _inspection_time(
    config: SimulationConfig, source: SwingSource, swing: _SwingSeries
) -> float:
    """Return the requested or maximum-speed inspection instant."""
    if config.impact_time_s is not None:
        return min(max(float(config.impact_time_s), 0.0), float(source.duration))
    speeds = np.linalg.norm(swing.twists[:, 3:], axis=1)
    return float(swing.times[int(np.argmax(speeds))])


def _solve_hit(
    config: SimulationConfig, source: SwingSource, impact_time_s: float
) -> _ImpactProducts:
    """Run delivery, rigid-body impact, launch, and flight for one hit."""
    delivery = delivery_at(source, impact_time_s, config.scenario, config.club)
    solver = ImpactSolverAPI(
        ImpactModelType.RIGID_BODY,
        ImpactParameters(cg_depth=config.club.cg_depth_m),
    )
    post = solver.solve_with_gear_effect(
        timestamp=impact_time_s,
        clubhead_velocity=delivery.clubhead_velocity,
        clubhead_orientation=delivery.face_normal,
        impact_offset=delivery.impact_offset,
        clubhead_mass=config.club.head_mass_kg,
        clubhead_moi=config.club.moi_about_shaft_kg_m2,
        face_normal_at_offset=_face_normal_callable(config),
        record=False,
    )
    flight = _simulate_flight(post, config.flight_model)
    times, positions, velocities = _flight_arrays(flight)
    return _ImpactProducts(
        delivery,
        post,
        _launch_summary(post, delivery, flight),
        times,
        positions,
        velocities,
    )


def _face_normal_callable(config: SimulationConfig):  # type: ignore[no-untyped-def]
    """Return the club's bulge/roll normal callback for the impact solver."""

    def _normal(toe_m: float, high_m: float) -> np.ndarray:
        return np.array(face_normal_at_offset(config.club, toe_m * 1e3, high_m * 1e3))

    return _normal


def _simulate_flight(post: PostImpactState, model_name: str):  # type: ignore[no-untyped-def]
    """Integrate flight from a post-impact state."""
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    return flight_simulate(launch, model_name=model_name)


def _flight_arrays(flight: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flight-model output to app-frame arrays."""
    trajectory = flight.trajectory  # type: ignore[attr-defined]
    times = np.array([point.time for point in trajectory])
    if not trajectory:
        return times, np.zeros((0, 3)), np.zeros((0, 3))
    positions = from_flight_frame(flight.to_position_array()) + BALL_POSITION_M  # type: ignore[attr-defined]
    velocities = from_flight_frame(np.array([point.velocity for point in trajectory]))
    return times, positions, velocities


def _launch_summary(
    post: PostImpactState, delivery: DeliveryDerived, flight: object
) -> dict[str, float]:
    """Flatten launch and flight metrics into the export summary."""
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    return {
        "ball_speed_mph": launch.ball_speed * MPH_PER_MPS,
        "launch_angle_deg": math.degrees(launch.launch_angle),
        "launch_azimuth_deg": -math.degrees(launch.azimuth_angle),
        "spin_rpm": launch.spin_rate,
        "spin_axis_tilt_deg": delivery.spin_axis_tilt_deg,
        "carry_m": float(flight.carry_distance),  # type: ignore[attr-defined]
        "max_height_m": float(flight.max_height),  # type: ignore[attr-defined]
        "flight_time_s": float(flight.flight_time),  # type: ignore[attr-defined]
        "landing_angle_deg": float(flight.landing_angle),  # type: ignore[attr-defined]
    }


def _empty_impact_products() -> _ImpactProducts:
    """Return typed empty downstream phases for a miss."""
    return _ImpactProducts(
        None,
        None,
        None,
        np.zeros((0,)),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
    )


def _assemble_run(
    config: SimulationConfig,
    swing: _SwingSeries,
    outcome: ImpactOutcome,
    impact_time_s: float | None,
    products: _ImpactProducts,
) -> SimulationRun:
    """Assemble the immutable public simulation record."""
    return SimulationRun(
        config=config,
        swing_times=swing.times,
        swing_positions=swing.positions,
        swing_poses=swing.poses,
        swing_twists=swing.twists,
        swing_joints=swing.joints,
        swing_joint_ids=swing.joint_ids,
        swing_applied_torques_nm=swing.applied_torques_nm,
        impact_outcome=outcome,
        impact_time_s=impact_time_s,
        delivery=products.delivery,
        post_impact=products.post_impact,
        launch=products.launch,
        flight_times=products.flight_times,
        flight_positions=products.flight_positions,
        flight_velocities=products.flight_velocities,
    )
