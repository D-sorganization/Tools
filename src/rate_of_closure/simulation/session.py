"""Simulation session: swing -> impact -> flight, one exportable record.

The orchestration seam of the epic (#4103): a :class:`SimulationConfig`
selects a swing source (manual constant-twist scenario, double pendulum,
or triple pendulum), a swing-plane orientation, a club, and a flight
model; :func:`run_simulation` samples the swing, forms the delivery at
the impact instant, solves the impact through ``swing_sim.impact`` (with
the club package's bulge/roll face-normal callable wired in), derives
launch conditions, and integrates the flight through
``swing_sim.flight``.

Impact-time scrubber convention: the ball sits at the FIXED world
position :data:`BALL_POSITION_M`; scrubbing the impact time ``tau``
translates the whole swing so the clubhead at ``tau`` meets the ball
(``offset = ball - clubhead(tau)``). :func:`delivery_at` returns the
live delivery numbers for any ``tau`` without running impact + flight.

Everything here is in the app frame (x target, y up, z right); frame
adapters convert at the flight boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure._contracts import ensure, require
from rate_of_closure.club import ClubSpec, face_normal_at_offset
from rate_of_closure.model import MPH_PER_MPS, ImpactScenario
from rate_of_closure.simulation.sources import make_source
from shared.python.swing_sim.flight import (
    derive_launch_conditions,
    from_flight_frame,
    to_flight_frame,
)
from shared.python.swing_sim.flight import simulate as flight_simulate
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import (
    GOLF_BALL_RADIUS_M,
    DeliveryDerived,
    DeliveryParameters,
    ImpactModelType,
    ImpactParameters,
    ImpactSolverAPI,
    PostImpactState,
    derive_delivery,
    face_basis,
)
from shared.python.swing_sim.impact_interval import (
    BoundaryKind,
    ClubRigidBody,
    ImpactIntervalConfig,
    ImpactIntervalInitialState,
    ImpactIntervalResult,
    KelvinVoigtContactLaw,
    solve_impact_interval,
)
from shared.python.swing_sim.swing_source import SwingSource
from shared.python.swing_sim.types import PlaneOrientation

__all__ = [
    "BALL_POSITION_M",
    "SimulationConfig",
    "SimulationRun",
    "delivery_at",
    "run_simulation",
]

#: Fixed world position of the ball (app frame): on the target line at
#: the origin, resting on the ground plane.
BALL_POSITION_M = np.array([0.0, GOLF_BALL_RADIUS_M, 0.0])

#: Delivery angles are clamped inside the impact package's ±89° domain.
_MAX_DELIVERY_ANGLE_DEG = 89.0

#: Uniform sampling step for the stored swing time series [s]. Dense
#: enough for screw-axis extraction near impact (recon #4108).
_SAMPLE_DT_S = 1e-3


@dataclass(frozen=True)
class SimulationConfig:
    """One simulation request.

    Args:
        scenario: The explorer scenario — the manual source wraps it as a
            constant twist; every source takes its impact offsets from it.
        club: Club spec used for loft, head mass, MOI, CG depth, and the
            bulge/roll face-normal callable.
        source_kind: ``"manual"``, ``"double_pendulum"``, or
            ``"triple_pendulum"``.
        plane: Swing-plane orientation (three sequential tilts).
        impact_time_s: Impact instant tau within the swing, seconds; the
            scrubber drives this. ``None`` picks the instant of maximum
            clubhead speed.
        flight_model: ``swing_sim.flight`` registry model name.
        swing_duration_s: Pendulum integration length [s].
    """

    scenario: ImpactScenario
    club: ClubSpec
    source_kind: str = "manual"
    plane: PlaneOrientation = field(default_factory=PlaneOrientation)
    impact_time_s: float | None = None
    flight_model: str = "waterloo_penner"
    swing_duration_s: float = 1.5
    impact_model: str = "instantaneous"
    impact_interval_config: ImpactIntervalConfig | None = None
    impact_interval_club: ClubRigidBody | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.scenario, ImpactScenario),
            "scenario must be an ImpactScenario",
            self.scenario,
        )
        require(isinstance(self.club, ClubSpec), "club must be a ClubSpec", self.club)
        FlightModelType(self.flight_model)  # raises ValueError on unknown names
        if self.impact_time_s is not None:
            require(
                math.isfinite(self.impact_time_s) and self.impact_time_s >= 0.0,
                "impact_time_s must be finite and >= 0",
                self.impact_time_s,
            )
        require(
            math.isfinite(self.swing_duration_s) and self.swing_duration_s > 0.0,
            "swing_duration_s must be finite and > 0",
            self.swing_duration_s,
        )
        require(
            self.impact_model in ("instantaneous", "impact_interval"),
            "impact_model must be 'instantaneous' or 'impact_interval'",
            self.impact_model,
        )
        if self.impact_interval_config is not None:
            require(
                isinstance(self.impact_interval_config, ImpactIntervalConfig),
                "impact_interval_config must be an ImpactIntervalConfig",
                self.impact_interval_config,
            )
        if self.impact_interval_club is not None:
            require(
                isinstance(self.impact_interval_club, ClubRigidBody),
                "impact_interval_club must be a ClubRigidBody",
                self.impact_interval_club,
            )


@dataclass(frozen=True)
class SimulationRun:
    """One complete swing -> impact -> flight record (app frame).

    Attributes:
        config: The request that produced this run.
        swing_times: (N,) swing sample times [s].
        swing_positions: (N, 3) clubhead positions, ball-aligned (the
            swing is translated so the clubhead meets the ball at tau).
        swing_poses: (N, 4, 4) ball-aligned SE(3) clubhead poses.
        swing_twists: (N, 6) world twists ``[wx, wy, wz, vx, vy, vz]``.
        impact_time_s: The impact instant tau actually used.
        delivery: Delivery numbers + D-plane diagnostics at tau.
        post_impact: Post-impact ball and club state.
        launch: Launch summary (mph / deg / rpm plus flight metrics).
        flight_times: (M,) flight sample times [s] (from impact).
        flight_positions: (M, 3) ball positions from the ball position.
        flight_velocities: (M, 3) ball velocities.
    """

    config: SimulationConfig
    swing_times: np.ndarray
    swing_positions: np.ndarray
    swing_poses: np.ndarray
    swing_twists: np.ndarray
    impact_time_s: float
    delivery: DeliveryDerived
    post_impact: PostImpactState
    impact_interval: ImpactIntervalResult | None
    launch: dict[str, float]
    flight_times: np.ndarray
    flight_positions: np.ndarray
    flight_velocities: np.ndarray

    @property
    def total_duration_s(self) -> float:
        """Swing duration plus flight time — the playback timeline span."""
        flight_span = float(self.flight_times[-1]) if len(self.flight_times) else 0.0
        return float(self.swing_times[-1]) + flight_span


def _clamped_angle(value_deg: float) -> float:
    """Clamp an angle into the impact package's delivery domain."""
    return max(-_MAX_DELIVERY_ANGLE_DEG, min(_MAX_DELIVERY_ANGLE_DEG, value_deg))


def _delivery_parameters(
    velocity: np.ndarray, scenario: ImpactScenario, club: ClubSpec
) -> DeliveryParameters:
    """Delivery numbers from a sampled clubhead velocity (app frame).

    The face is delivered square to the target (face angle 0) with the
    club's static loft as dynamic loft; path and attack angles come from
    the sampled velocity direction. Impact offsets carry over from the
    scenario. Residual lie rotation is zero — the offsets are already
    expressed in the face's toe/high axes.
    """
    speed = float(np.linalg.norm(velocity))
    require(speed > 1e-6, "clubhead speed at impact must be > 0", speed)
    path_deg = math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))
    aoa_deg = math.degrees(
        math.atan2(
            float(velocity[1]), math.hypot(float(velocity[0]), float(velocity[2]))
        )
    )
    return DeliveryParameters(
        clubhead_speed_mps=speed,
        club_path_deg=_clamped_angle(path_deg),
        face_angle_deg=0.0,
        attack_angle_deg=_clamped_angle(aoa_deg),
        dynamic_loft_deg=_clamped_angle(club.loft_deg),
        lie_deg=0.0,
        impact_offset_toe_mm=scenario.impact_offset_toe_mm,
        impact_offset_high_mm=scenario.impact_offset_high_mm,
    )


def delivery_at(
    source: SwingSource,
    tau: float,
    scenario: ImpactScenario,
    club: ClubSpec,
) -> DeliveryDerived:
    """Live delivery numbers for the scrubber at impact time ``tau``.

    Args:
        source: An app-frame swing source.
        tau: Impact instant within ``[0, source.duration]``.
        scenario: Scenario carrying the impact offsets.
        club: Club providing the dynamic loft.

    Returns:
        The delivery vectors + D-plane diagnostics at ``tau``.
    """
    sample = source.sample(tau)
    params = _delivery_parameters(sample.twist[3:], scenario, club)
    return derive_delivery(params, clubhead_angular_velocity=sample.twist[:3])


def _auto_impact_time(source: SwingSource, times: np.ndarray) -> float:
    """The sampled instant of maximum clubhead speed."""
    speeds = [float(np.linalg.norm(source.sample(float(t)).twist[3:])) for t in times]
    return float(times[int(np.argmax(speeds))])


def _face_normal_callable(club: ClubSpec):  # type: ignore[no-untyped-def]
    """Bulge/roll callable seam ``(toe_m, high_m) -> normal`` for the club."""

    def _normal(toe_m: float, high_m: float) -> np.ndarray:
        return np.array(face_normal_at_offset(club, toe_m * 1e3, high_m * 1e3))

    return _normal


def _default_interval_config(config: SimulationConfig) -> ImpactIntervalConfig:
    """Calibrate the compliant law to scenario contact time and driver COR."""
    duration = config.scenario.contact_duration_us * 1.0e-6
    require(duration > 0.0, "impact-interval contact duration must be > 0", duration)
    ball_mass = 0.04593
    reduced_mass = (
        ball_mass * config.club.head_mass_kg / (ball_mass + config.club.head_mass_kg)
    )
    restitution = ImpactParameters().cor
    log_e = math.log(restitution)
    damping_ratio = -log_e / math.sqrt(math.pi**2 + log_e**2)
    natural_frequency = math.pi / (duration * math.sqrt(1.0 - damping_ratio**2))
    stiffness = reduced_mass * natural_frequency**2
    law = KelvinVoigtContactLaw.from_restitution(
        stiffness_n_per_m=stiffness,
        restitution=restitution,
        effective_mass_kg=reduced_mass,
    )
    return ImpactIntervalConfig(
        contact_law=law,
        time_step_s=min(1.0e-7, duration / 2_000.0),
        maximum_time_s=max(3.0 * duration, 1.0e-3),
        friction_coefficient=ImpactParameters().friction_coefficient,
    )


def _default_interval_club(delivery: DeliveryDerived, club: ClubSpec) -> ClubRigidBody:
    """Adapt the scalar club catalog to a replaceable full-tensor body."""
    toe_m, high_m = delivery.impact_offset
    return ClubRigidBody(
        mass_kg=club.head_mass_kg,
        # Catalogs currently expose one measured axis. Isotropic expansion is
        # explicit and replaceable through SimulationConfig.impact_interval_club.
        inertia_body_kg_m2=club.moi_about_shaft_kg_m2 * np.eye(3),
        cg_to_contact_body_m=np.array([club.cg_depth_m, high_m, toe_m]),
        cg_to_attachment_body_m=np.array([0.0, club.length_m, 0.0]),
        face_normal_body=np.array([1.0, 0.0, 0.0]),
    )


def _solve_interval(
    config: SimulationConfig, delivery: DeliveryDerived
) -> ImpactIntervalResult:
    """Build a contact-consistent initial state and run the interval façade."""
    interval_config = config.impact_interval_config or _default_interval_config(config)
    body = config.impact_interval_club or _default_interval_club(delivery, config.club)
    toe_axis, up_axis = face_basis(delivery.face_normal)
    orientation = np.column_stack((delivery.face_normal, up_axis, toe_axis))
    contact_offset = orientation @ body.cg_to_contact_body_m
    club_position = (
        BALL_POSITION_M - GOLF_BALL_RADIUS_M * delivery.face_normal - contact_offset
    )
    club_omega = delivery.clubhead_angular_velocity.copy()
    club_velocity = delivery.clubhead_velocity.copy()
    if interval_config.boundary is not BoundaryKind.FREE:
        r_attachment = orientation @ body.cg_to_attachment_body_m
        pivot_to_contact = contact_offset - r_attachment
        radius_sq = float(np.dot(pivot_to_contact, pivot_to_contact))
        require(radius_sq > 1.0e-12, "attachment and contact must be distinct")
        driven_omega = (
            np.cross(pivot_to_contact, delivery.clubhead_velocity) / radius_sq
        )
        shaft_axis = orientation @ body.shaft_axis_body
        axial_omega = float(np.dot(club_omega, shaft_axis)) * shaft_axis
        club_omega = driven_omega + axial_omega
        club_velocity = -np.cross(club_omega, r_attachment)
    initial = ImpactIntervalInitialState(
        club_position_m=club_position,
        club_orientation=orientation,
        club_velocity_mps=club_velocity,
        club_angular_velocity_rad_s=club_omega,
        ball_position_m=BALL_POSITION_M.copy(),
        ball_velocity_mps=np.zeros(3),
        ball_angular_velocity_rad_s=np.zeros(3),
    )
    return solve_impact_interval(initial, body, interval_config)


def _launch_summary(
    post: PostImpactState, delivery: DeliveryDerived, flight: object
) -> dict[str, float]:
    """Flatten launch + flight numbers into the exportable summary dict."""
    launch = derive_launch_conditions(
        to_flight_frame(post.ball_velocity),
        to_flight_frame(post.ball_angular_velocity),
    )
    return {
        "ball_speed_mph": launch.ball_speed * MPH_PER_MPS,
        "launch_angle_deg": math.degrees(launch.launch_angle),
        # Flight-frame azimuth is + toward +y (left); app convention is
        # + right of target, so the sign flips.
        "launch_azimuth_deg": -math.degrees(launch.azimuth_angle),
        "spin_rpm": launch.spin_rate,
        "spin_axis_tilt_deg": delivery.spin_axis_tilt_deg,
        "carry_m": float(flight.carry_distance),  # type: ignore[attr-defined]
        "max_height_m": float(flight.max_height),  # type: ignore[attr-defined]
        "flight_time_s": float(flight.flight_time),  # type: ignore[attr-defined]
        "landing_angle_deg": float(flight.landing_angle),  # type: ignore[attr-defined]
    }


def run_simulation(config: SimulationConfig) -> SimulationRun:
    """Run one full swing -> impact -> flight simulation.

    Args:
        config: The simulation request.

    Returns:
        A complete, exportable :class:`SimulationRun`.
    """
    source = make_source(
        config.source_kind,
        config.scenario,
        plane=config.plane,
        duration=config.swing_duration_s,
    )
    n = int(round(source.duration / _SAMPLE_DT_S))
    times = np.linspace(0.0, source.duration, max(n, 2) + 1)

    tau = (
        min(max(config.impact_time_s, 0.0), source.duration)
        if config.impact_time_s is not None
        else _auto_impact_time(source, times)
    )

    # Scrubber math: translate the swing so the clubhead at tau meets
    # the fixed ball.
    impact_sample = source.sample(tau)
    offset = BALL_POSITION_M - impact_sample.pose[:3, 3]

    samples = [source.sample(float(t)) for t in times]
    poses = np.stack([s.pose for s in samples])
    poses[:, :3, 3] += offset
    positions = poses[:, :3, 3].copy()
    twists = np.stack([s.twist for s in samples])

    delivery = delivery_at(source, tau, config.scenario, config.club)
    interval_result: ImpactIntervalResult | None = None
    if config.impact_model == "impact_interval":
        interval_result = _solve_interval(config, delivery)
        post = interval_result.to_post_impact_state()
        post.impact_location = delivery.impact_offset.copy()
    else:
        solver = ImpactSolverAPI(
            ImpactModelType.RIGID_BODY,
            ImpactParameters(cg_depth=config.club.cg_depth_m),
        )
        post = solver.solve_with_gear_effect(
            timestamp=tau,
            clubhead_velocity=delivery.clubhead_velocity,
            clubhead_orientation=delivery.face_normal,
            impact_offset=delivery.impact_offset,
            clubhead_mass=config.club.head_mass_kg,
            clubhead_moi=config.club.moi_about_shaft_kg_m2,
            face_normal_at_offset=_face_normal_callable(config.club),
            record=False,
        )

    flight = flight_simulate(
        derive_launch_conditions(
            to_flight_frame(post.ball_velocity),
            to_flight_frame(post.ball_angular_velocity),
        ),
        model_name=config.flight_model,
    )
    flight_times = np.array([p.time for p in flight.trajectory])
    flight_positions = (
        from_flight_frame(flight.to_position_array()) + BALL_POSITION_M
        if len(flight.trajectory)
        else np.zeros((0, 3))
    )
    flight_velocities = (
        from_flight_frame(np.array([p.velocity for p in flight.trajectory]))
        if len(flight.trajectory)
        else np.zeros((0, 3))
    )

    run = SimulationRun(
        config=config,
        swing_times=times,
        swing_positions=positions,
        swing_poses=poses,
        swing_twists=twists,
        impact_time_s=tau,
        delivery=delivery,
        post_impact=post,
        impact_interval=interval_result,
        launch=_launch_summary(post, delivery, flight),
        flight_times=flight_times,
        flight_positions=flight_positions,
        flight_velocities=flight_velocities,
    )
    ensure(
        bool(
            np.allclose(
                source.sample(tau).pose[:3, 3] + offset, BALL_POSITION_M, atol=1e-9
            )
        ),
        "clubhead at tau must coincide with the ball position",
    )
    return run
