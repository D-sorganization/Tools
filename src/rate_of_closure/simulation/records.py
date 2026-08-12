"""Typed request and result records for Rate of Closure simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import ClubSpec, ClubType
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation.contact import ContactMode, ImpactOutcome
from rate_of_closure.simulation.manual_delivery import (
    ManualDeliveryConfig,
    ShaftAxisDatum,
)
from shared.python.swing_sim.ball_setup import (
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import (
    GOLF_BALL_RADIUS_M,
    DeliveryDerived,
    PostImpactState,
)
from shared.python.swing_sim.run_config import (
    DOUBLE_PENDULUM_JOINT_IDS,
    DoublePendulumRunConfig,
    SwingRunMode,
)
from shared.python.swing_sim.torque_library import TorqueProfileLibrary
from shared.python.swing_sim.types import PendulumParameters, PlaneOrientation

__all__ = ["BALL_POSITION_M", "SimulationConfig", "SimulationRun"]

BALL_POSITION_M = np.array([0.0, GOLF_BALL_RADIUS_M, 0.0])
"""Legacy ground-ball position; canonical simulations use ``ball_setup``."""

_AUTO_BALL_SETUP = BallSetup()
_MAX_DELIVERED_LOFT_DEG = 89.0


@dataclass(frozen=True)
class SimulationConfig:
    """One simulation request.

    ``delivery_inspection`` preserves legacy forced swing alignment.
    ``fixed_ball_contact`` retains the original swing and detects sampled
    point-to-sphere proximity.
    """

    scenario: ImpactScenario
    club: ClubSpec
    ball_setup: BallSetup = _AUTO_BALL_SETUP
    source_kind: str = "manual"
    plane: PlaneOrientation = field(default_factory=PlaneOrientation)
    impact_time_s: float | None = None
    impact_time_offset_s: float = 0.0
    flight_model: str = "waterloo_penner"
    swing_duration_s: float = 1.5
    contact_mode: ContactMode = ContactMode.DELIVERY_INSPECTION
    swing_run_config: DoublePendulumRunConfig = field(
        default_factory=DoublePendulumRunConfig
    )
    torque_library: TorqueProfileLibrary | None = None
    pendulum_parameters: PendulumParameters = field(
        default_factory=PendulumParameters.golf_default
    )
    manual_attack_angle_deg: float = 0.0
    manual_club_path_deg: float = 0.0
    manual_forward_shaft_lean_deg: float = 0.0
    manual_shaft_axis_datum: ShaftAxisDatum = ShaftAxisDatum.TRACKED_REFERENCE

    def __post_init__(self) -> None:
        """Validate and normalize the immutable request."""
        require(
            isinstance(self.scenario, ImpactScenario),
            "scenario must be an ImpactScenario",
            self.scenario,
        )
        require(isinstance(self.club, ClubSpec), "club must be a ClubSpec", self.club)
        resolved_setup = (
            _default_ball_setup(self.club)
            if self.ball_setup is _AUTO_BALL_SETUP
            else self.ball_setup
        )
        require(
            isinstance(resolved_setup, BallSetup),
            "ball_setup must be a BallSetup",
            resolved_setup,
        )
        object.__setattr__(self, "ball_setup", resolved_setup)
        object.__setattr__(self, "contact_mode", _contact_mode(self.contact_mode))
        manual_delivery = self.manual_delivery
        delivered_loft_deg = self.club.loft_deg - manual_delivery.forward_shaft_lean_deg
        if self.source_kind == "manual":
            require(
                abs(delivered_loft_deg) <= _MAX_DELIVERED_LOFT_DEG,
                "manual delivered dynamic loft must remain within +/-89 deg",
                delivered_loft_deg,
            )
        object.__setattr__(
            self, "manual_shaft_axis_datum", manual_delivery.shaft_axis_datum
        )
        require(
            isinstance(self.swing_run_config, DoublePendulumRunConfig),
            "swing_run_config must be a DoublePendulumRunConfig",
            self.swing_run_config,
        )
        require(
            isinstance(self.pendulum_parameters, PendulumParameters),
            "pendulum_parameters must be PendulumParameters",
            self.pendulum_parameters,
        )
        require(
            self.torque_library is None
            or isinstance(self.torque_library, TorqueProfileLibrary),
            "torque_library must be a TorqueProfileLibrary",
            self.torque_library,
        )
        if self.swing_run_config.mode is SwingRunMode.PRESCRIBED:
            require(
                self.source_kind == "double_pendulum",
                "prescribed torque currently requires the double-pendulum source",
                self.source_kind,
            )
            require(
                self.torque_library is not None,
                "prescribed torque requires a profile library",
            )
        if self.swing_run_config.joint_locks.has_locks:
            require(
                self.source_kind == "double_pendulum",
                "joint locks currently require the double-pendulum source",
                self.source_kind,
            )
        FlightModelType(self.flight_model)
        _validate_optional_impact_time(self.impact_time_s)
        require(
            math.isfinite(self.impact_time_offset_s),
            "impact_time_offset_s must be finite",
            self.impact_time_offset_s,
        )
        require(
            math.isfinite(self.swing_duration_s) and self.swing_duration_s > 0.0,
            "swing_duration_s must be finite and > 0",
            self.swing_duration_s,
        )

    @property
    def ball_position_m(self) -> np.ndarray:
        """Return a new ball-center vector for geometry calculations."""
        position: np.ndarray = np.asarray(self.ball_setup.ball_center_m, dtype=float)
        return position

    @property
    def manual_delivery(self) -> ManualDeliveryConfig:
        """Return the normalized manual-source declaration as one value object."""
        return ManualDeliveryConfig(
            attack_angle_deg=self.manual_attack_angle_deg,
            club_path_deg=self.manual_club_path_deg,
            forward_shaft_lean_deg=self.manual_forward_shaft_lean_deg,
            shaft_axis_datum=self.manual_shaft_axis_datum,
        )


def _default_ball_setup(club: ClubSpec) -> BallSetup:
    """Return the representative setup for a newly selected club."""
    if club.club_type is ClubType.DRIVER:
        return BallSetup(BallSupportMode.TEE, DEFAULT_DRIVER_TEE_HEIGHT_M)
    return BallSetup(BallSupportMode.GROUND, 0.0)


@dataclass(frozen=True)
class SimulationRun:
    """One complete swing and optional impact/flight record in the app frame."""

    config: SimulationConfig
    swing_times: np.ndarray
    swing_positions: np.ndarray
    swing_poses: np.ndarray
    swing_twists: np.ndarray
    swing_joints: np.ndarray
    swing_joint_ids: tuple[str, ...]
    swing_applied_torques_nm: np.ndarray
    impact_outcome: ImpactOutcome
    impact_time_s: float | None
    delivery: DeliveryDerived | None
    post_impact: PostImpactState | None
    launch: dict[str, float] | None
    flight_times: np.ndarray
    flight_positions: np.ndarray
    flight_velocities: np.ndarray

    def __post_init__(self) -> None:
        """Enforce coherent optional phases for hits and misses."""
        _validate_swing_shapes(self)
        _validate_flight_shapes(self)
        if self.impact_outcome.is_hit:
            require(self.impact_time_s is not None, "a hit requires impact_time_s")
            require(self.delivery is not None, "a hit requires delivery")
            require(self.post_impact is not None, "a hit requires post_impact")
            require(self.launch is not None, "a hit requires launch")
            return
        require(self.impact_time_s is None, "a miss cannot have impact_time_s")
        require(self.delivery is None, "a miss cannot have delivery")
        require(self.post_impact is None, "a miss cannot have post_impact")
        require(self.launch is None, "a miss cannot have launch")
        require(len(self.flight_times) == 0, "a miss cannot have a flight series")

    @property
    def total_duration_s(self) -> float:
        """Return the playback span, with no fabricated flight for a miss."""
        flight_span = float(self.flight_times[-1]) if len(self.flight_times) else 0.0
        return float(self.swing_times[-1]) + flight_span

    @property
    def inspection_time_s(self) -> float:
        """Return impact time, or the explicitly labeled closest approach for a miss."""
        if self.impact_time_s is not None:
            return float(self.impact_time_s)
        return float(self.impact_outcome.candidate_time_s)

    @property
    def inspection_event_label(self) -> str:
        """Return the honest event label paired with :attr:`inspection_time_s`."""
        return "Impact" if self.impact_outcome.is_hit else "Closest Approach"


def _contact_mode(value: ContactMode) -> ContactMode:
    """Normalize a contact mode while preserving useful validation errors."""
    try:
        return ContactMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown contact mode {value!r}") from error


def _validate_optional_impact_time(impact_time_s: float | None) -> None:
    """Validate a requested delivery-inspection time when present."""
    if impact_time_s is None:
        return
    require(
        math.isfinite(impact_time_s) and impact_time_s >= 0.0,
        "impact_time_s must be finite and >= 0",
        impact_time_s,
    )


def _validate_swing_shapes(run: SimulationRun) -> None:
    """Require complete, sample-aligned swing data for every outcome."""
    sample_count = len(run.swing_times)
    require(sample_count >= 3, "a run requires at least three swing samples")
    require(
        run.swing_positions.shape == (sample_count, 3),
        "swing_positions must have shape (N, 3)",
        run.swing_positions.shape,
    )
    require(
        run.swing_poses.shape == (sample_count, 4, 4),
        "swing_poses must have shape (N, 4, 4)",
        run.swing_poses.shape,
    )
    require(
        run.swing_twists.shape == (sample_count, 6),
        "swing_twists must have shape (N, 6)",
        run.swing_twists.shape,
    )
    require(
        run.swing_joints.ndim == 3
        and run.swing_joints.shape[0] == sample_count
        and run.swing_joints.shape[2] == 3,
        "swing_joints must have shape (N, J, 3)",
        run.swing_joints.shape,
    )
    _validate_torque_history(run, sample_count)


def _validate_torque_history(run: SimulationRun, sample_count: int) -> None:
    """Require source-compatible, finite, sample-aligned applied torques."""
    require(
        all(
            isinstance(joint_id, str) and bool(joint_id.strip())
            for joint_id in run.swing_joint_ids
        ),
        "swing_joint_ids must contain nonempty stable identifiers",
        run.swing_joint_ids,
    )
    require(
        len(set(run.swing_joint_ids)) == len(run.swing_joint_ids),
        "swing_joint_ids must be unique",
        run.swing_joint_ids,
    )
    require(
        run.swing_applied_torques_nm.shape == (sample_count, len(run.swing_joint_ids)),
        "swing_applied_torques_nm must have shape (N, len(swing_joint_ids))",
        run.swing_applied_torques_nm.shape,
    )
    require(
        bool(np.all(np.isfinite(run.swing_applied_torques_nm))),
        "swing_applied_torques_nm must be finite",
    )
    expected_joint_ids = (
        DOUBLE_PENDULUM_JOINT_IDS if run.config.source_kind == "double_pendulum" else ()
    )
    require(
        run.swing_joint_ids == expected_joint_ids,
        "applied torque joint IDs are incompatible with the swing source",
        run.swing_joint_ids,
    )


def _validate_flight_shapes(run: SimulationRun) -> None:
    """Require aligned, three-dimensional flight arrays when present."""
    sample_count = len(run.flight_times)
    require(run.flight_times.ndim == 1, "flight_times must be one-dimensional")
    require(
        run.flight_positions.shape == (sample_count, 3),
        "flight_positions must have shape (M, 3)",
        run.flight_positions.shape,
    )
    require(
        run.flight_velocities.shape == (sample_count, 3),
        "flight_velocities must have shape (M, 3)",
        run.flight_velocities.shape,
    )
