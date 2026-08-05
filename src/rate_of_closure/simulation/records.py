"""Typed request and result records for Rate of Closure simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import ClubSpec
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation.contact import ContactMode, ImpactOutcome
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import (
    GOLF_BALL_RADIUS_M,
    DeliveryDerived,
    PostImpactState,
)
from shared.python.swing_sim.types import PlaneOrientation

__all__ = ["BALL_POSITION_M", "SimulationConfig", "SimulationRun"]

BALL_POSITION_M = np.array([0.0, GOLF_BALL_RADIUS_M, 0.0])


@dataclass(frozen=True)
class SimulationConfig:
    """One simulation request.

    ``delivery_inspection`` preserves legacy forced swing alignment.
    ``fixed_ball_contact`` retains the original swing and detects sampled
    point-to-sphere proximity.
    """

    scenario: ImpactScenario
    club: ClubSpec
    source_kind: str = "manual"
    plane: PlaneOrientation = field(default_factory=PlaneOrientation)
    impact_time_s: float | None = None
    flight_model: str = "waterloo_penner"
    swing_duration_s: float = 1.5
    contact_mode: ContactMode = ContactMode.DELIVERY_INSPECTION

    def __post_init__(self) -> None:
        """Validate and normalize the immutable request."""
        require(
            isinstance(self.scenario, ImpactScenario),
            "scenario must be an ImpactScenario",
            self.scenario,
        )
        require(isinstance(self.club, ClubSpec), "club must be a ClubSpec", self.club)
        object.__setattr__(self, "contact_mode", _contact_mode(self.contact_mode))
        FlightModelType(self.flight_model)
        _validate_optional_impact_time(self.impact_time_s)
        require(
            math.isfinite(self.swing_duration_s) and self.swing_duration_s > 0.0,
            "swing_duration_s must be finite and > 0",
            self.swing_duration_s,
        )


@dataclass(frozen=True)
class SimulationRun:
    """One complete swing and optional impact/flight record in the app frame."""

    config: SimulationConfig
    swing_times: np.ndarray
    swing_positions: np.ndarray
    swing_poses: np.ndarray
    swing_twists: np.ndarray
    swing_joints: np.ndarray
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
