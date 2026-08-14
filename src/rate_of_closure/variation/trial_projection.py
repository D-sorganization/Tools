"""Shared capture and scalar projection for complete Rate simulation trials."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    EVALUATED_HIT,
    EVALUATED_NO_IMPACT,
    NUMERICAL_FAILURE,
    SimulationTrialOutcome,
)
from shared.python.contracts import ContractViolationError, require

logger = logging.getLogger(__name__)

SimulationExecutor = Callable[[SimulationConfig], SimulationRun]
TRIAL_NUMERICAL_FAILURES = (
    ValueError,
    ContractViolationError,
    RuntimeError,
    FloatingPointError,
    OverflowError,
)


@dataclass(frozen=True)
class TrialCapture:
    """One completed simulation run or an explicitly caught numerical failure."""

    run: SimulationRun | None
    error: Exception | None

    def __post_init__(self) -> None:
        require(
            (self.run is None) != (self.error is None),
            "capture must contain exactly one run or error",
        )
        require(
            self.run is None or isinstance(self.run, SimulationRun),
            "capture run must be a SimulationRun",
        )
        require(
            self.error is None or isinstance(self.error, Exception),
            "capture error must be an Exception",
        )


def capture_simulation(
    config: SimulationConfig,
    executor: SimulationExecutor = run_simulation,
) -> TrialCapture:
    """Execute one config, catching only the canonical numerical failure tuple."""
    require(isinstance(config, SimulationConfig), "config must be SimulationConfig")
    require(callable(executor), "executor must be callable")
    try:
        run = executor(config)
    except TRIAL_NUMERICAL_FAILURES as error:
        logger.debug("simulation trial failed: %s", error)
        return TrialCapture(None, error)
    require(isinstance(run, SimulationRun), "executor must return SimulationRun")
    return TrialCapture(run, None)


def project_simulation_outcome(
    index: int, capture: TrialCapture
) -> SimulationTrialOutcome:
    """Project one capture to the canonical 17-scalar typed outcome."""
    require(
        isinstance(index, int) and not isinstance(index, bool) and index >= 0,
        "index must be a nonnegative integer",
        index,
    )
    require(isinstance(capture, TrialCapture), "capture must be a TrialCapture")
    if capture.run is None:
        assert capture.error is not None
        return SimulationTrialOutcome(
            index,
            NUMERICAL_FAILURE,
            _empty_values(),
            type(capture.error).__name__,
            str(capture.error),
        )
    run = capture.run
    status = EVALUATED_HIT if run.impact_outcome.is_hit else EVALUATED_NO_IMPACT
    values = _contact_values(run)
    if status is EVALUATED_HIT:
        values.update(_impact_values(run))
        values.update(_shot_values(run))
    return SimulationTrialOutcome(index, status, values)


def _empty_values() -> dict[str, float | None]:
    return dict.fromkeys(ALL_OUTPUT_NAMES)


def _contact_values(run: SimulationRun) -> dict[str, float | None]:
    values = _empty_values()
    outcome = run.impact_outcome
    values.update(
        candidate_time_s=outcome.candidate_time_s,
        closest_approach_m=outcome.closest_approach_m,
        contact_margin_m=outcome.contact_margin_m,
    )
    return values


def _impact_values(run: SimulationRun) -> dict[str, float]:
    delivery = run.delivery
    assert run.impact_time_s is not None and delivery is not None
    return {
        "impact_time_s": run.impact_time_s,
        "clubhead_speed_mps": float(np.linalg.norm(delivery.clubhead_velocity)),
        "spin_loft_deg": delivery.spin_loft_deg,
        "face_to_path_deg": delivery.face_to_path_deg,
        "spin_axis_tilt_deg": delivery.spin_axis_tilt_deg,
    }


def _shot_values(run: SimulationRun) -> dict[str, float]:
    launch = run.launch
    assert launch is not None and len(run.flight_positions) > 0
    return {
        "ball_speed_mph": launch["ball_speed_mph"],
        "launch_angle_deg": launch["launch_angle_deg"],
        "launch_azimuth_deg": launch["launch_azimuth_deg"],
        "spin_rpm": launch["spin_rpm"],
        "carry_m": launch["carry_m"],
        "lateral_m": float(run.flight_positions[-1, 2]),
        "max_height_m": launch["max_height_m"],
        "flight_time_s": launch["flight_time_s"],
        "landing_angle_deg": launch["landing_angle_deg"],
    }


__all__ = [
    "SimulationExecutor",
    "TRIAL_NUMERICAL_FAILURES",
    "TrialCapture",
    "capture_simulation",
    "project_simulation_outcome",
]
