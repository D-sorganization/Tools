"""Adapt retained Rate wedge states to the shared compliant-turf model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rate_of_closure.simulation.ground_clearance import (
    RunGroundClearanceSnapshot,
    _registered_wedge_sweep,
    ground_clearance_for_run,
)
from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    GroundPlane,
    ReducedTurfContactResult,
    TurfContactProfile,
    WedgeHeadParameters,
    WedgeTurfWrench,
    evaluate_wedge_turf_wrench,
    simulate_reduced_turf_contact,
)

_LIMITATIONS = (
    "The instantaneous wrench consumes the retained dynamics pose/twist and is "
    "ready for force-coupled solvers. This adapter does not replay the retained "
    "swing under turf force; the effective-mass result is a diagnostic, not a "
    "full rigid-body post-contact trajectory."
)


@dataclass(frozen=True)
class RunTurfInteractionSnapshot:
    """Geometry event plus optional first-contact turf diagnostics."""

    ground_clearance: RunGroundClearanceSnapshot
    profile: TurfContactProfile
    first_contact_wrench: WedgeTurfWrench | None
    reduced_contact: ReducedTurfContactResult | None
    limitations: str = _LIMITATIONS


def _interpolated_twist(
    times_s: np.ndarray, twists: np.ndarray, time_s: float
) -> np.ndarray:
    values: np.ndarray = np.array(
        [np.interp(time_s, times_s, twists[:, column]) for column in range(6)],
        dtype=float,
    )
    return values


def turf_interaction_for_run(
    run: SimulationRun,
    parameters: WedgeHeadParameters,
    ground: GroundPlane,
    profile: TurfContactProfile,
    *,
    time_step_s: float = 5e-6,
) -> RunTurfInteractionSnapshot:
    """Evaluate the shared contact law at a retained run's first ground event."""
    if not isinstance(run, SimulationRun):
        raise TypeError("run must be a SimulationRun")
    if not isinstance(parameters, WedgeHeadParameters):
        raise TypeError("parameters must be WedgeHeadParameters")
    if not isinstance(ground, GroundPlane):
        raise TypeError("ground must be GroundPlane")
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    clearance = ground_clearance_for_run(run, parameters, ground)
    event = clearance.analysis.first_ground_contact
    if event is None:
        return RunTurfInteractionSnapshot(clearance, profile, None, None)
    _, twists = _registered_wedge_sweep(run, parameters)
    twist = _interpolated_twist(run.swing_times, twists, event.time_s)
    pose = np.asarray(event.pose_head_to_ground)
    wrench = evaluate_wedge_turf_wrench(parameters, profile, pose, twist, ground)
    normal = np.asarray(ground.normal_unit)
    contact_velocity = np.asarray(event.tangential_velocity_mps) + (
        event.normal_velocity_mps * normal
    )
    reduced = simulate_reduced_turf_contact(
        profile,
        initial_contact_velocity_mps=contact_velocity,
        surface_normal_unit=ground.normal_unit,
        effective_mass_kg=run.config.club.head_mass_kg,
        time_step_s=time_step_s,
    )
    return RunTurfInteractionSnapshot(clearance, profile, wrench, reduced)


__all__ = ["RunTurfInteractionSnapshot", "turf_interaction_for_run"]
