"""Adapt retained Rate runs to shared swept-wedge ground clearance."""

from __future__ import annotations

from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    GroundPlane,
    WedgeGroundClearanceAnalysis,
    WedgeHeadParameters,
    analyze_wedge_ground_clearance,
)


def ground_clearance_for_run(
    run: SimulationRun,
    parameters: WedgeHeadParameters,
    ground: GroundPlane,
) -> WedgeGroundClearanceAnalysis:
    """Analyze the complete retained sweep and preserve hit/miss semantics."""
    if not isinstance(run, SimulationRun):
        raise TypeError("run must be a SimulationRun")
    return analyze_wedge_ground_clearance(
        parameters,
        run.swing_times,
        run.swing_poses,
        run.swing_twists,
        ground,
        ball_contact_time_s=run.impact_time_s,
    )


__all__ = ["ground_clearance_for_run"]
