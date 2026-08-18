"""Adapt retained Rate runs to shared swept-wedge ground clearance."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rate_of_closure.club import ClubSpec, ClubType
from rate_of_closure.model import impact_lever_m
from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    GroundPlane,
    WedgeGroundClearanceAnalysis,
    WedgeHeadParameters,
    WedgePreset,
    analyze_wedge_ground_clearance,
    wedge_face_contact_point_m,
    wedge_preset,
)


@dataclass(frozen=True)
class RunGroundClearanceSnapshot:
    """One registered Rate sweep and its explicit geometric limitations."""

    geometry_basis: str
    model_limitations: str
    parameters: WedgeHeadParameters
    analysis: WedgeGroundClearanceAnalysis


def representative_wedge_parameters_for_club(
    club: ClubSpec,
) -> WedgeHeadParameters | None:
    """Return an explicit illustrative mid-bounce head for a Rate wedge."""
    if not isinstance(club, ClubSpec):
        raise TypeError("club must be a ClubSpec")
    if club.club_type is not ClubType.WEDGE:
        return None
    baseline = wedge_preset(WedgePreset.MID_BOUNCE)
    provenance = replace(
        baseline.provenance,
        source_name="Rate selected-wedge static datums",
        geometry_basis=(
            "Selected Rate loft, lie, and mass with the generic mid-bounce "
            "canonical wedge geometry"
        ),
        uncertainty_note=(
            "Illustrative 10-degree mid-bounce sole; not a measured or "
            "manufacturer-specific grind for the selected club."
        ),
    )
    return replace(
        baseline,
        head_id="rate-" + club.name.lower().replace(" ", "-"),
        loft_deg=club.loft_deg,
        lie_deg=club.lie_deg,
        target_mass_kg=club.head_mass_kg,
        provenance=provenance,
    )


def _registered_wedge_sweep(
    run: SimulationRun, parameters: WedgeHeadParameters
) -> tuple[np.ndarray, np.ndarray]:
    """Move the retained twist from Rate's reference to the wedge head datum."""
    scenario = run.config.scenario
    face_contact = np.asarray(
        wedge_face_contact_point_m(
            parameters,
            scenario.impact_offset_toe_mm * 1.0e-3,
            scenario.impact_offset_high_mm * 1.0e-3,
        )
    )
    local_shift = impact_lever_m(scenario) - face_contact
    poses = np.array(run.swing_poses, dtype=float, copy=True)
    twists = np.array(run.swing_twists, dtype=float, copy=True)
    world_shifts = np.einsum("nij,j->ni", poses[:, :3, :3], local_shift)
    poses[:, :3, 3] += world_shifts
    twists[:, 3:] += np.cross(twists[:, :3], world_shifts)
    return poses, twists


def ground_clearance_for_run(
    run: SimulationRun,
    parameters: WedgeHeadParameters,
    ground: GroundPlane,
) -> RunGroundClearanceSnapshot:
    """Analyze the complete retained sweep and preserve hit/miss semantics."""
    if not isinstance(run, SimulationRun):
        raise TypeError("run must be a SimulationRun")
    poses, twists = _registered_wedge_sweep(run, parameters)
    analysis = analyze_wedge_ground_clearance(
        parameters,
        run.swing_times,
        poses,
        twists,
        ground,
        ball_contact_time_s=run.impact_time_s,
    )
    return RunGroundClearanceSnapshot(
        geometry_basis="canonical_wedge_face_contact_registration",
        model_limitations=(
            "The canonical wedge face point is rigidly registered to the Rate "
            "scenario impact lever. Ball timing inherits the "
            f"{run.impact_outcome.geometry_model} contact policy: "
            f"{run.impact_outcome.geometry_limitations} Ground results remain "
            "rigid planar geometry with no turf mechanics."
        ),
        parameters=parameters,
        analysis=analysis,
    )


__all__ = [
    "RunGroundClearanceSnapshot",
    "ground_clearance_for_run",
    "representative_wedge_parameters_for_club",
]
