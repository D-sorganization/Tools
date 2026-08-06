"""Adapt retained Rate runs to the shared frame-explicit wedge analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import face_normal_at_offset
from rate_of_closure.model import impact_frame, impact_lever_m
from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    WedgeKinematicAnalysis,
    WedgeKinematicState,
    analyze_wedge_kinematics,
)

__all__ = ["ImpactKinematicSnapshot", "impact_kinematics_for_run"]

_APP_FRAME_ID = "app_frame:x_target,y_up,z_right"
_GROUND_UP = np.array([0.0, 1.0, 0.0])
_LOCAL_LEADING_EDGE = np.array([0.0, 0.0, 1.0])
_MIN_DIRECTION_NORM = 1e-12


@dataclass(frozen=True)
class ImpactKinematicSnapshot:
    """One auditable impact or closest-approach kinematic inspection."""

    event_time_s: float
    event_label: str
    geometry_basis: str
    model_limitations: str
    sample_index: int
    state: WedgeKinematicState
    analysis: WedgeKinematicAnalysis


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    require(magnitude > _MIN_DIRECTION_NORM, f"{name} must have nonzero length")
    normalized: np.ndarray = vector / magnitude
    return normalized


def _event_index(run: SimulationRun) -> int:
    return int(np.argmin(np.abs(run.swing_times - run.inspection_time_s)))


def _path_tangent(run: SimulationRun, index: int) -> np.ndarray:
    velocity = run.swing_twists[index, 3:]
    if float(np.linalg.norm(velocity)) > _MIN_DIRECTION_NORM:
        return _unit(velocity, "reference velocity")
    if index == 0:
        first, second = 0, 1
    elif index == len(run.swing_times) - 1:
        first, second = index - 1, index
    else:
        first, second = index - 1, index + 1
    displacement = run.swing_positions[second] - run.swing_positions[first]
    return _unit(displacement, "sampled path displacement")


def _arc_tangent_rate(run: SimulationRun, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampled unit path tangent and its stable finite difference."""
    if index == 0:
        first, second = 0, 1
    elif index == len(run.swing_times) - 1:
        first, second = index - 1, index
    else:
        first, second = index - 1, index + 1
    dt = float(run.swing_times[second] - run.swing_times[first])
    require(dt > 0.0, "swing sample times must increase")
    tangent = _path_tangent(run, index)
    raw_rate = (_path_tangent(run, second) - _path_tangent(run, first)) / dt
    rate = raw_rate - float(np.dot(raw_rate, tangent)) * tangent
    return tangent, rate


def _shaft_geometry(
    run: SimulationRun, index: int, rotation: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if run.swing_joints.shape[1] >= 2:
        wrist = run.swing_joints[index, -2]
        shaft_axis = _unit(wrist - reference, "articulated shaft line")
        return (
            wrist,
            shaft_axis,
            "articulated_wrist_to_reference_shaft_line",
            "The articulated source supplies the measured wrist-to-reference shaft "
            "line but currently has no shaft-twist degree of freedom; nonzero shaft "
            "rotation attribution requires a future torsional head state.",
        )
    local_shaft, _plane_normal = impact_frame(run.config.scenario.lie_angle_deg)
    return (
        reference,
        _unit(rotation @ local_shaft, "scenario shaft axis"),
        "scenario_shaft_line",
        "The physical shaft axis is assumed to pass through the tracked head "
        "reference point, matching the current prescribed manual twist model.",
    )


def impact_kinematics_for_run(run: SimulationRun) -> ImpactKinematicSnapshot:
    """Analyze the retained inspection sample without fabricating contact on a miss."""
    if not isinstance(run, SimulationRun):
        raise TypeError("run must be a SimulationRun")
    index = _event_index(run)
    pose = run.swing_poses[index]
    rotation = pose[:3, :3]
    reference = pose[:3, 3]
    twist = run.swing_twists[index]
    shaft_point, shaft_axis, basis, limitations = _shaft_geometry(
        run, index, rotation, reference
    )
    lever = rotation @ impact_lever_m(run.config.scenario)
    face_normal = rotation @ np.asarray(
        face_normal_at_offset(
            run.config.club,
            run.config.scenario.impact_offset_toe_mm,
            run.config.scenario.impact_offset_high_mm,
        )
    )
    nominal_edge = rotation @ _LOCAL_LEADING_EDGE
    leading_edge = _unit(
        nominal_edge - float(np.dot(nominal_edge, face_normal)) * face_normal,
        "leading-edge face tangent",
    )
    arc_tangent, arc_rate = _arc_tangent_rate(run, index)
    state = WedgeKinematicState(
        frame_id=_APP_FRAME_ID,
        reference_position_m=reference,
        reference_velocity_mps=twist[3:],
        angular_velocity_rad_s=twist[:3],
        shaft_axis_point_m=shaft_point,
        shaft_axis_unit=shaft_axis,
        contact_point_m=reference + lever,
        face_normal_unit=face_normal,
        leading_edge_tangent_unit=leading_edge,
        ground_up_unit=_GROUND_UP,
        arc_tangent_unit=arc_tangent,
        arc_tangent_rate_per_s=arc_rate,
    )
    return ImpactKinematicSnapshot(
        event_time_s=run.inspection_time_s,
        event_label=run.inspection_event_label,
        geometry_basis=basis,
        model_limitations=limitations,
        sample_index=index,
        state=state,
        analysis=analyze_wedge_kinematics(state),
    )
