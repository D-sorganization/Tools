"""Adapt retained Rate runs to the shared frame-explicit wedge analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.club import (
    face_center_point,
    face_normal_at_offset,
    hosel_point,
)
from rate_of_closure.model import impact_frame, impact_lever_m
from rate_of_closure.simulation.manual_delivery import ShaftAxisDatum
from rate_of_closure.simulation.records import SimulationRun
from shared.python.golf_club import (
    WedgeKinematicAnalysis,
    WedgeKinematicState,
    analyze_wedge_kinematics,
)
from shared.python.golf_club._wedge_sweep import (
    interpolated_pose,
    interpolated_twist,
)
from shared.python.swing_sim.impact import DPlaneAnalysis, analyze_dplane

__all__ = ["ImpactKinematicSnapshot", "impact_kinematics_for_run"]

_APP_FRAME_ID = "app_frame:x_target,y_up,z_right"
_GROUND_UP = np.array([0.0, 1.0, 0.0])
_LOCAL_LEADING_EDGE = np.array([0.0, 0.0, 1.0])
_MIN_DIRECTION_NORM = 1e-12
_REFERENCE_IMPACT_LIMITATION = (
    " The downstream rigid impact and ball-flight pipeline currently consumes "
    "reference-point translation; shaft-induced contact-point velocity is an "
    "analysis result and does not alter the simulated flight."
)


@dataclass(frozen=True)
class ImpactKinematicSnapshot:
    """One auditable impact or closest-approach kinematic inspection."""

    event_time_s: float
    event_label: str
    geometry_basis: str
    model_limitations: str
    sample_index: int
    face_center_point_m: tuple[float, float, float]
    face_center_velocity_mps: tuple[float, float, float]
    face_center_normal_unit: tuple[float, float, float]
    reference_dplane: DPlaneAnalysis
    face_center_dplane: DPlaneAnalysis
    contact_dplane: DPlaneAnalysis
    state: WedgeKinematicState
    analysis: WedgeKinematicAnalysis


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    magnitude = float(np.linalg.norm(vector))
    require(magnitude > _MIN_DIRECTION_NORM, f"{name} must have nonzero length")
    normalized: np.ndarray = vector / magnitude
    return normalized


def _xyz(vector: np.ndarray) -> tuple[float, float, float]:
    """Convert a validated 3-vector to the shared immutable wire type."""
    require(vector.shape == (3,), "vector must contain exactly three components")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


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


def _event_path_tangent(
    run: SimulationRun, time_s: float, fallback_index: int
) -> np.ndarray:
    """Return the interpolated event velocity direction with a sampled fallback."""
    velocity = interpolated_twist(run.swing_times, run.swing_twists, time_s)[3:]
    if float(np.linalg.norm(velocity)) > _MIN_DIRECTION_NORM:
        return _unit(velocity, "interpolated reference velocity")
    return _path_tangent(run, fallback_index)


def _arc_tangent_rate(
    run: SimulationRun, index: int, event_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sampled unit path tangent and its stable finite difference."""
    if index == 0:
        first, second = 0, 1
    elif index == len(run.swing_times) - 1:
        first, second = index - 1, index
    else:
        first, second = index - 1, index + 1
    dt = float(run.swing_times[second] - run.swing_times[first])
    require(dt > 0.0, "swing sample times must increase")
    tangent = _event_path_tangent(run, event_time_s, index)
    raw_rate = (_path_tangent(run, second) - _path_tangent(run, first)) / dt
    rate = raw_rate - float(np.dot(raw_rate, tangent)) * tangent
    return tangent, rate


def _shaft_geometry(
    run: SimulationRun,
    event_time_s: float,
    rotation: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if run.swing_joints.shape[1] >= 2:
        wrist = interpolated_twist(
            run.swing_times,
            run.swing_joints[:, -2, :],
            event_time_s,
        )
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
    shaft_axis = _unit(rotation @ local_shaft, "scenario shaft axis")
    if (
        run.config.source_kind == "manual"
        and run.config.manual_shaft_axis_datum is ShaftAxisDatum.GENERATED_HOSEL
    ):
        local_hosel = np.asarray(hosel_point(run.config.club), dtype=float)
        local_face_center = np.asarray(face_center_point(run.config.club), dtype=float)
        registered_hosel = local_hosel - local_face_center
        registered_hosel[0] += run.config.scenario.com_to_face_mm / 1000.0
        return (
            reference + rotation @ registered_hosel,
            shaft_axis,
            "generated_head_profile_hosel",
            "The shaft datum uses the selected club's representative generated "
            "head-profile hosel and local lie direction. It is not a measured "
            "manufacturer CAD datum or a fitted shaft centerline."
            + _REFERENCE_IMPACT_LIMITATION,
        )
    return (
        reference,
        shaft_axis,
        "scenario_shaft_line",
        "The physical shaft axis is assumed to pass through the tracked head "
        "reference point, matching the current prescribed manual twist model."
        + _REFERENCE_IMPACT_LIMITATION,
    )


def impact_kinematics_for_run(run: SimulationRun) -> ImpactKinematicSnapshot:
    """Analyze the exact inspection event without fabricating contact on a miss."""
    if not isinstance(run, SimulationRun):
        raise TypeError("run must be a SimulationRun")
    index = _event_index(run)
    event_time_s = run.inspection_time_s
    pose = interpolated_pose(run.swing_times, run.swing_poses, event_time_s)
    rotation = pose[:3, :3]
    reference = pose[:3, 3]
    twist = interpolated_twist(run.swing_times, run.swing_twists, event_time_s)
    shaft_point, shaft_axis, basis, limitations = _shaft_geometry(
        run, event_time_s, rotation, reference
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
    arc_tangent, arc_rate = _arc_tangent_rate(run, index, event_time_s)
    face_center_lever = rotation @ np.array(
        [run.config.scenario.com_to_face_mm / 1000.0, 0.0, 0.0]
    )
    face_center_point = reference + face_center_lever
    face_center_velocity = twist[3:] + np.cross(twist[:3], face_center_lever)
    face_center_normal = rotation @ np.asarray(
        face_normal_at_offset(run.config.club, 0.0, 0.0)
    )
    state = WedgeKinematicState(
        frame_id=_APP_FRAME_ID,
        reference_position_m=_xyz(reference),
        reference_velocity_mps=_xyz(twist[3:]),
        angular_velocity_rad_s=_xyz(twist[:3]),
        shaft_axis_point_m=_xyz(shaft_point),
        shaft_axis_unit=_xyz(shaft_axis),
        face_center_point_m=_xyz(face_center_point),
        contact_point_m=_xyz(reference + lever),
        face_normal_unit=_xyz(face_normal),
        leading_edge_tangent_unit=_xyz(leading_edge),
        ground_up_unit=_xyz(_GROUND_UP),
        arc_tangent_unit=_xyz(arc_tangent),
        arc_tangent_rate_per_s=_xyz(arc_rate),
    )
    analysis = analyze_wedge_kinematics(state)
    return ImpactKinematicSnapshot(
        event_time_s=event_time_s,
        event_label=run.inspection_event_label,
        geometry_basis=basis,
        model_limitations=limitations,
        sample_index=index,
        face_center_point_m=_xyz(face_center_point),
        face_center_velocity_mps=_xyz(face_center_velocity),
        face_center_normal_unit=_xyz(face_center_normal),
        reference_dplane=analyze_dplane(twist[3:], face_center_normal),
        face_center_dplane=analyze_dplane(face_center_velocity, face_center_normal),
        contact_dplane=analyze_dplane(analysis.contact_velocity_mps, face_normal),
        state=state,
        analysis=analysis,
    )
