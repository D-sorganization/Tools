"""Shared text presentation for frame-explicit wedge impact kinematics."""

from __future__ import annotations

import math
from html import escape

from rate_of_closure.simulation import (
    ImpactKinematicSnapshot,
    RunGroundClearanceSnapshot,
    SimulationRun,
    ground_clearance_for_run,
    impact_kinematics_for_run,
    representative_wedge_parameters_for_club,
)
from shared.python.golf_club import GroundPlane
from shared.python.golf_club._wedge_delivery_metrics import (
    WedgeDeliveryMetrics,
    compute_wedge_delivery_metrics,
)
from shared.python.golf_club._wedge_sweep import (
    interpolated_pose,
    interpolated_twist,
)

__all__ = [
    "format_impact_kinematics",
    "format_simulation_engineering_readout",
    "format_simulation_key_metrics",
    "format_wedge_delivery_metrics",
    "ground_clearance_snapshot_for_scene",
    "simulation_ground_clearance_snapshot",
    "simulation_wedge_delivery_metrics",
]


def _number(value: float | None, unit: str, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "Unavailable"
    return f"{value:.{decimals}f} {unit}"


def _degrees_per_second(value_rad_s: float | None) -> str:
    return _number(None if value_rad_s is None else math.degrees(value_rad_s), "°/s", 1)


def format_impact_kinematics(snapshot: ImpactKinematicSnapshot) -> str:
    """Return a compact, provenance-bearing engineering readout."""
    analysis = snapshot.analysis
    shaft_vertical = analysis.shaft_rotation_velocity_mps[1]
    screw_distance = (
        None if analysis.screw_axis is None else analysis.screw_axis.contact_distance_m
    )
    metrics = (
        ("Contact-Point AoA", _number(analysis.total_aoa_deg, "°")),
        ("Without Shaft Rotation", _number(analysis.without_shaft_aoa_deg, "°")),
        (
            "Shaft AoA Contribution",
            _number(analysis.shaft_counterfactual_aoa_delta_deg, "°"),
        ),
        (
            "Shaft Rotation Rate",
            _degrees_per_second(analysis.shaft_rotation_rate_rad_s),
        ),
        ("Shaft-Induced Vertical Velocity", _number(shaft_vertical, "m/s", 3)),
        (
            "Face-Normal 3D Rate",
            _degrees_per_second(analysis.face_normal_3d_rate_rad_s),
        ),
        (
            "Leading Edge vs Arc Rate",
            _degrees_per_second(analysis.leading_edge_relative_arc_heading_rate_rad_s),
        ),
        ("Contact-to-Screw-Axis Distance", _number(screw_distance, "m", 4)),
        (
            "Face-Center Spin Loft (3D)",
            _number(snapshot.face_center_dplane.spin_loft_3d_deg, "°"),
        ),
        (
            "Planar Spin-Loft Approximation",
            _number(snapshot.face_center_dplane.planar_spin_loft_deg, "°"),
        ),
        (
            "3D Minus Planar Residual",
            _number(snapshot.face_center_dplane.spin_loft_residual_deg, "°"),
        ),
        (
            "D-Plane Normal Tilt",
            _number(snapshot.face_center_dplane.dplane_tilt_deg, "°"),
        ),
    )
    metric_html = " • ".join(f"<b>{label}:</b> {value}" for label, value in metrics)
    return (
        f"<b>{snapshot.event_label} Kinematics</b> at "
        f"{snapshot.event_time_s:.3f} s — {metric_html}<br>"
        f"<b>Geometry Basis:</b> {snapshot.geometry_basis}. "
        "<b>D-Plane Basis:</b> exact rigid-body face-center travel including "
        "ω × r versus the face-center normal; positive normal tilt is face-right "
        "and fade-side only under the current right-handed display convention. "
        f"<b>Model Boundary:</b> {snapshot.model_limitations}"
    )


def _format_ground_clearance(snapshot: RunGroundClearanceSnapshot) -> str:
    analysis = snapshot.analysis
    first_contact = analysis.first_ground_contact
    first_contact_text = (
        "No Ground Contact"
        if first_contact is None
        else (
            first_contact.feature.value.replace("_", " ").title()
            + f" at {first_contact.time_s:.4f} s"
        )
    )
    metrics = (
        (
            "Leading-Edge Clearance at Ball",
            _number(analysis.leading_edge_clearance_at_ball_m, "m", 4),
        ),
        ("Sole-Entry Margin", _number(analysis.sole_entry_margin_m, "m", 4)),
        (
            "Ground After Ball Time Margin",
            _number(analysis.ground_after_ball_time_margin_s, "s", 4),
        ),
        (
            "Delivered Bounce",
            _number(analysis.delivered_bounce_deg_at_ball, "°"),
        ),
        (
            "Path-Projected Effective Bounce",
            _number(
                analysis.path_projected_effective_bounce_deg_at_ball,
                "°",
            ),
        ),
        (
            "Bounce-Utilization Angle Margin",
            _number(analysis.bounce_utilization_margin_deg, "°"),
        ),
        ("First Ground Contact", first_contact_text),
    )
    metric_html = " • ".join(f"<b>{label}:</b> {value}" for label, value in metrics)
    sequence = analysis.sequence.value.replace("_", " ").title()
    uncertainty = escape(snapshot.parameters.provenance.uncertainty_note)
    limitations = escape(snapshot.model_limitations)
    return (
        f"<br><b>Wedge Ground-Clearance Sequence:</b> {sequence} — {metric_html}<br>"
        f"<b>Wedge Geometry:</b> {uncertainty} "
        f"<b>Ground-Clearance Boundary:</b> {limitations}"
    )


def simulation_ground_clearance_snapshot(
    run: SimulationRun,
) -> RunGroundClearanceSnapshot | None:
    """Return the shared wedge snapshot used by text and scene presentation."""
    parameters = representative_wedge_parameters_for_club(run.config.club)
    if parameters is None:
        return None
    return ground_clearance_for_run(
        run,
        parameters,
        GroundPlane(frame_id="app_frame:x_target,y_up,z_right"),
    )


def ground_clearance_snapshot_for_scene(
    run: SimulationRun | None,
) -> RunGroundClearanceSnapshot | None:
    """Resolve optional scene geometry without letting invalid input break drawing."""
    if run is None:
        return None
    try:
        return simulation_ground_clearance_snapshot(run)
    except ValueError:
        return None


def simulation_wedge_delivery_metrics(
    run: SimulationRun,
) -> WedgeDeliveryMetrics | None:
    """Compute synchronized wedge delivery metrics and waterfall for a run."""
    parameters = representative_wedge_parameters_for_club(run.config.club)
    if parameters is None:
        return None
    event_time_s = run.inspection_time_s
    pose = interpolated_pose(run.swing_times, run.swing_poses, event_time_s)
    twist = interpolated_twist(run.swing_times, run.swing_twists, event_time_s)
    ground_snapshot = simulation_ground_clearance_snapshot(run)
    low_point = ground_snapshot.analysis.low_point_world_m if ground_snapshot else None
    return compute_wedge_delivery_metrics(
        parameters=parameters,
        pose=pose,
        twist=twist,
        low_point_world_m=low_point,
    )


def format_wedge_delivery_metrics(metrics: WedgeDeliveryMetrics) -> str:
    """Format delivery cards, LE rates, and linear velocity waterfall."""
    wf = metrics.waterfall
    le_items = (
        ("LE Downrange Rate", _number(metrics.le_downrange_rate_mps, "m/s")),
        ("LE Vertical Rate", _number(metrics.le_vertical_rate_mps, "m/s")),
        ("LE Lateral Rate", _number(metrics.le_lateral_rate_mps, "m/s")),
        ("LE Total Speed", _number(metrics.le_total_speed_mps, "m/s")),
        ("LE 3D Rate", _number(metrics.le_3d_angular_rate_dps, "°/s", 1)),
        ("Dynamic Loft", _number(metrics.dynamic_loft_deg, "°")),
        ("Dynamic Lie", _number(metrics.dynamic_lie_deg, "°")),
        ("Dynamic Face Angle", _number(metrics.dynamic_face_angle_deg, "°")),
        ("Delivered Bounce", _number(metrics.delivered_bounce_deg, "°")),
    )
    le_html = " • ".join(f"<b>{k}:</b> {v}" for k, v in le_items)

    w_axis = (
        f"({wf.base_axis.downrange_mps:.2f}, "
        f"{wf.base_axis.vertical_mps:.2f}, "
        f"{wf.base_axis.lateral_mps:.2f}) m/s"
    )
    w_shaft = (
        f"({wf.shaft_rotation.downrange_mps:.2f}, "
        f"{wf.shaft_rotation.vertical_mps:.2f}, "
        f"{wf.shaft_rotation.lateral_mps:.2f}) m/s"
    )
    w_other = (
        f"({wf.other_rotation.downrange_mps:.2f}, "
        f"{wf.other_rotation.vertical_mps:.2f}, "
        f"{wf.other_rotation.lateral_mps:.2f}) m/s"
    )
    w_total = (
        f"({wf.total_contact.downrange_mps:.2f}, "
        f"{wf.total_contact.vertical_mps:.2f}, "
        f"{wf.total_contact.lateral_mps:.2f}) m/s"
    )
    waterfall_html = (
        f"<b>Axis Translation:</b> {w_axis} + "
        f"<b>Shaft Rotation:</b> {w_shaft} + "
        f"<b>Other Rotation:</b> {w_other} = "
        f"<b>Total Contact:</b> {w_total}"
    )

    aoa_items = (
        ("Total AoA", _number(wf.total_aoa_deg, "°")),
        ("Without Shaft AoA", _number(wf.without_shaft_aoa_deg, "°")),
        ("Shaft AoA Δ", _number(wf.shaft_counterfactual_aoa_delta_deg, "°")),
        ("Shaft Shapley AoA", _number(wf.shaft_shapley_aoa_deg, "°")),
        ("Other Shapley AoA", _number(wf.other_shapley_aoa_deg, "°")),
    )
    aoa_html = " • ".join(f"<b>{k}:</b> {v}" for k, v in aoa_items)

    lp = metrics.low_point_world_m
    lp_text = (
        f"({lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f}) m"
        if lp is not None
        else "Unavailable"
    )

    return (
        f"<br><b>Wedge Delivery Metrics:</b> {le_html}<br>"
        f"<b>Low Point:</b> {lp_text}<br>"
        f"<b>Linear-Velocity Contribution Waterfall:</b> {waterfall_html}<br>"
        f"<b>Attack Angle Attribution:</b> {aoa_html}<br>"
        "<i>Linear velocity components are strictly additive "
        "(v_contact = v_axis + v_shaft + v_other). "
        "Attack angles are nonlinear (atan2) and reported as "
        "counterfactual deltas; never Euler-additive.</i>"
    )


def format_simulation_engineering_readout(run: SimulationRun) -> str:
    """Format impact metrics, ground clearance, and wedge delivery metrics."""
    impact_html = format_impact_kinematics(impact_kinematics_for_run(run))
    try:
        ground_snapshot = simulation_ground_clearance_snapshot(run)
    except ValueError as error:
        return (
            impact_html
            + "<br><b>Wedge Ground-Clearance:</b> Unavailable — "
            + escape(str(error))
        )
    ground_html = (
        "" if ground_snapshot is None else _format_ground_clearance(ground_snapshot)
    )
    try:
        delivery_metrics = simulation_wedge_delivery_metrics(run)
    except ValueError:
        delivery_metrics = None
    delivery_html = (
        ""
        if delivery_metrics is None
        else format_wedge_delivery_metrics(delivery_metrics)
    )
    return impact_html + ground_html + delivery_html


def format_simulation_key_metrics(run: SimulationRun | None) -> str:
    """Return the essential current-calculation metrics for persistent display."""
    if run is None:
        return "Run a simulation to inspect key impact metrics."
    snapshot = impact_kinematics_for_run(run)
    analysis = snapshot.analysis
    metrics = (
        ("Contact AoA", _number(analysis.total_aoa_deg, "°")),
        (
            "Shaft contribution",
            _number(analysis.shaft_counterfactual_aoa_delta_deg, "°"),
        ),
        (
            "Face-center spin loft",
            _number(snapshot.face_center_dplane.spin_loft_3d_deg, "°"),
        ),
        ("D-plane tilt", _number(snapshot.face_center_dplane.dplane_tilt_deg, "°")),
    )
    values = " · ".join(f"<b>{label}:</b> {value}" for label, value in metrics)
    return f"<b>{escape(snapshot.event_label)}:</b> {values}"
