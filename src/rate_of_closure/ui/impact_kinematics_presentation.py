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

__all__ = [
    "format_impact_kinematics",
    "format_simulation_engineering_readout",
    "format_simulation_key_metrics",
    "ground_clearance_snapshot_for_scene",
    "simulation_ground_clearance_snapshot",
]


def _number(value: float | None, unit: str, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "Unavailable"
    return f"{value:.{decimals}f} {unit}"


def _degrees_per_second(value_rad_s: float | None) -> str:
    return _number(None if value_rad_s is None else math.degrees(value_rad_s), "°/s", 1)


def _vector(value: tuple[float, float, float], unit: str) -> str:
    return f"[{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}] {unit}"


def format_impact_kinematics(snapshot: ImpactKinematicSnapshot) -> str:
    """Return a compact, provenance-bearing engineering readout."""
    analysis = snapshot.analysis
    shaft_vertical = analysis.shaft_rotation_velocity_mps[1]
    screw_distance = (
        None if analysis.screw_axis is None else analysis.screw_axis.contact_distance_m
    )
    sasho = analysis.sasho_face_center_rotation
    sasho_horizontal = math.hypot(sasho.velocity_mps[0], sasho.velocity_mps[2])
    metrics = (
        ("Contact-Point AoA", _number(analysis.total_aoa_deg, "°")),
        ("Without Shaft Rotation", _number(analysis.without_shaft_aoa_deg, "°")),
        (
            "Shaft AoA Contribution",
            _number(analysis.shaft_counterfactual_aoa_delta_deg, "°"),
        ),
        (
            "Shaft-Rotation Shapley AoA",
            _number(analysis.shaft_shapley_aoa_deg, "°"),
        ),
        (
            "Sasho Face-Center Rotation-Only AoA",
            _number(analysis.sasho_face_center_rotation.aoa_deg, "°"),
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
        "<b>AoA Method Options:</b> remove-shaft counterfactual, two-factor "
        "Shapley attribution, and Sasho nearest-shaft face-center rotation-only "
        "AoA are separate non-additive measures. "
        f"<b>Sasho Geometry:</b> method {escape(sasho.method_id)}; nearest shaft "
        f"point {_vector(sasho.nearest_shaft_point_m, 'm')}; perpendicular lever "
        f"{_vector(sasho.lever_arm_m, 'm')}; complete angular velocity "
        f"{_vector(snapshot.state.angular_velocity_rad_s, 'rad/s')}; rotation-only "
        f"velocity {_vector(sasho.velocity_mps, 'm/s')}; vertical/horizontal "
        f"{sasho.velocity_mps[1]:.4f}/{sasho_horizontal:.4f} m/s. "
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


def format_simulation_engineering_readout(run: SimulationRun) -> str:
    """Format impact metrics and wedge-only swept ground-clearance metrics."""
    impact_html = format_impact_kinematics(impact_kinematics_for_run(run))
    try:
        ground_snapshot = simulation_ground_clearance_snapshot(run)
    except ValueError as error:
        return (
            impact_html
            + "<br><b>Wedge Ground-Clearance:</b> Unavailable — "
            + escape(str(error))
        )
    return (
        impact_html
        if ground_snapshot is None
        else impact_html + _format_ground_clearance(ground_snapshot)
    )


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
