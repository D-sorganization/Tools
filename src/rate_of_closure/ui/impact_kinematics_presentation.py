"""Shared text presentation for frame-explicit wedge impact kinematics."""

from __future__ import annotations

import math

from rate_of_closure.simulation import ImpactKinematicSnapshot

__all__ = ["format_impact_kinematics"]


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
    )
    metric_html = " • ".join(f"<b>{label}:</b> {value}" for label, value in metrics)
    return (
        f"<b>{snapshot.event_label} Kinematics</b> at "
        f"{snapshot.event_time_s:.3f} s — {metric_html}<br>"
        f"<b>Geometry Basis:</b> {snapshot.geometry_basis}. "
        f"<b>Model Boundary:</b> {snapshot.model_limitations}"
    )
