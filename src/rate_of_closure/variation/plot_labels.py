"""Canonical display labels and units for variation scalar outputs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

OUTPUT_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "candidate_time_s": "s",
        "closest_approach_m": "m",
        "contact_margin_m": "m",
        "impact_time_s": "s",
        "clubhead_speed_mps": "m/s",
        "spin_loft_deg": "deg",
        "face_to_path_deg": "deg",
        "spin_axis_tilt_deg": "deg",
        "ball_speed_mph": "mph",
        "launch_angle_deg": "deg",
        "launch_azimuth_deg": "deg",
        "spin_rpm": "rpm",
        "carry_m": "m",
        "lateral_m": "m",
        "max_height_m": "m",
        "flight_time_s": "s",
        "landing_angle_deg": "deg",
        "club_path_deg": "deg",
        "face_angle_deg": "deg",
        "attack_angle_deg": "deg",
        "dynamic_loft_deg": "deg",
        "spin_axis_deg": "deg",
        "apex_m": "m",
        "loss": "1",
        "constraint_violated": "0/1",
        "leading_edge_clearance_at_ball_m": "m",
        "minimum_pre_ball_clearance_m": "m",
        "ground_after_ball_margin_s": "s",
        "low_point_clearance_m": "m",
        "delivered_bounce_deg": "deg",
        "path_projected_effective_bounce_deg": "deg",
        "reference_aoa_deg": "deg",
        "bounce_utilization_margin_deg": "deg",
        "peak_turf_penetration_m": "m",
        "normal_turf_impulse_n_s": "N s",
        "shaft_rotation_rate_rad_s": "rad/s",
        "shaft_counterfactual_aoa_delta_deg": "deg",
        "shaft_shapley_aoa_deg": "deg",
        "shaft_vertical_velocity_share": "1",
        "leading_edge_3d_rate_rad_s": "rad/s",
        "face_normal_3d_rate_rad_s": "rad/s",
        "leading_edge_relative_arc_heading_rate_rad_s": "rad/s",
    }
)

OUTPUT_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "candidate_time_s": "Candidate Contact Time",
        "closest_approach_m": "Closest Approach",
        "contact_margin_m": "Contact Margin",
        "impact_time_s": "Impact Time",
        "clubhead_speed_mps": "Clubhead Speed",
        "spin_loft_deg": "Spin Loft",
        "face_to_path_deg": "Face to Path",
        "spin_axis_tilt_deg": "Spin-Axis Tilt",
        "ball_speed_mph": "Ball Speed",
        "launch_angle_deg": "Launch Angle",
        "launch_azimuth_deg": "Launch Azimuth",
        "spin_rpm": "Spin Rate",
        "carry_m": "Carry",
        "lateral_m": "Lateral Landing Position",
        "max_height_m": "Maximum Height",
        "flight_time_s": "Flight Time",
        "landing_angle_deg": "Landing Angle",
        "club_path_deg": "Club Path",
        "face_angle_deg": "Face Angle",
        "attack_angle_deg": "Attack Angle",
        "dynamic_loft_deg": "Dynamic Loft",
        "spin_axis_deg": "Spin-Axis Tilt",
        "apex_m": "Apex Height",
        "loss": "Decision Loss",
        "constraint_violated": "Constraint Violation",
        "leading_edge_clearance_at_ball_m": "Leading-Edge Clearance at Ball",
        "minimum_pre_ball_clearance_m": "Minimum Pre-Ball Clearance",
        "ground_after_ball_margin_s": "Ground-After-Ball Time Margin",
        "low_point_clearance_m": "Low-Point Clearance",
        "delivered_bounce_deg": "Delivered Bounce",
        "path_projected_effective_bounce_deg": "Path-Projected Effective Bounce",
        "reference_aoa_deg": "Reference-Point Angle of Attack",
        "bounce_utilization_margin_deg": "Bounce-Utilization Margin",
        "peak_turf_penetration_m": "Peak Turf Penetration",
        "normal_turf_impulse_n_s": "Normal Turf Impulse",
        "shaft_rotation_rate_rad_s": "Shaft Rotation Rate",
        "shaft_counterfactual_aoa_delta_deg": "Shaft Counterfactual AoA Change",
        "shaft_shapley_aoa_deg": "Shaft AoA Shapley Contribution",
        "shaft_vertical_velocity_share": "Shaft Vertical-Velocity Share",
        "leading_edge_3d_rate_rad_s": "Leading-Edge 3D Rate",
        "face_normal_3d_rate_rad_s": "Face-Normal 3D Rate",
        "leading_edge_relative_arc_heading_rate_rad_s": (
            "Leading-Edge Arc-Relative Heading Rate"
        ),
    }
)

__all__ = ["OUTPUT_LABELS", "OUTPUT_UNITS"]
