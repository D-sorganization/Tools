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
        "launch_azimuth_deg": "Launch Direction",
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
    }
)

__all__ = ["OUTPUT_LABELS", "OUTPUT_UNITS"]
