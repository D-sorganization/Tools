"""Shared labels and display specifications for the simulation UI."""

from __future__ import annotations

__all__ = ["LAUNCH_ROWS", "RATE_PRESETS", "SOURCE_LABELS", "TILT_SPECS"]

SOURCE_LABELS: dict[str, str] = {
    "manual": "Manual Scenario (Constant Twist)",
    "double_pendulum": "Double Pendulum",
    "triple_pendulum": "Triple Pendulum",
}

# (launch field, Title Case label, unit suffix) in display order.
LAUNCH_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ball_speed_mph", "Ball Speed", " mph"),
    ("launch_angle_deg", "Launch Angle", "°"),
    ("launch_azimuth_deg", "Launch Direction", "°"),
    ("spin_rpm", "Total Spin", " rpm"),
    ("carry_m", "Carry Distance", " m"),
    ("max_height_m", "Apex Height", " m"),
    ("flight_time_s", "Flight Time", " s"),
    ("landing_angle_deg", "Landing Angle", "°"),
)

TILT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("yaw_deg", "Plane Yaw", "plane_yaw_deg"),
    ("side_tilt_deg", "Plane Side Tilt", "plane_side_tilt_deg"),
    ("forward_tilt_deg", "Plane Forward Tilt", "plane_forward_tilt_deg"),
)

# 1x maps one second of wall time to one second of simulated time.
RATE_PRESETS: tuple[tuple[str, float], ...] = (
    ("0.1×", 0.1),
    ("0.25×", 0.25),
    ("0.5×", 0.5),
    ("1× real-time", 1.0),
    ("2×", 2.0),
)
