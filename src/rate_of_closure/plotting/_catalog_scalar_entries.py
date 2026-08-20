"""Scalar input, impact, launch, and metric catalog rows."""

from rate_of_closure.model import solve
from rate_of_closure.plotting import optional_values
from rate_of_closure.plotting._catalog_entry_types import CatalogRow

INPUT_ROWS: tuple[CatalogRow, ...] = (
    (
        "clubhead_speed_mph",
        "Clubhead Speed",
        "mph",
        lambda run: float(run.config.scenario.clubhead_speed_mph),
    ),
    (
        "omega_plane_dps",
        "In-Plane Rotation (SPV)",
        "deg/s",
        lambda run: float(run.config.scenario.omega_plane_dps),
    ),
    (
        "omega_shaft_dps",
        "About-Shaft Rotation (HTV)",
        "deg/s",
        lambda run: float(run.config.scenario.omega_shaft_dps),
    ),
    (
        "lie_angle_deg",
        "Shaft Lie at Impact",
        "deg",
        lambda run: float(run.config.scenario.lie_angle_deg),
    ),
    (
        "com_to_face_mm",
        "GC to Face Center",
        "mm",
        lambda run: float(run.config.scenario.com_to_face_mm),
    ),
    (
        "impact_offset_toe_mm",
        "Impact Toward Toe",
        "mm",
        lambda run: float(run.config.scenario.impact_offset_toe_mm),
    ),
    (
        "impact_offset_high_mm",
        "Impact Above Center",
        "mm",
        lambda run: float(run.config.scenario.impact_offset_high_mm),
    ),
    (
        "contact_duration_us",
        "Contact Duration",
        "µs",
        lambda run: float(run.config.scenario.contact_duration_us),
    ),
    (
        "plane_yaw_deg",
        "Plane Yaw",
        "deg",
        lambda run: float(run.config.plane.yaw_deg),
    ),
    (
        "plane_side_tilt_deg",
        "Plane Side Tilt",
        "deg",
        lambda run: float(run.config.plane.side_tilt_deg),
    ),
    (
        "plane_forward_tilt_deg",
        "Plane Forward Tilt",
        "deg",
        lambda run: float(run.config.plane.forward_tilt_deg),
    ),
    (
        "impact_time_s",
        "Impact Time (τ)",
        "s",
        lambda run: optional_values.optional_float(run.impact_time_s),
    ),
)

IMPACT_ROWS: tuple[CatalogRow, ...] = (
    (
        "clubhead_speed_mps",
        "Delivered Clubhead Speed",
        "m/s",
        optional_values.delivered_speed,
    ),
    ("club_path_deg", "Club Path", "deg", optional_values.path_deg),
    (
        "attack_angle_deg",
        "Attack Angle",
        "deg",
        optional_values.attack_angle_deg,
    ),
    (
        "spin_loft_deg",
        "Spin Loft",
        "deg",
        lambda run: optional_values.delivery_scalar(run, "spin_loft_deg"),
    ),
    (
        "face_to_path_deg",
        "Face to Path",
        "deg",
        lambda run: optional_values.delivery_scalar(run, "face_to_path_deg"),
    ),
    (
        "spin_axis_tilt_deg",
        "Spin Axis Tilt",
        "deg",
        lambda run: optional_values.delivery_scalar(run, "spin_axis_tilt_deg"),
    ),
    (
        "energy_transfer_j",
        "Impact Energy Transfer",
        "J",
        optional_values.impact_energy,
    ),
)

LAUNCH_ROWS: tuple[CatalogRow, ...] = (
    (
        "ball_speed_mph",
        "Ball Speed",
        "mph",
        lambda run: optional_values.launch_scalar(run, "ball_speed_mph"),
    ),
    (
        "launch_angle_deg",
        "Launch Angle",
        "deg",
        lambda run: optional_values.launch_scalar(run, "launch_angle_deg"),
    ),
    (
        "launch_azimuth_deg",
        "Launch Direction",
        "deg",
        lambda run: optional_values.launch_scalar(run, "launch_azimuth_deg"),
    ),
    (
        "spin_rpm",
        "Total Spin",
        "rpm",
        lambda run: optional_values.launch_scalar(run, "spin_rpm"),
    ),
)

METRIC_ROWS: tuple[CatalogRow, ...] = (
    (
        "carry_m",
        "Carry Distance",
        "m",
        lambda run: optional_values.launch_scalar(run, "carry_m"),
    ),
    (
        "max_height_m",
        "Apex Height",
        "m",
        lambda run: optional_values.launch_scalar(run, "max_height_m"),
    ),
    (
        "flight_time_s",
        "Flight Time",
        "s",
        lambda run: optional_values.launch_scalar(run, "flight_time_s"),
    ),
    (
        "landing_angle_deg",
        "Landing Angle",
        "deg",
        lambda run: optional_values.launch_scalar(run, "landing_angle_deg"),
    ),
    (
        "path_deviation_deg",
        "Impact-Point Path Deviation",
        "deg",
        lambda run: float(solve(run.config.scenario).path_deviation_deg),
    ),
    (
        "closure_rate_dps",
        "Closure Rate (CCV)",
        "deg/s",
        lambda run: float(solve(run.config.scenario).closure_rate_dps),
    ),
)
