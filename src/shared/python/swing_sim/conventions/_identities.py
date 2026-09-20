"""Intrinsic parameter identities for all audited launch-monitor parameters."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import (
    ParameterId,
    SignRule,
)


@dataclass(frozen=True)
class Identity:
    """Intrinsic parameter identity independent of measurement convention."""

    label: str
    unit: str
    sign: SignRule
    geometry: str
    definition: str


IDENTITIES: dict[ParameterId, Identity] = {
    # Club Delivery
    ParameterId.CLUB_SPEED: Identity(
        "Club Speed",
        "m/s",
        SignRule.NONNEGATIVE,
        "magnitude(club_velocity)",
        "Linear speed of the club head reference point.",
    ),
    ParameterId.CLUB_PATH: Identity(
        "Club Path",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "heading(club_velocity)",
        "Horizontal direction of club head motion relative to target line.",
    ),
    ParameterId.ATTACK_ANGLE: Identity(
        "Attack Angle",
        "deg",
        SignRule.POSITIVE_UP,
        "elevation(club_velocity)",
        "Vertical direction of club head motion relative to horizontal.",
    ),
    ParameterId.DYNAMIC_LIE: Identity(
        "Dynamic Lie",
        "deg",
        SignRule.POSITIVE_UP,
        "elevation(shaft_axis_transverse)",
        "Angle of club shaft or head sole relative to ground at impact.",
    ),
    ParameterId.CLOSURE_RATE: Identity(
        "Closure Rate",
        "deg/s",
        SignRule.POSITIVE_RIGHT,
        "yaw_rate(face_normal)",
        "Rate of rotation of the club face closing toward the swing path.",
    ),
    ParameterId.SWING_DIRECTION: Identity(
        "Swing Direction",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "heading(swing_plane_base)",
        "Horizontal direction of the base of the swing plane.",
    ),
    ParameterId.LOW_POINT: Identity(
        "Low Point",
        "m",
        SignRule.POSITIVE_RIGHT,
        "arc_distance(impact, lowest_point)",
        "Distance before or after impact where club head reaches minimum height.",
    ),
    # Face Orientation & Impact
    ParameterId.FACE_ANGLE: Identity(
        "Face Angle",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "heading(face_normal)",
        "Horizontal direction the club face points relative to target line.",
    ),
    ParameterId.DYNAMIC_LOFT: Identity(
        "Dynamic Loft",
        "deg",
        SignRule.POSITIVE_UP,
        "elevation(face_normal)",
        "Vertical angle of the club face normal relative to horizontal.",
    ),
    ParameterId.FACE_TO_PATH: Identity(
        "Face to Path",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "wrapped(face_angle-club_path)",
        "Difference between face angle and club path.",
    ),
    ParameterId.SPIN_LOFT: Identity(
        "Spin Loft",
        "deg",
        SignRule.NONNEGATIVE,
        "angle_3d(club_velocity,face_normal)",
        "Three-dimensional angle between club head velocity and face normal.",
    ),
    ParameterId.IMPACT_OFFSET: Identity(
        "Impact Offset",
        "m",
        SignRule.POSITIVE_RIGHT,
        "face_coordinate_x(impact)",
        "Horizontal offset of ball impact from face center (+ toe / - heel).",
    ),
    ParameterId.IMPACT_HEIGHT: Identity(
        "Impact Height",
        "m",
        SignRule.POSITIVE_UP,
        "face_coordinate_y(impact)",
        "Vertical offset of ball impact from face center (+ high / - low).",
    ),
    # Ball Launch
    ParameterId.BALL_SPEED: Identity(
        "Ball Speed",
        "m/s",
        SignRule.NONNEGATIVE,
        "magnitude(initial_ball_velocity)",
        "Linear speed of the golf ball immediately after leaving the club face.",
    ),
    ParameterId.LAUNCH_ANGLE: Identity(
        "Launch Angle",
        "deg",
        SignRule.POSITIVE_UP,
        "elevation(initial_ball_velocity)",
        "Vertical angle of initial ball velocity vector relative to ground.",
    ),
    ParameterId.LAUNCH_DIRECTION: Identity(
        "Launch Direction",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "heading(initial_ball_velocity)",
        "Horizontal direction of initial ball flight relative to target line.",
    ),
    ParameterId.SMASH_FACTOR: Identity(
        "Smash Factor",
        "ratio",
        SignRule.NONNEGATIVE,
        "ball_speed / club_speed",
        "Ratio of ball speed to club speed representing energy transfer efficiency.",
    ),
    # Ball Spin
    ParameterId.TOTAL_SPIN: Identity(
        "Total Spin",
        "rpm",
        SignRule.NONNEGATIVE,
        "magnitude(ball_angular_velocity)",
        "Total rate of rotation of the golf ball immediately after separation.",
    ),
    ParameterId.SPIN_AXIS: Identity(
        "Spin Axis",
        "deg",
        SignRule.POSITIVE_RIGHT,
        "tilt_angle(ball_angular_velocity)",
        "Angle of ball rotation axis relative to horizontal (+ right tilt / slice).",
    ),
    ParameterId.BACK_SPIN: Identity(
        "Back Spin",
        "rpm",
        SignRule.NONNEGATIVE,
        "projected_backspin(spin_vector)",
        (
            "Backspin component of rotation around horizontal axis "
            "perpendicular to velocity."
        ),
    ),
    ParameterId.SIDE_SPIN: Identity(
        "Side Spin",
        "rpm",
        SignRule.POSITIVE_RIGHT,
        "projected_sidespin(spin_vector)",
        "Sidespin component around vertical axis (+ clockwise / right curve).",
    ),
    # Ball Flight
    ParameterId.APEX_HEIGHT: Identity(
        "Apex Height",
        "m",
        SignRule.POSITIVE_UP,
        "max_y(trajectory)",
        "Peak vertical height reached by the ball trajectory above ground level.",
    ),
    ParameterId.CARRY_DISTANCE: Identity(
        "Carry Distance",
        "m",
        SignRule.NONNEGATIVE,
        "downrange_x(landing_point)",
        "Downrange distance from launch point to first ground impact.",
    ),
    ParameterId.TOTAL_DISTANCE: Identity(
        "Total Distance",
        "m",
        SignRule.NONNEGATIVE,
        "downrange_x(final_rest_point)",
        "Total downrange distance including carry and ground rollout.",
    ),
    ParameterId.CARRY_OFFLINE: Identity(
        "Carry Offline",
        "m",
        SignRule.POSITIVE_RIGHT,
        "lateral_z(landing_point)",
        "Lateral distance from target line at point of first ground impact.",
    ),
    ParameterId.CURVE: Identity(
        "Curve",
        "m",
        SignRule.POSITIVE_RIGHT,
        "lateral_deviation_from_launch_azimuth(landing)",
        "Lateral distance between landing point and initial launch direction ray.",
    ),
    ParameterId.FLIGHT_TIME: Identity(
        "Flight Time",
        "s",
        SignRule.NONNEGATIVE,
        "t_landing - t_launch",
        "Total duration the ball remains airborne from launch to landing.",
    ),
    ParameterId.LANDING_ANGLE: Identity(
        "Landing Angle",
        "deg",
        SignRule.POSITIVE_UP,
        "elevation(landing_velocity)",
        "Descent angle of ball velocity vector relative to ground at landing.",
    ),
}
