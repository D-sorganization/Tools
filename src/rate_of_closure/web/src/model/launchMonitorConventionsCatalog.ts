/** Source-backed parameter identities and policies for launch-monitor conventions. */

import type {
  AvailabilityRule,
  EventTime,
  ParameterGroup,
  ParameterId,
  QuantityStatus,
  ReferencePoint,
  SignRule,
} from "./launchMonitorConventions";

export interface IdentityTs {
  readonly label: string;
  readonly unit: string;
  readonly signRule: SignRule;
  readonly geometryContract: string;
  readonly definition: string;
}

export interface PolicyTs {
  readonly referencePoint: ReferencePoint;
  readonly eventTime: EventTime;
  readonly quantityStatus: QuantityStatus;
  readonly availability: AvailabilityRule;
  readonly sourceUrl: string;
  readonly signRule?: SignRule;
}

export const FRAME_ID = "target_frame:x_target,y_up,z_right";
export const RETRIEVED_ON = "2026-08-05";

export const APP_SOURCE = "https://github.com/D-sorganization/Tools/blob/main/docs/specs/D_PLANE_GEOMETRY.md";
export const TRACKMAN_CLUB = "https://www.trackman.com/blog/golf/club-data-definitions";
export const TRACKMAN_PARAMETERS = "https://www.trackman.com/blog/golf/40-trackman-parameters";
export const FORESIGHT_CLUB = "https://help.foresightsports.com/hc/en-us/articles/47214673873811-Club-Head-Data-Measurements-Definitions";
export const FORESIGHT_BALL = "https://help.foresightsports.com/hc/en-us/articles/47144162581523-Ball-Launch-Data-Measurements-Ball-Flight-Results";

export const PARAMETER_GROUP_MAP: Record<ParameterId, ParameterGroup> = {
  club_speed: "club_delivery",
  club_path: "club_delivery",
  attack_angle: "club_delivery",
  dynamic_lie: "club_delivery",
  closure_rate: "club_delivery",
  swing_direction: "club_delivery",
  low_point: "club_delivery",
  face_angle: "face_orientation",
  dynamic_loft: "face_orientation",
  face_to_path: "face_orientation",
  spin_loft: "face_orientation",
  impact_offset: "face_orientation",
  impact_height: "face_orientation",
  ball_speed: "ball_launch",
  launch_angle: "ball_launch",
  launch_direction: "ball_launch",
  smash_factor: "ball_launch",
  total_spin: "ball_spin",
  spin_axis: "ball_spin",
  back_spin: "ball_spin",
  side_spin: "ball_spin",
  apex_height: "ball_flight",
  carry_distance: "ball_flight",
  total_distance: "ball_flight",
  carry_offline: "ball_flight",
  curve: "ball_flight",
  flight_time: "ball_flight",
  landing_angle: "ball_flight",
};

export const GROUP_LABELS: Record<ParameterGroup, string> = {
  club_delivery: "Club Delivery",
  face_orientation: "Face Orientation",
  ball_launch: "Ball Launch",
  ball_spin: "Ball Spin",
  ball_flight: "Ball Flight",
};

export const IDENTITIES: Record<ParameterId, IdentityTs> = {
  club_speed: { label: "Club Speed", unit: "m/s", signRule: "nonnegative", geometryContract: "magnitude(club_velocity)", definition: "Linear speed of the club head reference point." },
  club_path: { label: "Club Path", unit: "deg", signRule: "positive_right", geometryContract: "heading(club_velocity)", definition: "Horizontal direction of club head motion relative to target line." },
  attack_angle: { label: "Attack Angle", unit: "deg", signRule: "positive_up", geometryContract: "elevation(club_velocity)", definition: "Vertical direction of club head motion relative to horizontal." },
  dynamic_lie: { label: "Dynamic Lie", unit: "deg", signRule: "positive_up", geometryContract: "elevation(shaft_axis_transverse)", definition: "Angle of club shaft or head sole relative to ground at impact." },
  closure_rate: { label: "Closure Rate", unit: "deg/s", signRule: "positive_right", geometryContract: "yaw_rate(face_normal)", definition: "Rate of rotation of the club face closing toward the swing path." },
  swing_direction: { label: "Swing Direction", unit: "deg", signRule: "positive_right", geometryContract: "heading(swing_plane_base)", definition: "Horizontal direction of the base of the swing plane." },
  low_point: { label: "Low Point", unit: "m", signRule: "positive_right", geometryContract: "arc_distance(impact, lowest_point)", definition: "Distance before or after impact where club head reaches minimum height." },
  face_angle: { label: "Face Angle", unit: "deg", signRule: "positive_right", geometryContract: "heading(face_normal)", definition: "Horizontal direction the club face points relative to target line." },
  dynamic_loft: { label: "Dynamic Loft", unit: "deg", signRule: "positive_up", geometryContract: "elevation(face_normal)", definition: "Vertical angle of the club face normal relative to horizontal." },
  face_to_path: { label: "Face to Path", unit: "deg", signRule: "positive_right", geometryContract: "wrapped(face_angle-club_path)", definition: "Difference between face angle and club path." },
  spin_loft: { label: "Spin Loft", unit: "deg", signRule: "nonnegative", geometryContract: "angle_3d(club_velocity,face_normal)", definition: "Three-dimensional angle between club head velocity and face normal." },
  impact_offset: { label: "Impact Offset", unit: "m", signRule: "positive_right", geometryContract: "face_coordinate_x(impact)", definition: "Horizontal offset of ball impact from face center (+ toe / - heel)." },
  impact_height: { label: "Impact Height", unit: "m", signRule: "positive_up", geometryContract: "face_coordinate_y(impact)", definition: "Vertical offset of ball impact from face center (+ high / - low)." },
  ball_speed: { label: "Ball Speed", unit: "m/s", signRule: "nonnegative", geometryContract: "magnitude(initial_ball_velocity)", definition: "Linear speed of the golf ball immediately after leaving the club face." },
  launch_angle: { label: "Launch Angle", unit: "deg", signRule: "positive_up", geometryContract: "elevation(initial_ball_velocity)", definition: "Vertical angle of initial ball velocity vector relative to ground." },
  launch_direction: { label: "Launch Direction", unit: "deg", signRule: "positive_right", geometryContract: "heading(initial_ball_velocity)", definition: "Horizontal direction of initial ball flight relative to target line." },
  smash_factor: { label: "Smash Factor", unit: "ratio", signRule: "nonnegative", geometryContract: "ball_speed / club_speed", definition: "Ratio of ball speed to club speed representing energy transfer efficiency." },
  total_spin: { label: "Total Spin", unit: "rpm", signRule: "nonnegative", geometryContract: "magnitude(ball_angular_velocity)", definition: "Total rate of rotation of the golf ball immediately after separation." },
  spin_axis: { label: "Spin Axis", unit: "deg", signRule: "positive_right", geometryContract: "tilt_angle(ball_angular_velocity)", definition: "Angle of ball rotation axis relative to horizontal (+ right tilt / slice)." },
  back_spin: { label: "Back Spin", unit: "rpm", signRule: "nonnegative", geometryContract: "projected_backspin(spin_vector)", definition: "Backspin component of rotation around horizontal axis perpendicular to velocity." },
  side_spin: { label: "Side Spin", unit: "rpm", signRule: "positive_right", geometryContract: "projected_sidespin(spin_vector)", definition: "Sidespin component around vertical axis (+ clockwise / right curve)." },
  apex_height: { label: "Apex Height", unit: "m", signRule: "positive_up", geometryContract: "max_y(trajectory)", definition: "Peak vertical height reached by the ball trajectory above ground level." },
  carry_distance: { label: "Carry Distance", unit: "m", signRule: "nonnegative", geometryContract: "downrange_x(landing_point)", definition: "Downrange distance from launch point to first ground impact." },
  total_distance: { label: "Total Distance", unit: "m", signRule: "nonnegative", geometryContract: "downrange_x(final_rest_point)", definition: "Total downrange distance including carry and ground rollout." },
  carry_offline: { label: "Carry Offline", unit: "m", signRule: "positive_right", geometryContract: "lateral_z(landing_point)", definition: "Lateral distance from target line at point of first ground impact." },
  curve: { label: "Curve", unit: "m", signRule: "positive_right", geometryContract: "lateral_deviation_from_launch_azimuth(landing)", definition: "Lateral distance between landing point and initial launch direction ray." },
  flight_time: { label: "Flight Time", unit: "s", signRule: "nonnegative", geometryContract: "t_landing - t_launch", definition: "Total duration the ball remains airborne from launch to landing." },
  landing_angle: { label: "Landing Angle", unit: "deg", signRule: "positive_up", geometryContract: "elevation(landing_velocity)", definition: "Descent angle of ball velocity vector relative to ground at landing." },
};

const appClub = (referencePoint: ReferencePoint, eventTime: EventTime = "inspection_event"): PolicyTs => ({
  referencePoint, eventTime, quantityStatus: "derived", availability: "nonzero_club_travel", sourceUrl: APP_SOURCE,
});
const appFace = (referencePoint: ReferencePoint): PolicyTs => ({
  referencePoint, eventTime: "inspection_event", quantityStatus: "derived", availability: "face_geometry", sourceUrl: APP_SOURCE,
});
const appBall = (): PolicyTs => ({
  referencePoint: "ball_center", eventTime: "just_after_separation", quantityStatus: "modeled", availability: "collision_complete", sourceUrl: APP_SOURCE,
});
const appFlight = (eventTime: EventTime): PolicyTs => ({
  referencePoint: "ball_center", eventTime, quantityStatus: "modeled", availability: "trajectory_complete", sourceUrl: APP_SOURCE,
});

export const APP_POLICIES: Record<ParameterId, PolicyTs> = {
  club_speed: appClub("tracked_head_reference"),
  club_path: appClub("tracked_head_reference"),
  attack_angle: appClub("tracked_head_reference"),
  dynamic_lie: appFace("face_center"),
  closure_rate: appFace("face_center"),
  swing_direction: appClub("tracked_head_reference"),
  low_point: appClub("tracked_head_reference"),
  face_angle: appFace("face_center"),
  dynamic_loft: appFace("face_center"),
  face_to_path: appFace("mixed_club_delivery"),
  spin_loft: appFace("mixed_club_delivery"),
  impact_offset: appFace("impact_location"),
  impact_height: appFace("impact_location"),
  ball_speed: appBall(),
  launch_angle: appBall(),
  launch_direction: appBall(),
  smash_factor: { referencePoint: "mixed_club_delivery", eventTime: "just_after_separation", quantityStatus: "derived", availability: "collision_complete", sourceUrl: APP_SOURCE },
  total_spin: appBall(),
  spin_axis: appBall(),
  back_spin: appBall(),
  side_spin: appBall(),
  apex_height: appFlight("apex"),
  carry_distance: appFlight("landing"),
  total_distance: appFlight("landing"),
  carry_offline: appFlight("landing"),
  curve: appFlight("landing"),
  flight_time: appFlight("flight_duration"),
  landing_angle: appFlight("landing"),
};

const tmClub = (referencePoint: ReferencePoint, eventTime: EventTime, sourceUrl: string): PolicyTs => ({
  referencePoint, eventTime, quantityStatus: "measured_comparable", availability: "nonzero_club_travel", sourceUrl,
});
const tmFace = (referencePoint: ReferencePoint, eventTime: EventTime, sourceUrl: string): PolicyTs => ({
  referencePoint, eventTime, quantityStatus: "measured_comparable", availability: "face_geometry", sourceUrl,
});
const tmBall = (): PolicyTs => ({
  referencePoint: "ball_center", eventTime: "just_after_separation", quantityStatus: "measured_comparable", availability: "collision_complete", sourceUrl: TRACKMAN_PARAMETERS,
});
const tmFlight = (eventTime: EventTime): PolicyTs => ({
  referencePoint: "ball_center", eventTime, quantityStatus: "modeled", availability: "trajectory_complete", sourceUrl: TRACKMAN_PARAMETERS,
});

export const TRACKMAN_POLICIES: Record<ParameterId, PolicyTs> = {
  club_speed: tmClub("geometric_center", "just_before_first_contact", TRACKMAN_PARAMETERS),
  club_path: tmClub("geometric_center", "maximum_compression", TRACKMAN_CLUB),
  attack_angle: tmClub("geometric_center", "maximum_compression", TRACKMAN_CLUB),
  dynamic_lie: tmFace("face_center", "maximum_compression", TRACKMAN_CLUB),
  closure_rate: { referencePoint: "impact_location", eventTime: "maximum_compression", quantityStatus: "derived", availability: "face_geometry", sourceUrl: TRACKMAN_PARAMETERS },
  swing_direction: tmClub("geometric_center", "just_before_first_contact", TRACKMAN_PARAMETERS),
  low_point: tmClub("geometric_center", "maximum_compression", TRACKMAN_PARAMETERS),
  face_angle: tmFace("impact_location", "maximum_compression", TRACKMAN_CLUB),
  dynamic_loft: tmFace("impact_location", "maximum_compression", TRACKMAN_CLUB),
  face_to_path: { referencePoint: "mixed_club_delivery", eventTime: "maximum_compression", quantityStatus: "derived", availability: "face_geometry", sourceUrl: TRACKMAN_PARAMETERS },
  spin_loft: { referencePoint: "mixed_club_delivery", eventTime: "maximum_compression", quantityStatus: "derived", availability: "face_geometry", sourceUrl: TRACKMAN_PARAMETERS },
  impact_offset: tmFace("impact_location", "maximum_compression", TRACKMAN_CLUB),
  impact_height: tmFace("impact_location", "maximum_compression", TRACKMAN_CLUB),
  ball_speed: tmBall(),
  launch_angle: tmBall(),
  launch_direction: tmBall(),
  smash_factor: { referencePoint: "mixed_club_delivery", eventTime: "just_after_separation", quantityStatus: "derived", availability: "collision_complete", sourceUrl: TRACKMAN_PARAMETERS },
  total_spin: tmBall(),
  spin_axis: tmBall(),
  back_spin: tmBall(),
  side_spin: tmBall(),
  apex_height: tmFlight("apex"),
  carry_distance: tmFlight("landing"),
  total_distance: tmFlight("landing"),
  carry_offline: tmFlight("landing"),
  curve: tmFlight("landing"),
  flight_time: tmFlight("flight_duration"),
  landing_angle: tmFlight("landing"),
};

const fsClub = (referencePoint: ReferencePoint, eventTime: EventTime, sourceUrl: string): PolicyTs => ({
  referencePoint, eventTime, quantityStatus: "measured_comparable", availability: "nonzero_club_travel", sourceUrl,
});
const fsFace = (referencePoint: ReferencePoint, eventTime: EventTime, sourceUrl: string): PolicyTs => ({
  referencePoint, eventTime, quantityStatus: "measured_comparable", availability: "face_geometry", sourceUrl,
});
const fsBall = (signRule?: SignRule): PolicyTs => ({
  referencePoint: "ball_center", eventTime: "just_after_separation", quantityStatus: "measured_comparable", availability: "collision_complete", sourceUrl: FORESIGHT_BALL, ...(signRule ? { signRule } : {}),
});
const fsFlight = (eventTime: EventTime): PolicyTs => ({
  referencePoint: "ball_center", eventTime, quantityStatus: "modeled", availability: "trajectory_complete", sourceUrl: FORESIGHT_BALL,
});

export const FORESIGHT_POLICIES: Record<ParameterId, PolicyTs> = {
  club_speed: fsClub("face_center", "just_before_first_contact", FORESIGHT_CLUB),
  club_path: fsClub("face_center", "impact", FORESIGHT_CLUB),
  attack_angle: fsClub("face_center", "impact", FORESIGHT_CLUB),
  dynamic_lie: fsFace("face_center", "impact", FORESIGHT_CLUB),
  closure_rate: { referencePoint: "face_center", eventTime: "impact", quantityStatus: "derived", availability: "face_geometry", sourceUrl: FORESIGHT_CLUB },
  swing_direction: { referencePoint: "face_center", eventTime: "impact", quantityStatus: "unavailable", availability: "unavailable", sourceUrl: FORESIGHT_CLUB },
  low_point: { referencePoint: "face_center", eventTime: "impact", quantityStatus: "unavailable", availability: "unavailable", sourceUrl: FORESIGHT_CLUB },
  face_angle: fsFace("impact_location", "impact", FORESIGHT_CLUB),
  dynamic_loft: fsFace("impact_location", "impact", FORESIGHT_CLUB),
  face_to_path: { referencePoint: "mixed_club_delivery", eventTime: "impact", quantityStatus: "derived", availability: "face_geometry", sourceUrl: FORESIGHT_CLUB },
  spin_loft: { referencePoint: "mixed_club_delivery", eventTime: "impact", quantityStatus: "derived", availability: "face_geometry", sourceUrl: FORESIGHT_CLUB },
  impact_offset: fsFace("impact_location", "impact", FORESIGHT_CLUB),
  impact_height: fsFace("impact_location", "impact", FORESIGHT_CLUB),
  ball_speed: fsBall(),
  launch_angle: fsBall(),
  launch_direction: fsBall("unspecified"),
  smash_factor: { referencePoint: "mixed_club_delivery", eventTime: "just_after_separation", quantityStatus: "derived", availability: "collision_complete", sourceUrl: FORESIGHT_CLUB },
  total_spin: fsBall(),
  spin_axis: fsBall(),
  back_spin: fsBall(),
  side_spin: fsBall(),
  apex_height: fsFlight("apex"),
  carry_distance: fsFlight("landing"),
  total_distance: fsFlight("landing"),
  carry_offline: fsFlight("landing"),
  curve: { referencePoint: "ball_center", eventTime: "landing", quantityStatus: "unavailable", availability: "unavailable", sourceUrl: FORESIGHT_BALL },
  flight_time: fsFlight("flight_duration"),
  landing_angle: fsFlight("landing"),
};
