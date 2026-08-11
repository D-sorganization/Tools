/** Exact TypeScript records for the flight-to-ground v1 wire boundary. */

export const FLIGHT_TO_GROUND_REQUEST_VERSION = "flight-to-ground-request/v1" as const;
export const FLIGHT_TO_GROUND_RESULT_VERSION = "flight-to-ground-result/v1" as const;
export const GROUND_TARGET_FRAME = "target_frame:x_downrange,y_up,z_right" as const;

export type GroundVec3 = readonly [number, number, number];
export type GroundFrame = typeof GROUND_TARGET_FRAME;
export type CalibrationKind = "measured" | "literature" | "estimated" | "unvalidated";
export type GroundPhase = "impact" | "bounce" | "skid" | "roll" | "rest";
export type GroundEventType =
  | "first_contact"
  | "bounce"
  | "skid_to_roll"
  | "surface_transition"
  | "rest"
  | "left_surface";
export type GroundResultStatus = "complete" | "partial" | "failed" | "unavailable";
export type GroundTerminationReason =
  | "rest" | "time_limit" | "event_limit" | "left_surface"
  | "numerical_failure" | "unavailable_input";
export type GroundWarningSeverity = "info" | "warning" | "error";
export type GroundUnavailableFieldId =
  | "terminal_angular_velocity_rad_s" | "physical_contact_bracket" | "surface_profile";
export type GroundUnavailableReason =
  | "source_does_not_propagate" | "no_physical_contact"
  | "unsupported_surface" | "source_out_of_bounds";

export interface GroundProvenance {
  readonly producer: string;
  readonly producer_version: string;
  readonly source_revision: string;
  readonly input_sha256: string;
}

export interface GroundCalibration {
  readonly calibration_id: string;
  readonly kind: CalibrationKind;
  readonly source: string;
  readonly confidence: number;
}

export interface GroundSurfaceProfile {
  readonly surface_id: string;
  readonly provider_id: string;
  readonly provider_version: string;
  readonly frame: GroundFrame;
  readonly height_m: number;
  readonly normal_unit: GroundVec3;
  readonly surface_velocity_m_s: GroundVec3;
  readonly normal_restitution: number;
  readonly static_friction: number;
  readonly kinetic_friction: number;
  readonly rolling_resistance: number;
  readonly firmness_pa: number;
  readonly hardness_fraction: number;
  readonly grass_height_m: number;
  readonly compressibility_fraction: number;
  readonly compression_damping_fraction: number;
  readonly turf_density_kg_m3: number;
  readonly moisture_fraction: number;
}

export interface GroundContactState {
  readonly time_s: number;
  readonly frame: GroundFrame;
  readonly position_m: GroundVec3;
  readonly velocity_m_s: GroundVec3;
  readonly angular_velocity_rad_s: GroundVec3;
}

export interface FlightToGroundRequest {
  readonly schema_version: typeof FLIGHT_TO_GROUND_REQUEST_VERSION;
  readonly request_id: string;
  readonly unit_system: "SI";
  readonly surface: GroundSurfaceProfile;
  readonly last_separated_state: GroundContactState;
  readonly first_penetrating_state: GroundContactState;
  readonly ball_radius_m: number;
  readonly ball_mass_kg: number;
  readonly rotational_inertia_factor: number;
  readonly max_time_s: number;
  readonly output_interval_s: number;
  readonly max_events: number;
  readonly calibration: GroundCalibration;
  readonly provenance: GroundProvenance;
}

export interface GroundTrajectoryPoint extends GroundContactState {
  readonly phase: GroundPhase;
}

export interface GroundEvent {
  readonly sequence: number;
  readonly event_type: GroundEventType;
  readonly time_s: number;
  readonly frame: GroundFrame;
  readonly position_m: GroundVec3;
  readonly velocity_before_m_s: GroundVec3;
  readonly velocity_after_m_s: GroundVec3;
  readonly angular_velocity_before_rad_s: GroundVec3;
  readonly angular_velocity_after_rad_s: GroundVec3;
}

export interface GroundSummary {
  readonly carry_distance_m: number;
  readonly bounce_air_distance_m: number;
  readonly skid_distance_m: number;
  readonly roll_distance_m: number;
  readonly surface_path_distance_m: number;
  readonly total_distance_m: number;
  readonly final_downrange_m: number;
  readonly final_offline_m: number;
  readonly bounce_count: number;
}

export interface GroundTermination {
  readonly reason: GroundTerminationReason;
  readonly time_s: number;
  readonly completed: boolean;
}

export interface GroundWarning {
  readonly code: string;
  readonly severity: GroundWarningSeverity;
  readonly message: string;
}

export interface GroundUnavailableField {
  readonly field_id: GroundUnavailableFieldId;
  readonly reason: GroundUnavailableReason;
  readonly provenance: string;
}

export interface FlightToGroundResult {
  readonly schema_version: typeof FLIGHT_TO_GROUND_RESULT_VERSION;
  readonly request_id: string;
  readonly unit_system: "SI";
  readonly frame: GroundFrame;
  readonly surface_id: string;
  readonly model_id: string;
  readonly model_version: string;
  readonly status: GroundResultStatus;
  readonly trajectory: readonly GroundTrajectoryPoint[];
  readonly events: readonly GroundEvent[];
  readonly summary: GroundSummary | null;
  readonly termination: GroundTermination;
  readonly calibration: GroundCalibration;
  readonly warnings: readonly GroundWarning[];
  readonly unavailable_fields: readonly GroundUnavailableField[];
  readonly provenance: GroundProvenance;
}
