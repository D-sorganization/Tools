//! Exact `flight-to-ground-result/v1` wire boundary.

use serde::{Deserialize, Serialize};

use super::canonical_json::{
    canonical_json, deserialize_safe_u64, normalize_f64, reject_unsafe_numbers, serialize_safe_u64,
};
use super::request_v1::{GroundCalibrationV1, GroundProvenanceV1};
use super::result_validation::validate_result;
use super::strict_json::reject_duplicate_keys;

pub const RESULT_SCHEMA_VERSION: &str = "flight-to-ground-result/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundPhaseV1 {
    Impact,
    Bounce,
    Skid,
    Roll,
    Rest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundEventTypeV1 {
    FirstContact,
    Bounce,
    SkidToRoll,
    Rest,
    LeftSurface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundResultStatusV1 {
    Complete,
    Partial,
    Failed,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundTerminationReasonV1 {
    Rest,
    TimeLimit,
    EventLimit,
    LeftSurface,
    NumericalFailure,
    UnavailableInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundWarningSeverityV1 {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundUnavailableFieldIdV1 {
    TerminalAngularVelocityRadS,
    PhysicalContactBracket,
    SurfaceProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundUnavailableReasonV1 {
    SourceDoesNotPropagate,
    NoPhysicalContact,
    UnsupportedSurface,
    SourceOutOfBounds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundTrajectoryPointV1 {
    pub time_s: f64,
    pub frame: String,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    pub angular_velocity_rad_s: [f64; 3],
    pub phase: GroundPhaseV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundEventV1 {
    #[serde(
        deserialize_with = "deserialize_safe_u64",
        serialize_with = "serialize_safe_u64"
    )]
    pub sequence: u64,
    pub event_type: GroundEventTypeV1,
    pub time_s: f64,
    pub frame: String,
    pub position_m: [f64; 3],
    pub velocity_before_m_s: [f64; 3],
    pub velocity_after_m_s: [f64; 3],
    pub angular_velocity_before_rad_s: [f64; 3],
    pub angular_velocity_after_rad_s: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundSummaryV1 {
    pub carry_distance_m: f64,
    pub bounce_air_distance_m: f64,
    pub skid_distance_m: f64,
    pub roll_distance_m: f64,
    pub surface_path_distance_m: f64,
    pub total_distance_m: f64,
    pub final_downrange_m: f64,
    pub final_offline_m: f64,
    #[serde(
        deserialize_with = "deserialize_safe_u64",
        serialize_with = "serialize_safe_u64"
    )]
    pub bounce_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundTerminationV1 {
    pub reason: GroundTerminationReasonV1,
    pub time_s: f64,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundWarningV1 {
    pub code: String,
    pub message: String,
    pub severity: GroundWarningSeverityV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundUnavailableFieldV1 {
    pub field_id: GroundUnavailableFieldIdV1,
    pub provenance: String,
    pub reason: GroundUnavailableReasonV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlightToGroundResultV1 {
    pub schema_version: String,
    pub request_id: String,
    pub surface_id: String,
    pub frame: String,
    pub model_id: String,
    pub model_version: String,
    pub status: GroundResultStatusV1,
    pub trajectory: Vec<GroundTrajectoryPointV1>,
    pub events: Vec<GroundEventV1>,
    pub summary: Option<GroundSummaryV1>,
    pub termination: GroundTerminationV1,
    pub calibration: GroundCalibrationV1,
    pub warnings: Vec<GroundWarningV1>,
    pub unavailable_fields: Vec<GroundUnavailableFieldV1>,
    pub provenance: GroundProvenanceV1,
    pub unit_system: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundResultV1Error {
    InvalidJson,
    InvalidSchema,
    InvalidField(&'static str),
    StatusPayload,
}

impl GroundResultV1Error {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidSchema => "invalid_schema",
            Self::InvalidField(field) => field,
            Self::StatusPayload => "status_payload",
        }
    }
}

pub fn parse_result_v1_json(payload: &str) -> Result<FlightToGroundResultV1, GroundResultV1Error> {
    reject_duplicate_keys(payload).map_err(|_| GroundResultV1Error::InvalidJson)?;
    let raw: serde_json::Value =
        serde_json::from_str(payload).map_err(|_| GroundResultV1Error::InvalidJson)?;
    reject_unsafe_numbers(&raw).map_err(|_| GroundResultV1Error::InvalidJson)?;
    let mut result: FlightToGroundResultV1 =
        serde_json::from_value(raw).map_err(|_| GroundResultV1Error::InvalidJson)?;
    validate_result(&result)?;
    normalize_result(&mut result).map_err(|_| GroundResultV1Error::InvalidJson)?;
    validate_result(&result)?;
    Ok(result)
}

pub fn canonical_result_v1_json(
    result: &FlightToGroundResultV1,
) -> Result<String, GroundResultV1Error> {
    validate_result(result)?;
    let mut normalized = result.clone();
    normalize_result(&mut normalized).map_err(|_| GroundResultV1Error::InvalidJson)?;
    validate_result(&normalized)?;
    let value = serde_json::to_value(normalized).map_err(|_| GroundResultV1Error::InvalidJson)?;
    reject_unsafe_numbers(&value).map_err(|_| GroundResultV1Error::InvalidJson)?;
    canonical_json(&value).map_err(|_| GroundResultV1Error::InvalidJson)
}

fn normalize_result(result: &mut FlightToGroundResultV1) -> Result<(), ()> {
    for point in &mut result.trajectory {
        normalize_scalar(&mut point.time_s)?;
        normalize_vector(&mut point.position_m)?;
        normalize_vector(&mut point.velocity_m_s)?;
        normalize_vector(&mut point.angular_velocity_rad_s)?;
    }
    for event in &mut result.events {
        normalize_scalar(&mut event.time_s)?;
        normalize_vector(&mut event.position_m)?;
        normalize_vector(&mut event.velocity_before_m_s)?;
        normalize_vector(&mut event.velocity_after_m_s)?;
        normalize_vector(&mut event.angular_velocity_before_rad_s)?;
        normalize_vector(&mut event.angular_velocity_after_rad_s)?;
    }
    if let Some(summary) = &mut result.summary {
        for value in [
            &mut summary.carry_distance_m,
            &mut summary.bounce_air_distance_m,
            &mut summary.skid_distance_m,
            &mut summary.roll_distance_m,
            &mut summary.surface_path_distance_m,
            &mut summary.total_distance_m,
            &mut summary.final_downrange_m,
            &mut summary.final_offline_m,
        ] {
            normalize_scalar(value)?;
        }
    }
    normalize_scalar(&mut result.termination.time_s)?;
    normalize_scalar(&mut result.calibration.confidence)
}

fn normalize_vector(values: &mut [f64; 3]) -> Result<(), ()> {
    values.iter_mut().try_for_each(normalize_scalar)
}

fn normalize_scalar(value: &mut f64) -> Result<(), ()> {
    *value = normalize_f64(*value)?;
    Ok(())
}
