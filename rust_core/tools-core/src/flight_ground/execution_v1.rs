//! Strict numerical controls and typed failures for compiled ground execution.

use serde::{Deserialize, Serialize};

use super::canonical_json::reject_unsafe_numbers;
use super::strict_json::reject_duplicate_keys;

pub const EXECUTION_SCHEMA_VERSION: &str = "ground-reference-execution/v1";
const STANDARD_GRAVITY: [f64; 3] = [0.0, -9.80665, 0.0];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BounceExecutionSettingsV1 {
    pub gravity_m_s2: [f64; 3],
    pub capture_speed_m_s: f64,
    pub velocity_tolerance_m_s: f64,
    pub time_tolerance_s: f64,
    pub model_id: String,
    pub model_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkidRollExecutionSettingsV1 {
    pub gravity_m_s2: [f64; 3],
    pub integration_step_s: f64,
    pub max_steps: u64,
    pub velocity_tolerance_m_s: f64,
    pub angular_tolerance_rad_s: f64,
    pub slip_tolerance_m_s: f64,
    pub time_tolerance_s: f64,
    pub model_id: String,
    pub model_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundReferenceExecutionV1 {
    pub schema_version: String,
    pub bounce_settings: BounceExecutionSettingsV1,
    pub skid_roll_settings: SkidRollExecutionSettingsV1,
    pub resolver: Option<serde_json::Value>,
    pub is_cancelled: Option<serde_json::Value>,
}

impl Default for BounceExecutionSettingsV1 {
    fn default() -> Self {
        Self {
            gravity_m_s2: STANDARD_GRAVITY,
            capture_speed_m_s: 0.05,
            velocity_tolerance_m_s: 1.0e-12,
            time_tolerance_s: 1.0e-12,
            model_id: "tools-ground-impact-bounce".to_owned(),
            model_version: "1.0.0".to_owned(),
        }
    }
}

impl Default for SkidRollExecutionSettingsV1 {
    fn default() -> Self {
        Self {
            gravity_m_s2: STANDARD_GRAVITY,
            integration_step_s: 0.001,
            max_steps: 200_000,
            velocity_tolerance_m_s: 1.0e-9,
            angular_tolerance_rad_s: 1.0e-9,
            slip_tolerance_m_s: 1.0e-9,
            time_tolerance_s: 1.0e-12,
            model_id: "tools-ground-skid-roll".to_owned(),
            model_version: "1.0.0".to_owned(),
        }
    }
}

impl Default for GroundReferenceExecutionV1 {
    fn default() -> Self {
        Self {
            schema_version: EXECUTION_SCHEMA_VERSION.to_owned(),
            bounce_settings: BounceExecutionSettingsV1::default(),
            skid_roll_settings: SkidRollExecutionSettingsV1::default(),
            resolver: None,
            is_cancelled: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundReferenceExecutionV1Error {
    InvalidJson,
    InvalidSchema,
    InvalidField(&'static str),
    UnsupportedResolver,
    SerializedCancellation,
}

impl GroundReferenceExecutionV1Error {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidSchema => "invalid_schema",
            Self::InvalidField(field) => field,
            Self::UnsupportedResolver => "unsupported_resolver",
            Self::SerializedCancellation => "serialized_cancellation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundReferencePhaseV1 {
    Bounce,
    SkidRoll,
    Composition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundReferenceRuntimeCodeV1 {
    Cancelled,
    ExecutionFailure,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundReferenceRuntimeErrorV1 {
    pub schema_version: String,
    pub code: GroundReferenceRuntimeCodeV1,
    pub phase: GroundReferencePhaseV1,
    pub native_reason: String,
    pub request_fingerprint_sha256: String,
}

impl GroundReferenceRuntimeErrorV1 {
    pub(super) fn new(
        code: GroundReferenceRuntimeCodeV1,
        phase: GroundReferencePhaseV1,
        native_reason: &str,
        request_fingerprint_sha256: String,
    ) -> Self {
        Self {
            schema_version: "ground-reference-execution-error/v1".to_owned(),
            code,
            phase,
            native_reason: native_reason.to_owned(),
            request_fingerprint_sha256,
        }
    }
}

pub fn parse_ground_reference_execution_v1_json(
    payload: &str,
) -> Result<GroundReferenceExecutionV1, GroundReferenceExecutionV1Error> {
    reject_duplicate_keys(payload).map_err(|_| GroundReferenceExecutionV1Error::InvalidJson)?;
    let raw: serde_json::Value =
        serde_json::from_str(payload).map_err(|_| GroundReferenceExecutionV1Error::InvalidJson)?;
    reject_unsafe_numbers(&raw).map_err(|_| GroundReferenceExecutionV1Error::InvalidJson)?;
    let execution: GroundReferenceExecutionV1 =
        serde_json::from_value(raw).map_err(|_| GroundReferenceExecutionV1Error::InvalidJson)?;
    execution.validate()?;
    Ok(execution)
}

impl GroundReferenceExecutionV1 {
    pub fn validate(&self) -> Result<(), GroundReferenceExecutionV1Error> {
        if self.schema_version != EXECUTION_SCHEMA_VERSION {
            return Err(GroundReferenceExecutionV1Error::InvalidSchema);
        }
        if self.resolver.is_some() {
            return Err(GroundReferenceExecutionV1Error::UnsupportedResolver);
        }
        if self.is_cancelled.is_some() {
            return Err(GroundReferenceExecutionV1Error::SerializedCancellation);
        }
        self.bounce_settings.validate()?;
        self.skid_roll_settings.validate()
    }
}

impl BounceExecutionSettingsV1 {
    fn validate(&self) -> Result<(), GroundReferenceExecutionV1Error> {
        validate_gravity(self.gravity_m_s2)?;
        validate_positive(self.capture_speed_m_s, "capture_speed_m_s")?;
        validate_positive(self.velocity_tolerance_m_s, "bounce_velocity_tolerance")?;
        validate_positive(self.time_tolerance_s, "bounce_time_tolerance")?;
        validate_identity(
            &self.model_id,
            &self.model_version,
            "tools-ground-impact-bounce",
            "bounce_model_identity",
        )
    }
}

impl SkidRollExecutionSettingsV1 {
    fn validate(&self) -> Result<(), GroundReferenceExecutionV1Error> {
        validate_gravity(self.gravity_m_s2)?;
        validate_positive(self.integration_step_s, "integration_step_s")?;
        if self.max_steps == 0 || self.max_steps > 9_007_199_254_740_991 {
            return Err(GroundReferenceExecutionV1Error::InvalidField("max_steps"));
        }
        validate_positive(self.velocity_tolerance_m_s, "skid_velocity_tolerance")?;
        validate_positive(self.angular_tolerance_rad_s, "angular_tolerance")?;
        validate_positive(self.slip_tolerance_m_s, "slip_tolerance")?;
        validate_positive(self.time_tolerance_s, "skid_time_tolerance")?;
        validate_identity(
            &self.model_id,
            &self.model_version,
            "tools-ground-skid-roll",
            "skid_roll_model_identity",
        )
    }
}

fn validate_gravity(value: [f64; 3]) -> Result<(), GroundReferenceExecutionV1Error> {
    if value != STANDARD_GRAVITY {
        return Err(GroundReferenceExecutionV1Error::InvalidField(
            "gravity_m_s2",
        ));
    }
    Ok(())
}

fn validate_positive(
    value: f64,
    field: &'static str,
) -> Result<(), GroundReferenceExecutionV1Error> {
    if !value.is_finite() || value <= 0.0 {
        return Err(GroundReferenceExecutionV1Error::InvalidField(field));
    }
    Ok(())
}

fn validate_identity(
    model_id: &str,
    version: &str,
    expected_id: &str,
    field: &'static str,
) -> Result<(), GroundReferenceExecutionV1Error> {
    if model_id != expected_id || version != "1.0.0" {
        return Err(GroundReferenceExecutionV1Error::InvalidField(field));
    }
    Ok(())
}
