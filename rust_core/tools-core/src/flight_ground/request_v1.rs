//! Exact `flight-to-ground-request/v1` wire boundary.

use serde::{Deserialize, Serialize};

use super::canonical_json::{
    canonical_json, deserialize_safe_u64, normalize_request, reject_unsafe_numbers,
    serialize_safe_u64, validate_raw_numbers,
};
use super::{
    qualify_ground_transfer, FlightState, GroundTransferOutcome, PlanarGround,
    TransferUnavailableReason, INCOMING_SPEED_TOLERANCE,
};
use math_primitives::types::Vector3;

pub const REQUEST_SCHEMA_VERSION: &str = "flight-to-ground-request/v1";
pub const TARGET_FRAME: &str = "target_frame:x_downrange,y_up,z_right";
pub const UNIT_SYSTEM: &str = "SI";
const MIN_POSITIVE: f64 = 1.0e-11;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundProvenanceV1 {
    pub producer: String,
    pub producer_version: String,
    pub source_revision: String,
    pub input_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundCalibrationV1 {
    pub calibration_id: String,
    pub kind: String,
    pub source: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundSurfaceV1 {
    pub surface_id: String,
    pub provider_id: String,
    pub provider_version: String,
    pub frame: String,
    pub height_m: f64,
    pub normal_unit: [f64; 3],
    pub surface_velocity_m_s: [f64; 3],
    pub normal_restitution: f64,
    pub static_friction: f64,
    pub kinetic_friction: f64,
    pub rolling_resistance: f64,
    pub firmness_pa: f64,
    pub hardness_fraction: f64,
    pub grass_height_m: f64,
    pub compressibility_fraction: f64,
    pub compression_damping_fraction: f64,
    pub turf_density_kg_m3: f64,
    pub moisture_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundContactStateV1 {
    pub time_s: f64,
    pub frame: String,
    pub position_m: [f64; 3],
    pub velocity_m_s: [f64; 3],
    pub angular_velocity_rad_s: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlightToGroundRequestV1 {
    pub schema_version: String,
    pub request_id: String,
    pub unit_system: String,
    pub surface: GroundSurfaceV1,
    pub last_separated_state: GroundContactStateV1,
    pub first_penetrating_state: GroundContactStateV1,
    pub ball_radius_m: f64,
    pub ball_mass_kg: f64,
    pub rotational_inertia_factor: f64,
    pub max_time_s: f64,
    pub output_interval_s: f64,
    #[serde(
        deserialize_with = "deserialize_safe_u64",
        serialize_with = "serialize_safe_u64"
    )]
    pub max_events: u64,
    pub calibration: GroundCalibrationV1,
    pub provenance: GroundProvenanceV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroundRequestV1Error {
    InvalidJson,
    InvalidSchema,
    InvalidField(&'static str),
    TransferUnavailable(TransferUnavailableReason),
}

impl GroundRequestV1Error {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidJson => "invalid_json",
            Self::InvalidSchema => "invalid_schema",
            Self::InvalidField(field) => field,
            Self::TransferUnavailable(reason) => reason.code(),
        }
    }
}

pub fn parse_request_v1_json(
    payload: &str,
) -> Result<FlightToGroundRequestV1, GroundRequestV1Error> {
    let mut request: FlightToGroundRequestV1 =
        serde_json::from_str(payload).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    let raw: serde_json::Value =
        serde_json::from_str(payload).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    reject_unsafe_numbers(&raw).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    validate_raw_numbers(&request).map_err(GroundRequestV1Error::InvalidField)?;
    normalize_request(&mut request).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    request.validate()?;
    Ok(request)
}

pub fn canonical_request_v1_json(
    request: &FlightToGroundRequestV1,
) -> Result<String, GroundRequestV1Error> {
    validate_raw_numbers(request).map_err(GroundRequestV1Error::InvalidField)?;
    let mut normalized = request.clone();
    normalize_request(&mut normalized).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    normalized.validate()?;
    let value = serde_json::to_value(normalized).map_err(|_| GroundRequestV1Error::InvalidJson)?;
    canonical_json(&value).map_err(|_| GroundRequestV1Error::InvalidJson)
}

pub fn adapt_samples_to_request_v1(
    samples: &[FlightState],
    mut request: FlightToGroundRequestV1,
) -> Result<FlightToGroundRequestV1, GroundRequestV1Error> {
    request.validate()?;
    let ground = request.surface.planar_ground()?;
    let outcome = qualify_ground_transfer(samples, &ground, request.ball_radius_m);
    let event = match outcome {
        GroundTransferOutcome::Contact(event) => event,
        GroundTransferOutcome::Unavailable(reason) => {
            return Err(GroundRequestV1Error::TransferUnavailable(reason));
        }
        GroundTransferOutcome::Grazing { .. } => {
            return Err(GroundRequestV1Error::TransferUnavailable(
                TransferUnavailableReason::UnqualifiedContactBracket,
            ));
        }
        GroundTransferOutcome::NoCrossing { .. } => {
            return Err(GroundRequestV1Error::TransferUnavailable(
                TransferUnavailableReason::NoPhysicalContact,
            ));
        }
    };
    request.last_separated_state = GroundContactStateV1::from(event.last_separated);
    request.first_penetrating_state = GroundContactStateV1::from(event.first_penetrating);
    request.validate()?;
    Ok(request)
}

impl FlightToGroundRequestV1 {
    pub fn validate(&self) -> Result<(), GroundRequestV1Error> {
        if self.schema_version != REQUEST_SCHEMA_VERSION || self.unit_system != UNIT_SYSTEM {
            return Err(GroundRequestV1Error::InvalidSchema);
        }
        validate_text(&self.request_id, "request_id")?;
        validate_positive(self.ball_radius_m, "ball_radius_m")?;
        validate_positive(self.ball_mass_kg, "ball_mass_kg")?;
        validate_positive(self.max_time_s, "max_time_s")?;
        validate_positive(self.output_interval_s, "output_interval_s")?;
        if self.output_interval_s > self.max_time_s
            || self.max_events == 0
            || self.max_events > 9_007_199_254_740_991
        {
            return Err(GroundRequestV1Error::InvalidField("solver_limits"));
        }
        if !in_range(self.rotational_inertia_factor, MIN_POSITIVE, 1.0) {
            return Err(GroundRequestV1Error::InvalidField(
                "rotational_inertia_factor",
            ));
        }
        self.surface.validate()?;
        self.calibration.validate()?;
        self.provenance.validate()?;
        self.validate_bracket()
    }

    fn validate_bracket(&self) -> Result<(), GroundRequestV1Error> {
        self.last_separated_state.validate()?;
        self.first_penetrating_state.validate()?;
        let before = &self.last_separated_state;
        let after = &self.first_penetrating_state;
        if after.time_s <= before.time_s {
            return Err(GroundRequestV1Error::InvalidField("contact_bracket_time"));
        }
        let ground = self.surface.planar_ground()?;
        let gaps = [
            before.gap(&ground, self.ball_radius_m),
            after.gap(&ground, self.ball_radius_m),
        ];
        let speeds = [before.normal_speed(&ground), after.normal_speed(&ground)];
        if gaps[0] <= 0.0 || gaps[1] > 0.0 {
            return Err(GroundRequestV1Error::InvalidField("contact_bracket_gap"));
        }
        if speeds
            .iter()
            .any(|speed| *speed >= -INCOMING_SPEED_TOLERANCE)
        {
            return Err(GroundRequestV1Error::InvalidField(
                "contact_bracket_velocity",
            ));
        }
        Ok(())
    }
}

impl GroundSurfaceV1 {
    fn validate(&self) -> Result<(), GroundRequestV1Error> {
        for value in [&self.surface_id, &self.provider_id, &self.provider_version] {
            validate_text(value, "surface_identity")?;
        }
        if self.frame != TARGET_FRAME || !self.height_m.is_finite() {
            return Err(GroundRequestV1Error::InvalidField("surface_frame"));
        }
        self.planar_ground()?;
        validate_bounded(self.normal_restitution, 1.0, "normal_restitution")?;
        validate_bounded(self.static_friction, 5.0, "static_friction")?;
        validate_bounded(self.kinetic_friction, 5.0, "kinetic_friction")?;
        if self.kinetic_friction > self.static_friction {
            return Err(GroundRequestV1Error::InvalidField("friction_order"));
        }
        validate_bounded(self.rolling_resistance, 1.0, "rolling_resistance")?;
        validate_positive(self.firmness_pa, "firmness_pa")?;
        validate_bounded(self.hardness_fraction, 1.0, "hardness_fraction")?;
        validate_nonnegative(self.grass_height_m, "grass_height_m")?;
        validate_bounded(
            self.compressibility_fraction,
            1.0,
            "compressibility_fraction",
        )?;
        validate_bounded(
            self.compression_damping_fraction,
            1.0,
            "compression_damping_fraction",
        )?;
        validate_nonnegative(self.turf_density_kg_m3, "turf_density_kg_m3")?;
        validate_bounded(self.moisture_fraction, 1.0, "moisture_fraction")
    }

    fn planar_ground(&self) -> Result<PlanarGround, GroundRequestV1Error> {
        PlanarGround::with_velocity(
            Vector3::new(0.0, self.height_m, 0.0),
            vector(self.normal_unit),
            vector(self.surface_velocity_m_s),
        )
        .map_err(|_| GroundRequestV1Error::InvalidField("surface_geometry"))
    }
}

impl GroundCalibrationV1 {
    fn validate(&self) -> Result<(), GroundRequestV1Error> {
        validate_text(&self.calibration_id, "calibration_id")?;
        validate_text(&self.source, "calibration_source")?;
        if !matches!(
            self.kind.as_str(),
            "measured" | "literature" | "estimated" | "unvalidated"
        ) {
            return Err(GroundRequestV1Error::InvalidField("calibration_kind"));
        }
        validate_bounded(self.confidence, 1.0, "calibration_confidence")
    }
}

impl GroundProvenanceV1 {
    fn validate(&self) -> Result<(), GroundRequestV1Error> {
        for value in [
            &self.producer,
            &self.producer_version,
            &self.source_revision,
        ] {
            validate_text(value, "provenance")?;
        }
        let valid_hash = self.input_sha256.len() == 64
            && self
                .input_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        if !valid_hash {
            return Err(GroundRequestV1Error::InvalidField("input_sha256"));
        }
        Ok(())
    }
}

impl GroundContactStateV1 {
    fn validate(&self) -> Result<(), GroundRequestV1Error> {
        if !self.time_s.is_finite() || self.time_s < 0.0 || self.frame != TARGET_FRAME {
            return Err(GroundRequestV1Error::InvalidField("contact_state"));
        }
        if [
            self.position_m,
            self.velocity_m_s,
            self.angular_velocity_rad_s,
        ]
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
        {
            return Err(GroundRequestV1Error::InvalidField("contact_state_vector"));
        }
        Ok(())
    }

    fn gap(&self, ground: &PlanarGround, radius: f64) -> f64 {
        ground.signed_gap(vector(self.position_m), radius)
    }

    fn normal_speed(&self, ground: &PlanarGround) -> f64 {
        ground.relative_normal_speed(vector(self.velocity_m_s))
    }
}

impl From<FlightState> for GroundContactStateV1 {
    fn from(state: FlightState) -> Self {
        Self {
            time_s: state.time,
            frame: TARGET_FRAME.to_owned(),
            position_m: array(state.position),
            velocity_m_s: array(state.velocity),
            angular_velocity_rad_s: array(state.angular_velocity),
        }
    }
}

fn validate_text(value: &str, field: &'static str) -> Result<(), GroundRequestV1Error> {
    if value.is_empty() || value.trim() != value {
        return Err(GroundRequestV1Error::InvalidField(field));
    }
    Ok(())
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), GroundRequestV1Error> {
    if !value.is_finite() || value < MIN_POSITIVE {
        return Err(GroundRequestV1Error::InvalidField(field));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, field: &'static str) -> Result<(), GroundRequestV1Error> {
    if !value.is_finite() || value < 0.0 {
        return Err(GroundRequestV1Error::InvalidField(field));
    }
    Ok(())
}

fn validate_bounded(
    value: f64,
    upper: f64,
    field: &'static str,
) -> Result<(), GroundRequestV1Error> {
    if !in_range(value, 0.0, upper) {
        return Err(GroundRequestV1Error::InvalidField(field));
    }
    Ok(())
}

fn in_range(value: f64, lower: f64, upper: f64) -> bool {
    value.is_finite() && value >= lower && value <= upper
}

fn vector(value: [f64; 3]) -> Vector3 {
    Vector3::new(value[0], value[1], value[2])
}

fn array(value: Vector3) -> [f64; 3] {
    [value.x, value.y, value.z]
}
