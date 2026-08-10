//! Qualified handoff from airborne ball flight to planar-ground dynamics.
//!
//! The canonical target frame is right handed: +x downrange, +y up, and +z
//! right of target. Contact occurs when the sphere surface reaches the plane,
//! not when the ball center reaches plane height.

use math_primitives::types::Vector3;
use serde::{Deserialize, Serialize};

use crate::ball_flight::BallProperties;

mod simulation;
pub use simulation::simulate_flight_to_ground;
mod canonical_json;
mod request_v1;
pub use request_v1::{
    adapt_samples_to_request_v1, canonical_request_v1_json, parse_request_v1_json,
    FlightToGroundRequestV1, GroundRequestV1Error,
};
mod execution_v1;
mod result_geometry;
mod result_v1;
mod result_validation;
mod strict_json;
pub use execution_v1::{
    parse_ground_reference_execution_v1_json, BounceExecutionSettingsV1,
    GroundReferenceExecutionV1, GroundReferenceExecutionV1Error, GroundReferencePhaseV1,
    GroundReferenceRuntimeCodeV1, GroundReferenceRuntimeErrorV1, SkidRollExecutionSettingsV1,
};
mod bounce_runtime;
mod impact_runtime;
mod reference_boundary;
mod reference_runtime;
mod resource_limits;
mod runtime_math;
mod surface_dynamics;
mod surface_events;
mod surface_runtime;
pub use reference_boundary::{run_ground_reference_v1_json, GroundReferenceBoundaryErrorV1};
pub use reference_runtime::{
    canonical_ground_reference_runtime_error_v1_json, run_ground_reference_v1,
};
pub use result_v1::{
    canonical_result_v1_json, parse_result_v1_json, FlightToGroundResultV1, GroundPhaseV1,
    GroundResultStatusV1, GroundResultV1Error, GroundTerminationReasonV1,
};
#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "python")]
pub mod python_reference;
#[cfg(feature = "wasm")]
mod wasm;
#[cfg(feature = "wasm")]
mod wasm_reference;
#[cfg(feature = "wasm")]
mod wasm_request;
#[cfg(feature = "wasm")]
mod wasm_result;

const UNIT_TOLERANCE: f64 = 1.0e-9;
const GRAZING_GAP_TOLERANCE: f64 = 1.0e-9;
const INCOMING_SPEED_TOLERANCE: f64 = 1.0e-12;

/// Physical support under the ball at launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchSupport {
    Ground,
    Tee,
}

/// Launch-center placement relative to a ground plane.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LaunchGeometry {
    pub support: LaunchSupport,
    tee_height: f64,
}

impl LaunchGeometry {
    #[must_use]
    pub const fn ground() -> Self {
        Self {
            support: LaunchSupport::Ground,
            tee_height: 0.0,
        }
    }

    /// Construct tee geometry with a non-negative tee height [m].
    pub fn tee(tee_height: f64) -> Result<Self, TransferUnavailableReason> {
        if !tee_height.is_finite() || tee_height < 0.0 {
            return Err(TransferUnavailableReason::InvalidLaunchGeometry);
        }
        Ok(Self {
            support: LaunchSupport::Tee,
            tee_height,
        })
    }

    #[must_use]
    pub fn initial_center(&self, ball: &BallProperties, _ground: &PlanarGround) -> Vector3 {
        Vector3::new(0.0, ball.radius() + self.tee_height, 0.0)
    }

    /// Ground translated into request coordinates where the launch center is zero.
    #[must_use]
    pub fn request_ground(&self, ball: &BallProperties, ground: &PlanarGround) -> PlanarGround {
        PlanarGround {
            point: ground.point - self.initial_center(ball, ground),
            normal: ground.normal,
            surface_velocity: ground.surface_velocity,
        }
    }

    fn is_valid(&self) -> bool {
        self.tee_height.is_finite()
            && self.tee_height >= 0.0
            && (self.support == LaunchSupport::Tee || self.tee_height == 0.0)
    }
}

/// Infinite planar ground surface with a unit normal directed out of the ground.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlanarGround {
    point: Vector3,
    normal: Vector3,
    surface_velocity: Vector3,
}

impl PlanarGround {
    pub fn new(point: Vector3, normal: Vector3) -> Result<Self, TransferUnavailableReason> {
        Self::with_velocity(point, normal, Vector3::zero())
    }

    pub fn with_velocity(
        point: Vector3,
        normal: Vector3,
        surface_velocity: Vector3,
    ) -> Result<Self, TransferUnavailableReason> {
        let unit_error = (normal.magnitude() - 1.0).abs();
        let normal_speed = surface_velocity.dot(&normal).abs();
        if !is_finite_vector(point)
            || !is_finite_vector(normal)
            || !is_finite_vector(surface_velocity)
            || unit_error > UNIT_TOLERANCE
            || normal.y <= 0.0
            || normal_speed > UNIT_TOLERANCE
        {
            return Err(TransferUnavailableReason::InvalidGroundPlane);
        }
        Ok(Self {
            point,
            normal,
            surface_velocity,
        })
    }

    #[must_use]
    pub fn horizontal(height: f64) -> Self {
        assert!(height.is_finite(), "ground height must be finite");
        Self {
            point: Vector3::new(0.0, height, 0.0),
            normal: Vector3::new(0.0, 1.0, 0.0),
            surface_velocity: Vector3::zero(),
        }
    }

    /// Signed sphere-to-plane clearance [m].
    #[must_use]
    pub fn signed_gap(&self, center: Vector3, radius: f64) -> f64 {
        (center - self.point).dot(&self.normal) - radius
    }

    #[must_use]
    pub const fn point(&self) -> Vector3 {
        self.point
    }

    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }

    #[must_use]
    pub const fn surface_velocity(&self) -> Vector3 {
        self.surface_velocity
    }

    #[must_use]
    fn relative_normal_speed(&self, velocity: Vector3) -> f64 {
        (velocity - self.surface_velocity).dot(&self.normal)
    }

    fn is_valid(&self) -> bool {
        is_finite_vector(self.point)
            && is_finite_vector(self.normal)
            && is_finite_vector(self.surface_velocity)
            && (self.normal.magnitude() - 1.0).abs() <= UNIT_TOLERANCE
            && self.normal.y > 0.0
            && self.surface_velocity.dot(&self.normal).abs() <= UNIT_TOLERANCE
    }
}

/// Complete translational and signed rotational state in the target frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(name = "FlightGroundState"))]
pub struct FlightState {
    pub time: f64,
    pub position: Vector3,
    pub velocity: Vector3,
    pub angular_velocity: Vector3,
}

impl FlightState {
    #[must_use]
    pub const fn new(
        time: f64,
        position: Vector3,
        velocity: Vector3,
        angular_velocity: Vector3,
    ) -> Self {
        Self {
            time,
            position,
            velocity,
            angular_velocity,
        }
    }

    fn is_finite(&self) -> bool {
        self.time.is_finite()
            && is_finite_vector(self.position)
            && is_finite_vector(self.velocity)
            && is_finite_vector(self.angular_velocity)
    }
}

/// Bracketing samples and their interpolated first-contact state.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GroundTransferEvent {
    pub last_separated: FlightState,
    pub first_penetrating: FlightState,
    pub contact: FlightState,
}

/// A runtime reason that prevents a physically qualified transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferUnavailableReason {
    EmptyTrajectory,
    NonFiniteState,
    InvalidBallRadius,
    InvalidGroundPlane,
    InvalidLaunchGeometry,
    InvalidSimulationConfiguration,
    InvalidSpinAxis,
    MalformedSamples,
    NoPhysicalContact,
    UnqualifiedContactBracket,
}

impl TransferUnavailableReason {
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::EmptyTrajectory => "empty_trajectory",
            Self::NonFiniteState => "non_finite_state",
            Self::InvalidBallRadius => "invalid_ball_radius",
            Self::InvalidGroundPlane => "invalid_ground_plane",
            Self::InvalidLaunchGeometry => "invalid_launch_geometry",
            Self::InvalidSimulationConfiguration => "invalid_simulation_configuration",
            Self::InvalidSpinAxis => "invalid_spin_axis",
            Self::MalformedSamples => "malformed_samples",
            Self::NoPhysicalContact => "no_physical_contact",
            Self::UnqualifiedContactBracket => "unqualified_contact_bracket",
        }
    }
}

/// Exhaustive result of searching a flight trajectory for ground transfer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GroundTransferOutcome {
    Contact(GroundTransferEvent),
    NoCrossing { last_state: FlightState },
    Grazing { state: FlightState },
    Unavailable(TransferUnavailableReason),
}

/// Parameters for a flight run that terminates at qualified sphere contact.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FlightGroundConfig {
    pub max_time: f64,
    pub dt: f64,
    pub launch_geometry: LaunchGeometry,
    pub ground: PlanarGround,
}

/// Flight samples and the corresponding typed transfer outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FlightGroundRun {
    pub trajectory: Vec<FlightState>,
    pub outcome: GroundTransferOutcome,
}

/// Convert a vector from legacy (+x forward, +y left, +z up) coordinates.
#[must_use]
pub fn legacy_to_target(vector: Vector3) -> Vector3 {
    Vector3::new(vector.x, vector.z, -vector.y)
}

/// Convert a complete state into the canonical target frame.
#[must_use]
pub fn legacy_state_to_target(state: FlightState) -> FlightState {
    FlightState::new(
        state.time,
        legacy_to_target(state.position),
        legacy_to_target(state.velocity),
        legacy_to_target(state.angular_velocity),
    )
}

/// Locate first physical sphere/plane contact using a separated/penetrating bracket.
#[must_use]
pub fn qualify_ground_transfer(
    samples: &[FlightState],
    ground: &PlanarGround,
    ball_radius: f64,
) -> GroundTransferOutcome {
    if samples.is_empty() {
        return GroundTransferOutcome::Unavailable(TransferUnavailableReason::EmptyTrajectory);
    }
    if samples.len() < 2 {
        return GroundTransferOutcome::Unavailable(TransferUnavailableReason::MalformedSamples);
    }
    if !ball_radius.is_finite() || ball_radius <= 0.0 {
        return GroundTransferOutcome::Unavailable(TransferUnavailableReason::InvalidBallRadius);
    }
    if !ground.is_valid() {
        return GroundTransferOutcome::Unavailable(TransferUnavailableReason::InvalidGroundPlane);
    }
    if samples
        .iter()
        .any(|sample| !sample.time.is_finite() || sample.time < 0.0)
        || samples.windows(2).any(|pair| pair[1].time <= pair[0].time)
    {
        return GroundTransferOutcome::Unavailable(TransferUnavailableReason::MalformedSamples);
    }
    let mut scan = ContactScan::default();
    for &sample in samples {
        if !sample.is_finite() {
            return GroundTransferOutcome::Unavailable(TransferUnavailableReason::NonFiniteState);
        }
        let gap = ground.signed_gap(sample.position, ball_radius);
        if let Some(outcome) = classify_sample(scan, sample, gap, ground) {
            return outcome;
        }
        if gap > 0.0 {
            scan.separated = Some(sample);
            scan.observed_clearance |= gap > GRAZING_GAP_TOLERANCE;
        }
    }
    GroundTransferOutcome::NoCrossing {
        last_state: *samples.last().expect("non-empty checked above"),
    }
}

#[derive(Clone, Copy, Default)]
struct ContactScan {
    separated: Option<FlightState>,
    observed_clearance: bool,
}

fn classify_sample(
    scan: ContactScan,
    sample: FlightState,
    gap: f64,
    ground: &PlanarGround,
) -> Option<GroundTransferOutcome> {
    let before = scan.separated?;
    let before_normal_speed = ground.relative_normal_speed(before.velocity);
    let normal_speed = ground.relative_normal_speed(sample.velocity);
    if scan.observed_clearance
        && gap.abs() <= GRAZING_GAP_TOLERANCE
        && normal_speed >= -INCOMING_SPEED_TOLERANCE
    {
        return Some(GroundTransferOutcome::Grazing { state: sample });
    }
    if gap <= 0.0 {
        if before_normal_speed >= -INCOMING_SPEED_TOLERANCE
            || normal_speed >= -INCOMING_SPEED_TOLERANCE
        {
            return Some(GroundTransferOutcome::Unavailable(
                TransferUnavailableReason::UnqualifiedContactBracket,
            ));
        }
        return Some(GroundTransferOutcome::Contact(interpolate_contact(
            before, sample, ground, gap,
        )));
    }
    None
}

fn interpolate_contact(
    before: FlightState,
    after: FlightState,
    ground: &PlanarGround,
    after_gap: f64,
) -> GroundTransferEvent {
    let radius = ground.signed_gap(after.position, 0.0) - after_gap;
    let before_gap = ground.signed_gap(before.position, radius);
    let alpha = (before_gap / (before_gap - after_gap)).clamp(0.0, 1.0);
    GroundTransferEvent {
        last_separated: before,
        first_penetrating: after,
        contact: interpolate_state(before, after, alpha),
    }
}

fn interpolate_state(before: FlightState, after: FlightState, alpha: f64) -> FlightState {
    FlightState::new(
        before.time + alpha * (after.time - before.time),
        before.position + (after.position - before.position) * alpha,
        before.velocity + (after.velocity - before.velocity) * alpha,
        before.angular_velocity + (after.angular_velocity - before.angular_velocity) * alpha,
    )
}

fn is_finite_vector(vector: Vector3) -> bool {
    vector.x.is_finite() && vector.y.is_finite() && vector.z.is_finite()
}
