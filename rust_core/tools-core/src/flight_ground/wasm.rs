use math_primitives::types::Vector3;
use wasm_bindgen::prelude::*;

use super::{
    qualify_ground_transfer, FlightState, GroundTransferOutcome, PlanarGround,
    TransferUnavailableReason,
};

const SAMPLE_WIDTH: usize = 10;

/// Request-frame ground-plane height for a ball radius and optional tee [m].
/// Pass zero tee height for ground-supported launch geometry.
#[wasm_bindgen(js_name = "requestGroundPlaneHeight")]
pub fn request_ground_plane_height(ball_radius: f64, tee_height: f64) -> Result<f64, JsValue> {
    if !ball_radius.is_finite() || ball_radius <= 0.0 {
        return Err(JsValue::from_str("ball radius must be finite and positive"));
    }
    if !tee_height.is_finite() || tee_height < 0.0 {
        return Err(JsValue::from_str(
            "tee height must be finite and non-negative",
        ));
    }
    Ok(-(ball_radius + tee_height))
}

/// WASM-friendly qualified ground-transfer result.
#[wasm_bindgen]
pub struct WasmGroundTransfer {
    status: u8,
    reason: Option<u8>,
    state: Option<FlightState>,
    bracket_times: Option<(f64, f64)>,
}

#[wasm_bindgen]
impl WasmGroundTransfer {
    /// 0=contact, 1=no crossing, 2=grazing, 3=unavailable.
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> u8 {
        self.status
    }

    #[wasm_bindgen(getter)]
    pub fn reason(&self) -> Option<u8> {
        self.reason
    }

    #[wasm_bindgen(js_name = "hasState", getter)]
    pub fn has_state(&self) -> bool {
        self.state.is_some()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> Option<f64> {
        self.state.map(|state| state.time)
    }

    #[wasm_bindgen(js_name = "positionX", getter)]
    pub fn position_x(&self) -> Option<f64> {
        self.state.map(|state| state.position.x)
    }

    #[wasm_bindgen(js_name = "positionY", getter)]
    pub fn position_y(&self) -> Option<f64> {
        self.state.map(|state| state.position.y)
    }

    #[wasm_bindgen(js_name = "positionZ", getter)]
    pub fn position_z(&self) -> Option<f64> {
        self.state.map(|state| state.position.z)
    }

    #[wasm_bindgen(js_name = "velocityX", getter)]
    pub fn velocity_x(&self) -> Option<f64> {
        self.state.map(|state| state.velocity.x)
    }

    #[wasm_bindgen(js_name = "velocityY", getter)]
    pub fn velocity_y(&self) -> Option<f64> {
        self.state.map(|state| state.velocity.y)
    }

    #[wasm_bindgen(js_name = "velocityZ", getter)]
    pub fn velocity_z(&self) -> Option<f64> {
        self.state.map(|state| state.velocity.z)
    }

    #[wasm_bindgen(js_name = "angularVelocityX", getter)]
    pub fn angular_velocity_x(&self) -> Option<f64> {
        self.state.map(|state| state.angular_velocity.x)
    }

    #[wasm_bindgen(js_name = "angularVelocityY", getter)]
    pub fn angular_velocity_y(&self) -> Option<f64> {
        self.state.map(|state| state.angular_velocity.y)
    }

    #[wasm_bindgen(js_name = "angularVelocityZ", getter)]
    pub fn angular_velocity_z(&self) -> Option<f64> {
        self.state.map(|state| state.angular_velocity.z)
    }

    #[wasm_bindgen(js_name = "hasBracket", getter)]
    pub fn has_bracket(&self) -> bool {
        self.bracket_times.is_some()
    }

    #[wasm_bindgen(js_name = "lastSeparatedTime", getter)]
    pub fn last_separated_time(&self) -> Option<f64> {
        self.bracket_times.map(|times| times.0)
    }

    #[wasm_bindgen(js_name = "firstPenetratingTime", getter)]
    pub fn first_penetrating_time(&self) -> Option<f64> {
        self.bracket_times.map(|times| times.1)
    }
}

/// Qualify flattened request-frame samples for horizontal-ground transfer.
///
/// Each sample contains time, position xyz, velocity xyz, and angular velocity
/// xyz. The wire format therefore preserves the full signed rotational vector.
#[wasm_bindgen(js_name = "qualifyHorizontalGroundTransfer")]
pub fn qualify_horizontal_ground_transfer(
    flat_samples: Vec<f64>,
    ball_radius: f64,
    ground_height: f64,
) -> WasmGroundTransfer {
    qualify_planar_ground_transfer(
        flat_samples,
        ball_radius,
        vec![0.0, ground_height, 0.0, 0.0, 1.0, 0.0],
    )
}

/// Qualify against a plane encoded as point xyz followed by outward normal xyz.
#[wasm_bindgen(js_name = "qualifyPlanarGroundTransfer")]
pub fn qualify_planar_ground_transfer(
    flat_samples: Vec<f64>,
    ball_radius: f64,
    plane: Vec<f64>,
) -> WasmGroundTransfer {
    let outcome = match (parse_samples(&flat_samples), parse_plane(&plane)) {
        (Some(samples), Ok(ground)) => qualify_ground_transfer(&samples, &ground, ball_radius),
        (None, _) => {
            GroundTransferOutcome::Unavailable(TransferUnavailableReason::MalformedSamples)
        }
        (_, Err(reason)) => GroundTransferOutcome::Unavailable(reason),
    };
    WasmGroundTransfer::from_outcome(outcome)
}

pub(super) fn parse_samples(values: &[f64]) -> Option<Vec<FlightState>> {
    if values.is_empty()
        || !values.chunks_exact(SAMPLE_WIDTH).remainder().is_empty()
        || values.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(
        values
            .chunks_exact(SAMPLE_WIDTH)
            .map(|row| {
                FlightState::new(
                    row[0],
                    Vector3::new(row[1], row[2], row[3]),
                    Vector3::new(row[4], row[5], row[6]),
                    Vector3::new(row[7], row[8], row[9]),
                )
            })
            .collect(),
    )
}

fn parse_plane(values: &[f64]) -> Result<PlanarGround, TransferUnavailableReason> {
    if values.len() != 6 || values.iter().any(|value| !value.is_finite()) {
        return Err(TransferUnavailableReason::InvalidGroundPlane);
    }
    PlanarGround::new(
        Vector3::new(values[0], values[1], values[2]),
        Vector3::new(values[3], values[4], values[5]),
    )
}

impl WasmGroundTransfer {
    fn from_outcome(outcome: GroundTransferOutcome) -> Self {
        match outcome {
            GroundTransferOutcome::Contact(event) => Self::new(
                0,
                Some(event.contact),
                Some((event.last_separated.time, event.first_penetrating.time)),
            ),
            GroundTransferOutcome::NoCrossing { last_state } => {
                Self::new(1, Some(last_state), None)
            }
            GroundTransferOutcome::Grazing { state } => Self::new(2, Some(state), None),
            GroundTransferOutcome::Unavailable(reason) => Self::unavailable(reason),
        }
    }

    fn new(status: u8, state: Option<FlightState>, bracket_times: Option<(f64, f64)>) -> Self {
        Self {
            status,
            reason: None,
            state,
            bracket_times,
        }
    }

    fn unavailable(reason: TransferUnavailableReason) -> Self {
        Self {
            status: 3,
            reason: Some(reason_code(reason)),
            state: None,
            bracket_times: None,
        }
    }
}

fn reason_code(reason: TransferUnavailableReason) -> u8 {
    match reason {
        TransferUnavailableReason::EmptyTrajectory => 1,
        TransferUnavailableReason::NonFiniteState => 2,
        TransferUnavailableReason::InvalidBallRadius => 3,
        TransferUnavailableReason::InvalidGroundPlane => 4,
        TransferUnavailableReason::InvalidLaunchGeometry => 5,
        TransferUnavailableReason::InvalidSimulationConfiguration => 6,
        TransferUnavailableReason::InvalidSpinAxis => 7,
        TransferUnavailableReason::MalformedSamples => 8,
        TransferUnavailableReason::UnqualifiedContactBracket => 9,
        TransferUnavailableReason::NoPhysicalContact => 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ground_height_distinguishes_ground_and_tee() {
        assert_eq!(request_ground_plane_height(0.02, 0.0).unwrap(), -0.02);
        assert_eq!(request_ground_plane_height(0.02, 0.04).unwrap(), -0.06);
    }

    #[test]
    fn wasm_flat_contract_preserves_signed_angular_velocity() {
        let samples = vec![
            0.0, 0.0, 0.03, 0.0, 1.0, -1.0, 0.0, 10.0, -20.0, 30.0, 0.1, 0.1, 0.01, 0.0, 1.0, -1.0,
            0.0, 8.0, -16.0, 24.0,
        ];
        let result = qualify_horizontal_ground_transfer(samples, 0.02, 0.0);
        assert_eq!(result.status, 0);
        assert_eq!(result.reason, None);
        assert!((result.state.unwrap().angular_velocity.y + 18.0).abs() < 1.0e-12);
        assert!(result.bracket_times.is_some());
    }

    #[test]
    fn malformed_wasm_samples_return_typed_unavailable() {
        let result = qualify_horizontal_ground_transfer(vec![0.0, 1.0], 0.02, 0.0);
        assert_eq!(result.status, 3);
        assert_eq!(result.reason, Some(8));
        assert!(result.state.is_none());
        assert!(result.bracket_times.is_none());
    }

    #[test]
    fn no_crossing_has_state_but_no_reason_or_bracket() {
        let samples = vec![
            0.0, 0.0, 0.03, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.04, 0.0, 1.0, 0.5, 0.0,
            0.0, 0.0, 0.0,
        ];
        let result = qualify_horizontal_ground_transfer(samples, 0.02, 0.0);
        assert_eq!(result.status, 1);
        assert_eq!(result.reason, None);
        assert!(result.state.is_some());
        assert!(result.bracket_times.is_none());
    }

    #[test]
    fn grazing_status_has_state_without_contact_bracket() {
        let samples = vec![
            0.0, 0.0, 0.03, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.02, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        let result = qualify_horizontal_ground_transfer(samples, 0.02, 0.0);
        assert_eq!(result.status, 2);
        assert_eq!(result.reason, None);
        assert!(result.state.is_some());
        assert!(result.bracket_times.is_none());
    }

    #[test]
    fn unavailable_reason_codes_are_stable() {
        let reasons = [
            TransferUnavailableReason::EmptyTrajectory,
            TransferUnavailableReason::NonFiniteState,
            TransferUnavailableReason::InvalidBallRadius,
            TransferUnavailableReason::InvalidGroundPlane,
            TransferUnavailableReason::InvalidLaunchGeometry,
            TransferUnavailableReason::InvalidSimulationConfiguration,
            TransferUnavailableReason::InvalidSpinAxis,
            TransferUnavailableReason::MalformedSamples,
            TransferUnavailableReason::UnqualifiedContactBracket,
            TransferUnavailableReason::NoPhysicalContact,
        ];
        let codes: Vec<u8> = reasons.into_iter().map(reason_code).collect();
        assert_eq!(codes, (1..=10).collect::<Vec<_>>());
    }

    #[test]
    fn tilted_unit_plane_is_qualified() {
        let unit = 2.0_f64.sqrt().recip();
        let before = 0.03 * unit;
        let after = 0.01 * unit;
        let velocity = -unit;
        let samples = vec![
            0.0, 0.0, before, before, 0.0, velocity, velocity, 1.0, -2.0, 3.0, 0.1, 0.0, after,
            after, 0.0, velocity, velocity, 1.0, -2.0, 3.0,
        ];
        let result =
            qualify_planar_ground_transfer(samples, 0.02, vec![0.0, 0.0, 0.0, 0.0, unit, unit]);
        assert_eq!(result.status, 0);
        assert!(result.bracket_times.is_some());
    }

    #[test]
    fn wasm_planar_boundary_rejects_non_unit_normal() {
        let result = qualify_planar_ground_transfer(
            vec![0.0, 0.0, 0.03, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            0.02,
            vec![0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        );
        assert_eq!(result.status, 3);
        assert_eq!(result.reason, Some(4));
    }
}
