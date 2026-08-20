use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::ball_flight::{BallProperties, EnvironmentalConditions, LaunchConditions};

use super::{
    adapt_samples_to_request_v1, canonical_request_v1_json, canonical_result_v1_json,
    parse_request_v1_json, parse_result_v1_json, simulate_flight_to_ground, FlightGroundConfig,
    FlightGroundRun, FlightState, GroundTransferOutcome, LaunchGeometry, PlanarGround,
};
use math_primitives::types::Vector3;

/// Validated planar-ground definition for Python callers.
#[pyclass(name = "FlightGroundPlane")]
#[derive(Clone)]
pub struct PyFlightGroundPlane {
    ground: PlanarGround,
}

#[pymethods]
impl PyFlightGroundPlane {
    #[new]
    fn new(point: Vector3, outward_normal: Vector3) -> PyResult<Self> {
        let ground = PlanarGround::new(point, outward_normal)
            .map_err(|reason| PyValueError::new_err(reason.code()))?;
        Ok(Self { ground })
    }

    #[staticmethod]
    fn horizontal(height: f64) -> PyResult<Self> {
        if !height.is_finite() {
            return Err(PyValueError::new_err("ground height must be finite"));
        }
        Ok(Self {
            ground: PlanarGround::horizontal(height),
        })
    }
}

/// Flight-ground request with launch center as the request-frame origin.
#[pyclass(name = "FlightGroundRequest")]
#[derive(Clone)]
pub struct PyFlightGroundRequest {
    max_time: f64,
    dt: f64,
    plane: PyFlightGroundPlane,
    tee_height: Option<f64>,
}

#[pymethods]
impl PyFlightGroundRequest {
    #[new]
    #[pyo3(signature = (max_time, dt, plane, tee_height=None))]
    fn new(max_time: f64, dt: f64, plane: PyFlightGroundPlane, tee_height: Option<f64>) -> Self {
        Self {
            max_time,
            dt,
            plane,
            tee_height,
        }
    }
}

impl PyFlightGroundRequest {
    fn core_config(&self) -> PyResult<FlightGroundConfig> {
        let launch_geometry = match self.tee_height {
            Some(height) => LaunchGeometry::tee(height)
                .map_err(|reason| PyValueError::new_err(reason.code()))?,
            None => LaunchGeometry::ground(),
        };
        Ok(FlightGroundConfig {
            max_time: self.max_time,
            dt: self.dt,
            launch_geometry,
            ground: self.plane.ground,
        })
    }
}

/// Typed Python result retaining full translational and rotational state.
#[pyclass(name = "FlightGroundResult")]
pub struct PyFlightGroundResult {
    run: FlightGroundRun,
}

#[pymethods]
impl PyFlightGroundResult {
    #[getter]
    fn status(&self) -> &'static str {
        match self.run.outcome {
            GroundTransferOutcome::Contact(_) => "contact",
            GroundTransferOutcome::NoCrossing { .. } => "no_crossing",
            GroundTransferOutcome::Grazing { .. } => "grazing",
            GroundTransferOutcome::Unavailable(_) => "unavailable",
        }
    }

    #[getter]
    fn reason(&self) -> Option<&'static str> {
        match self.run.outcome {
            GroundTransferOutcome::Unavailable(reason) => Some(reason.code()),
            _ => None,
        }
    }

    #[getter]
    fn trajectory(&self) -> Vec<FlightState> {
        self.run.trajectory.clone()
    }

    #[getter]
    fn contact(&self) -> Option<FlightState> {
        match self.run.outcome {
            GroundTransferOutcome::Contact(event) => Some(event.contact),
            GroundTransferOutcome::Grazing { state } => Some(state),
            _ => None,
        }
    }

    #[getter]
    fn last_separated(&self) -> Option<FlightState> {
        match self.run.outcome {
            GroundTransferOutcome::Contact(event) => Some(event.last_separated),
            _ => None,
        }
    }

    #[getter]
    fn first_penetrating(&self) -> Option<FlightState> {
        match self.run.outcome {
            GroundTransferOutcome::Contact(event) => Some(event.first_penetrating),
            _ => None,
        }
    }
}

#[pymethods]
impl FlightState {
    #[getter]
    fn time(&self) -> f64 {
        self.time
    }

    #[getter]
    fn position(&self) -> Vector3 {
        self.position
    }

    #[getter]
    fn velocity(&self) -> Vector3 {
        self.velocity
    }

    #[getter]
    fn angular_velocity(&self) -> Vector3 {
        self.angular_velocity
    }
}

/// Run the strict full-state transfer path through the native Python wheel.
#[pyfunction(name = "simulate_flight_to_ground_full_state")]
pub fn py_simulate_flight_to_ground(
    ball: BallProperties,
    env: EnvironmentalConditions,
    launch: LaunchConditions,
    request: PyFlightGroundRequest,
) -> PyResult<PyFlightGroundResult> {
    let config = request.core_config()?;
    let run = simulate_flight_to_ground(&ball, &env, &launch, &config)
        .map_err(|reason| PyValueError::new_err(reason.code()))?;
    Ok(PyFlightGroundResult { run })
}

/// Validate and losslessly re-emit the complete strict v1 request record.
#[pyfunction(name = "validate_flight_to_ground_request_v1")]
pub fn py_validate_request_v1(payload: String) -> PyResult<String> {
    let request =
        parse_request_v1_json(&payload).map_err(|error| PyValueError::new_err(error.code()))?;
    canonical_request_v1_json(&request).map_err(|_| PyValueError::new_err("request_serialization"))
}

/// Validate and canonically re-emit an exact `flight-to-ground-result/v1` record.
#[pyfunction(name = "validate_flight_to_ground_result_v1")]
pub fn py_validate_result_v1(payload: String) -> PyResult<String> {
    let result =
        parse_result_v1_json(&payload).map_err(|error| PyValueError::new_err(error.code()))?;
    canonical_result_v1_json(&result).map_err(|_| PyValueError::new_err("result_serialization"))
}

/// Replace only the contact bracket while preserving every v1 context field.
#[pyfunction(name = "adapt_flight_samples_to_ground_request_v1")]
pub fn py_adapt_request_v1(samples: Vec<FlightState>, payload: String) -> PyResult<String> {
    let request =
        parse_request_v1_json(&payload).map_err(|error| PyValueError::new_err(error.code()))?;
    let adapted = adapt_samples_to_request_v1(&samples, request)
        .map_err(|error| PyValueError::new_err(error.code()))?;
    canonical_request_v1_json(&adapted).map_err(|_| PyValueError::new_err("request_serialization"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_request() -> String {
        let fixture = include_str!(
            "../../../../src/rate_of_closure/web/src/model/__fixtures__/flight_to_ground_golden_v1.json"
        )
        .replace("\\ud800", "rejected-surrogate");
        let value: serde_json::Value = serde_json::from_str(&fixture).unwrap();
        serde_json::to_string(&value["request"]).unwrap()
    }

    fn fixture_result() -> String {
        let fixture = include_str!(
            "../../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
        );
        let value: serde_json::Value = serde_json::from_str(fixture).unwrap();
        serde_json::to_string(&value["result"]).unwrap()
    }

    #[test]
    fn request_builds_tee_geometry() {
        let request = PyFlightGroundRequest::new(
            4.0,
            0.01,
            PyFlightGroundPlane::horizontal(0.0).unwrap(),
            Some(0.04),
        );
        let config = request.core_config().unwrap();
        assert_eq!(config.launch_geometry, LaunchGeometry::tee(0.04).unwrap());
    }

    #[test]
    fn bound_entry_point_returns_signed_full_state() {
        let request = PyFlightGroundRequest::new(
            4.0,
            0.01,
            PyFlightGroundPlane::horizontal(0.0).unwrap(),
            None,
        );
        let result = py_simulate_flight_to_ground(
            BallProperties::default(),
            EnvironmentalConditions::default(),
            LaunchConditions::default(),
            request,
        )
        .unwrap();
        assert_eq!(result.status(), "contact");
        assert!(result.reason().is_none());
        assert!(result.trajectory()[0].angular_velocity.y < 0.0);
        assert!(result.contact().is_some());
    }

    #[test]
    fn python_v1_boundary_preserves_complete_shared_fixture() {
        let input = fixture_request();
        let output = py_validate_request_v1(input.clone()).unwrap();
        let expected = canonical_request_v1_json(&parse_request_v1_json(&input).unwrap()).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn python_result_boundary_preserves_complete_shared_fixture() {
        let input = fixture_result();
        let output = py_validate_result_v1(input.clone()).unwrap();
        let expected = canonical_result_v1_json(&parse_result_v1_json(&input).unwrap()).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn python_result_boundary_canonicalizes_uppercase_digest() {
        let mut value: serde_json::Value = serde_json::from_str(&fixture_result()).unwrap();
        let digest = value["provenance"]["input_sha256"]
            .as_str()
            .unwrap()
            .to_owned();
        value["provenance"]["input_sha256"] =
            serde_json::Value::String(digest.to_ascii_uppercase());
        let output = py_validate_result_v1(value.to_string()).unwrap();
        let emitted: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(emitted["provenance"]["input_sha256"], digest);
    }
}
