use wasm_bindgen::prelude::*;

use super::wasm::parse_samples;
use super::{
    adapt_samples_to_request_v1, canonical_request_v1_json, parse_request_v1_json,
    GroundRequestV1Error,
};

/// Validate and losslessly re-emit an exact `flight-to-ground-request/v1` record.
#[wasm_bindgen(js_name = "validateFlightToGroundRequestV1")]
pub fn validate_request_v1(payload: String) -> Result<String, JsValue> {
    let request = parse_request_v1_json(&payload).map_err(request_error)?;
    canonical_request_v1_json(&request).map_err(|_| JsValue::from_str("request_serialization"))
}

/// Replace only the physical bracket in a complete strict v1 request.
#[wasm_bindgen(js_name = "adaptFlightSamplesToGroundRequestV1")]
pub fn adapt_request_v1(flat_samples: Vec<f64>, payload: String) -> Result<String, JsValue> {
    let samples =
        parse_samples(&flat_samples).ok_or_else(|| JsValue::from_str("malformed_samples"))?;
    let request = parse_request_v1_json(&payload).map_err(request_error)?;
    let adapted = adapt_samples_to_request_v1(&samples, request).map_err(request_error)?;
    canonical_request_v1_json(&adapted).map_err(|_| JsValue::from_str("request_serialization"))
}

fn request_error(error: GroundRequestV1Error) -> JsValue {
    JsValue::from_str(error.code())
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

    #[test]
    fn wasm_v1_boundary_preserves_complete_shared_fixture() {
        let input = fixture_request();
        let output = validate_request_v1(input.clone()).unwrap();
        let expected = canonical_request_v1_json(&parse_request_v1_json(&input).unwrap()).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn wasm_v1_adapter_preserves_surface_and_evidence() {
        let input = fixture_request();
        let samples = vec![
            5.19, 209.7, 0.024, -3.01, 31.0, -12.0, 1.5, 0.0, 260.0, -4.0, 5.2, 210.0, 0.019, -3.0,
            31.0, -12.0, 1.5, 0.0, 260.0, -4.0,
        ];
        let output = adapt_request_v1(samples, input.clone()).unwrap();
        let before = parse_request_v1_json(&input).unwrap();
        let after = parse_request_v1_json(&output).unwrap();
        assert_eq!(before.surface, after.surface);
        assert_eq!(before.calibration, after.calibration);
        assert_eq!(before.provenance, after.provenance);
        assert_eq!(before.schema_version, after.schema_version);
    }
}
