//! WASM binding for strict result validation without ground-physics execution.

use wasm_bindgen::prelude::*;

use super::{canonical_result_v1_json, parse_result_v1_json, GroundResultV1Error};

/// Validate and canonically re-emit an exact `flight-to-ground-result/v1` record.
#[wasm_bindgen(js_name = "validateFlightToGroundResultV1")]
pub fn validate_result_v1(payload: String) -> Result<String, JsValue> {
    let result = parse_result_v1_json(&payload).map_err(result_error)?;
    canonical_result_v1_json(&result).map_err(|_| JsValue::from_str("result_serialization"))
}

fn result_error(error: GroundResultV1Error) -> JsValue {
    JsValue::from_str(error.code())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_result() -> String {
        let fixture = include_str!(
            "../../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
        );
        let value: serde_json::Value = serde_json::from_str(fixture).unwrap();
        serde_json::to_string(&value["result"]).unwrap()
    }

    #[test]
    fn wasm_result_boundary_preserves_complete_shared_fixture() {
        let input = fixture_result();
        let output = validate_result_v1(input.clone()).unwrap();
        let expected = canonical_result_v1_json(&parse_result_v1_json(&input).unwrap()).unwrap();
        assert_eq!(output, expected);
    }
}
