use serde_json::Value;
use sha2::{Digest, Sha256};
use tools_core::flight_ground::{canonical_result_v1_json, parse_result_v1_json};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
    ))
    .expect("valid shared fixture")
}

fn fixture_result_json() -> String {
    serde_json::to_string(&fixture()["result"]).expect("serializable fixture result")
}

fn sha256(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn canonical_result_matches_python_golden_digest() {
    let result = parse_result_v1_json(&fixture_result_json()).expect("valid Python result");
    let canonical = canonical_result_v1_json(&result).expect("canonical Rust result");
    assert_eq!(sha256(&canonical), fixture()["result_sha256"]);
}

#[test]
fn strict_result_rejects_unknown_fields_and_wrong_schema() {
    let mut unknown: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    unknown["unexpected"] = Value::Bool(true);
    assert_eq!(
        parse_result_v1_json(&unknown.to_string())
            .expect_err("unknown field must fail")
            .code(),
        "invalid_json"
    );

    let mut wrong_schema: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_schema["schema_version"] = Value::String("flight-to-ground-result/v2".into());
    assert_eq!(
        parse_result_v1_json(&wrong_schema.to_string())
            .expect_err("wrong schema must fail")
            .code(),
        "invalid_schema"
    );

    let duplicate = fixture_result_json().replacen(
        "\"request_id\":",
        "\"request_id\":\"duplicate\",\"request_id\":",
        1,
    );
    assert_eq!(
        parse_result_v1_json(&duplicate)
            .expect_err("duplicate object key must fail")
            .code(),
        "invalid_json"
    );
}

#[test]
fn strict_result_rejects_incoherent_status_and_unsafe_integer() {
    let mut incoherent: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    incoherent["status"] = Value::String("failed".into());
    assert_eq!(
        parse_result_v1_json(&incoherent.to_string())
            .expect_err("failed result cannot carry output")
            .code(),
        "status_payload"
    );

    let mut unsafe_integer: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    unsafe_integer["summary"]["bounce_count"] = Value::from(9_007_199_254_740_992_u64);
    assert_eq!(
        parse_result_v1_json(&unsafe_integer.to_string())
            .expect_err("unsafe integer must fail")
            .code(),
        "invalid_json"
    );
}

#[test]
fn strict_result_rejects_broken_trajectory_and_event_invariants() {
    let mut wrong_frame: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_frame["trajectory"][0]["frame"] = Value::String("world".into());
    assert!(parse_result_v1_json(&wrong_frame.to_string()).is_err());

    let mut unordered: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    unordered["trajectory"][1]["time_s"] = unordered["trajectory"][0]["time_s"].clone();
    assert!(parse_result_v1_json(&unordered.to_string()).is_err());

    let mut wrong_sequence: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_sequence["events"][1]["sequence"] = Value::from(7);
    assert!(parse_result_v1_json(&wrong_sequence.to_string()).is_err());

    let mut moving_rest: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    let last = moving_rest["trajectory"].as_array().unwrap().len() - 1;
    moving_rest["trajectory"][last]["velocity_m_s"][0] = Value::from(0.01);
    assert!(parse_result_v1_json(&moving_rest.to_string()).is_err());
}

#[test]
fn strict_result_rejects_broken_summary_and_terminal_state() {
    let mut wrong_summary: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_summary["summary"]["total_distance_m"] = Value::from(999.0);
    assert!(parse_result_v1_json(&wrong_summary.to_string()).is_err());

    let mut wrong_bounces: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_bounces["summary"]["bounce_count"] = Value::from(0);
    assert!(parse_result_v1_json(&wrong_bounces.to_string()).is_err());

    let mut wrong_terminal_time: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_terminal_time["termination"]["time_s"] = Value::from(0.0);
    assert!(parse_result_v1_json(&wrong_terminal_time.to_string()).is_err());

    let mut wrong_completion: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    wrong_completion["termination"]["completed"] = Value::Bool(false);
    assert!(parse_result_v1_json(&wrong_completion.to_string()).is_err());

    let mut tiny_negative: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    tiny_negative["summary"]["roll_distance_m"] = Value::from(-1.0e-12);
    assert!(parse_result_v1_json(&tiny_negative.to_string()).is_err());
}

#[test]
fn strict_result_rejects_invalid_evidence_and_duplicate_unavailable_fields() {
    let mut bad_hash: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    bad_hash["provenance"]["input_sha256"] = Value::String("ABC".into());
    assert!(parse_result_v1_json(&bad_hash.to_string()).is_err());

    let mut unavailable: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    unavailable["status"] = Value::String("unavailable".into());
    unavailable["trajectory"] = Value::Array(Vec::new());
    unavailable["events"] = Value::Array(Vec::new());
    unavailable["summary"] = Value::Null;
    unavailable["termination"] = serde_json::json!({
        "reason": "unavailable_input", "time_s": 0, "completed": false
    });
    let field = serde_json::json!({
        "field_id": "surface_profile",
        "reason": "unsupported_surface",
        "provenance": "fixture"
    });
    unavailable["unavailable_fields"] = serde_json::json!([field.clone(), field]);
    assert!(parse_result_v1_json(&unavailable.to_string()).is_err());
}

#[test]
fn uppercase_provenance_digest_is_accepted_and_canonicalized_to_lowercase() {
    let mut uppercase: Value = serde_json::from_str(&fixture_result_json()).unwrap();
    let expected = uppercase["provenance"]["input_sha256"]
        .as_str()
        .unwrap()
        .to_owned();
    uppercase["provenance"]["input_sha256"] = Value::String(expected.to_ascii_uppercase());

    let parsed = parse_result_v1_json(&uppercase.to_string())
        .expect("Python and TypeScript accept uppercase hexadecimal input");
    assert_eq!(parsed.provenance.input_sha256, expected);
    let canonical = canonical_result_v1_json(&parsed).expect("canonical result");
    let emitted: Value = serde_json::from_str(&canonical).unwrap();
    assert_eq!(emitted["provenance"]["input_sha256"], expected);
}
