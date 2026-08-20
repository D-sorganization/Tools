use sha2::{Digest, Sha256};

use tools_core::flight_ground::{
    canonical_result_v1_json, parse_ground_reference_execution_v1_json, parse_request_v1_json,
    parse_result_v1_json, run_ground_reference_v1, run_ground_reference_v1_json, GroundPhaseV1,
    GroundReferenceBoundaryErrorV1, GroundReferenceExecutionV1, GroundReferencePhaseV1,
    GroundReferenceRuntimeCodeV1, GroundResultStatusV1, GroundTerminationReasonV1,
};

const FIXTURE: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
);

fn fixture_parts() -> (String, String, String, String) {
    let value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut execution = value["execution"].clone();
    execution["schema_version"] = value["execution_schema_version"].clone();
    (
        value["request"].to_string(),
        execution.to_string(),
        value["result"].to_string(),
        value["result_sha256"].as_str().unwrap().to_owned(),
    )
}

fn parsed_fixture() -> (
    tools_core::flight_ground::FlightToGroundRequestV1,
    GroundReferenceExecutionV1,
) {
    let (request_json, execution_json, _, _) = fixture_parts();
    (
        parse_request_v1_json(&request_json).unwrap(),
        parse_ground_reference_execution_v1_json(&execution_json).unwrap(),
    )
}

#[test]
fn compiled_reference_matches_canonical_python_golden() {
    let (request_json, execution_json, expected_json, expected_digest) = fixture_parts();
    let request = parse_request_v1_json(&request_json).unwrap();
    let execution = parse_ground_reference_execution_v1_json(&execution_json).unwrap();
    let expected = parse_result_v1_json(&expected_json).unwrap();

    let actual = run_ground_reference_v1(&request, &execution, || false).unwrap();
    let actual_json = canonical_result_v1_json(&actual).unwrap();
    let expected_json = canonical_result_v1_json(&expected).unwrap();
    let digest = format!("{:x}", Sha256::digest(actual_json.as_bytes()));

    assert_eq!(actual_json, expected_json);
    assert_eq!(digest, expected_digest);
}

#[test]
fn compiled_reference_cancellation_is_typed_and_bounded() {
    let (request_json, execution_json, _, _) = fixture_parts();
    let request = parse_request_v1_json(&request_json).unwrap();
    let execution = parse_ground_reference_execution_v1_json(&execution_json).unwrap();
    let mut checks = 0_u64;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        checks == 2
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::Cancelled);
    assert_eq!(error.native_reason, "cancelled");
    assert_eq!(checks, 2);
}

#[test]
fn execution_contract_rejects_non_null_resolver() {
    let (_, execution_json, _, _) = fixture_parts();
    let mut value: serde_json::Value = serde_json::from_str(&execution_json).unwrap();
    value["resolver"] = serde_json::json!({"surface_id": "unsupported"});

    let error = parse_ground_reference_execution_v1_json(&value.to_string()).unwrap_err();

    assert_eq!(error.code(), "unsupported_resolver");
}

#[test]
fn compiled_reference_is_byte_deterministic() {
    let (request, execution) = parsed_fixture();
    let expected =
        canonical_result_v1_json(&run_ground_reference_v1(&request, &execution, || false).unwrap())
            .unwrap();

    for _ in 0..100 {
        let actual = run_ground_reference_v1(&request, &execution, || false).unwrap();
        assert_eq!(canonical_result_v1_json(&actual).unwrap(), expected);
    }
}

#[test]
fn compiled_reference_returns_valid_partial_time_limit() {
    let (mut request, execution) = parsed_fixture();
    request.max_time_s = 0.11;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();
    let canonical = canonical_result_v1_json(&result).unwrap();

    assert_eq!(result.status, GroundResultStatusV1::Partial);
    assert_eq!(
        result.termination.reason,
        GroundTerminationReasonV1::TimeLimit
    );
    assert!(canonical.contains("CENSORED_ENDPOINT"));
}

#[test]
fn compiled_reference_returns_valid_partial_event_limit() {
    let (mut request, execution) = parsed_fixture();
    request.max_events = 3;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();
    canonical_result_v1_json(&result).unwrap();

    assert_eq!(result.status, GroundResultStatusV1::Partial);
    assert_eq!(
        result.termination.reason,
        GroundTerminationReasonV1::EventLimit
    );
}

#[test]
fn compiled_reference_handles_tilted_plane_without_a_resolver() {
    let (mut request, execution) = parsed_fixture();
    let angle = 0.05_f64;
    let normal = [0.0, angle.cos(), angle.sin()];
    let contact = normal.map(|component| component * request.ball_radius_m);
    let incoming = [0.8, -0.1 * normal[1], -0.1 * normal[2]];
    request.surface.normal_unit = normal;
    request.surface.surface_velocity_m_s = [0.04, 0.0, 0.0];
    request.last_separated_state.position_m =
        std::array::from_fn(|index| contact[index] + 0.001 * normal[index]);
    request.first_penetrating_state.position_m =
        std::array::from_fn(|index| contact[index] - 0.001 * normal[index]);
    request.last_separated_state.velocity_m_s = incoming;
    request.first_penetrating_state.velocity_m_s = incoming;
    request.max_time_s = 0.5;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();
    canonical_result_v1_json(&result).unwrap();

    for point in result
        .trajectory
        .iter()
        .filter(|point| point.phase != GroundPhaseV1::Bounce)
    {
        let plane_distance = point
            .position_m
            .iter()
            .zip(normal)
            .map(|(value, component)| value * component)
            .sum::<f64>();
        assert!((plane_distance - request.ball_radius_m).abs() < 1.0e-8);
    }
}

#[test]
fn empty_event_limited_suffix_returns_coherent_partial_result() {
    let (mut request, execution) = parsed_fixture();
    request.max_events = 1;
    request.last_separated_state.velocity_m_s = [0.0, -0.04, 0.0];
    request.first_penetrating_state.velocity_m_s = [0.0, -0.04, 0.0];
    request.last_separated_state.angular_velocity_rad_s = [0.0; 3];
    request.first_penetrating_state.angular_velocity_rad_s = [0.0; 3];

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    assert_eq!(result.status, GroundResultStatusV1::Partial);
    assert_eq!(
        result.termination.reason,
        GroundTerminationReasonV1::EventLimit
    );
    assert!(!result.termination.completed);
    assert_eq!(result.events.len(), 1);
    assert_eq!(
        result.trajectory.last().unwrap().time_s,
        result.termination.time_s
    );
    let summary = result.summary.as_ref().unwrap();
    assert_eq!(summary.surface_path_distance_m, 0.0);
    canonical_result_v1_json(&result).unwrap();
}

#[test]
fn immediate_capture_reconstructs_the_first_impact_point() {
    let (mut request, execution) = parsed_fixture();
    let tangential_speed = 1.0;
    let rolling_spin = -tangential_speed / request.ball_radius_m;
    request.last_separated_state.velocity_m_s = [tangential_speed, -0.04, 0.0];
    request.first_penetrating_state.velocity_m_s = [tangential_speed, -0.04, 0.0];
    request.last_separated_state.angular_velocity_rad_s = [0.0, 0.0, rolling_spin];
    request.first_penetrating_state.angular_velocity_rad_s = [0.0, 0.0, rolling_spin];

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    assert_eq!(result.trajectory[0].phase, GroundPhaseV1::Impact);
    canonical_result_v1_json(&result).unwrap();
}

#[test]
fn json_boundary_distinguishes_request_execution_and_runtime_errors() {
    let (request_json, execution_json, _, _) = fixture_parts();
    assert!(matches!(
        run_ground_reference_v1_json("{}", None, || false).unwrap_err(),
        GroundReferenceBoundaryErrorV1::Request(_)
    ));
    let unsupported = execution_json.replace("\"resolver\":null", "\"resolver\":{}");
    assert!(matches!(
        run_ground_reference_v1_json(&request_json, Some(&unsupported), || false).unwrap_err(),
        GroundReferenceBoundaryErrorV1::Execution(_)
    ));
    let error =
        run_ground_reference_v1_json(&request_json, Some(&execution_json), || true).unwrap_err();
    assert!(error.is_cancelled());
    let canonical_error = error.payload();
    assert!(canonical_error.starts_with(
        "{\"code\":\"cancelled\",\"native_reason\":\"cancelled\",\"phase\":\"bounce\","
    ));
    let payload: serde_json::Value = serde_json::from_str(&canonical_error).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    assert_eq!(
        payload["schema_version"],
        "ground-reference-execution-error/v1"
    );
    assert_eq!(
        payload["request_fingerprint_sha256"],
        fixture["request_sha256"]
    );
}

#[test]
fn compiled_reference_cancels_inside_surface_loop() {
    let (request, execution) = parsed_fixture();
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        checks == 5
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::Cancelled);
    assert_eq!(error.phase, GroundReferencePhaseV1::SkidRoll);
    assert_eq!(checks, 5);
}

#[test]
fn compiled_reference_cancels_inside_bounce_grid_emission() {
    let (mut request, execution) = parsed_fixture();
    request.max_time_s = 0.1;
    request.output_interval_s = 0.000_01;
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        checks == 3
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::Cancelled);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(checks, 3);
}

#[test]
fn compiled_reference_cancels_inside_surface_grid_emission() {
    let (mut request, execution) = parsed_fixture();
    request.surface.normal_restitution = 0.0;
    request.max_time_s = 0.1;
    request.output_interval_s = 0.000_01;
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        checks == 3
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::Cancelled);
    assert_eq!(error.phase, GroundReferencePhaseV1::SkidRoll);
    assert_eq!(checks, 3);
}

#[test]
fn execution_contract_rejects_serialized_cancellation_and_duplicate_keys() {
    let (_, execution_json, _, _) = fixture_parts();
    let serialized = execution_json.replace("\"is_cancelled\":null", "\"is_cancelled\":true");
    assert_eq!(
        parse_ground_reference_execution_v1_json(&serialized)
            .unwrap_err()
            .code(),
        "serialized_cancellation"
    );
    let duplicate = execution_json.replacen(
        "{",
        "{\"schema_version\":\"ground-reference-execution/v1\",",
        1,
    );
    assert_eq!(
        parse_ground_reference_execution_v1_json(&duplicate)
            .unwrap_err()
            .code(),
        "invalid_json"
    );
}

#[test]
fn execution_contract_rejects_unqualified_numerics_and_model_identity() {
    let (_, execution_json, _, _) = fixture_parts();
    let base: serde_json::Value = serde_json::from_str(&execution_json).unwrap();
    let cases = [
        (
            "/bounce_settings/gravity_m_s2/1",
            serde_json::json!(-9.8),
            "gravity_m_s2",
        ),
        (
            "/skid_roll_settings/max_steps",
            serde_json::json!(0),
            "max_steps",
        ),
        (
            "/skid_roll_settings/model_version",
            serde_json::json!("2.0.0"),
            "skid_roll_model_identity",
        ),
    ];
    for (pointer, replacement, expected) in cases {
        let mut value = base.clone();
        *value.pointer_mut(pointer).unwrap() = replacement;
        assert_eq!(
            parse_ground_reference_execution_v1_json(&value.to_string())
                .unwrap_err()
                .code(),
            expected
        );
    }
}

#[test]
fn output_schedule_accepts_the_exact_endpoint_inclusive_boundary() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 200_000.0;
    request.output_interval_s = 1.0;
    execution.skid_roll_settings.integration_step_s = 0.2;
    execution.skid_roll_settings.max_steps = 1_000_000;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    canonical_result_v1_json(&result).unwrap();
}

#[test]
fn output_schedule_rejects_one_point_beyond_the_exact_boundary() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 200_001.0;
    request.output_interval_s = 1.0;
    execution.skid_roll_settings.integration_step_s = 0.200_001;
    execution.skid_roll_settings.max_steps = 1_000_000;

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "output_point_limit");
}

#[test]
fn pathological_output_interval_fails_before_callback_or_allocation() {
    let (mut request, execution) = parsed_fixture();
    request.max_time_s = 9_000_000_000_000_000.0;
    request.output_interval_s = 0.000_000_000_01;
    let mut calls = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        calls += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.native_reason, "output_point_limit");
    assert_eq!(calls, 0);
}

#[test]
fn unrepresentable_large_epoch_bounce_fails_before_physics() {
    let (mut request, execution) = parsed_fixture();
    request.last_separated_state.time_s = 9_000_000_000_000_000.0;
    request.first_penetrating_state.time_s = 9_000_000_000_000_002.0;
    request.max_time_s = 0.1;
    request.output_interval_s = 0.001;
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "time_resolution");
    assert_eq!(checks, 0);
}

#[test]
fn unrepresentable_large_epoch_immediate_capture_fails_before_physics() {
    let (mut request, mut execution) = parsed_fixture();
    request.last_separated_state.time_s = 9_000_000_000_000_000.0;
    request.first_penetrating_state.time_s = 9_000_000_000_000_002.0;
    request.surface.normal_restitution = 0.0;
    request.max_time_s = 0.1;
    request.output_interval_s = 0.001;
    execution.skid_roll_settings.integration_step_s = 0.01;
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "time_resolution");
    assert_eq!(checks, 0);
}

#[test]
fn absolute_time_overflow_fails_typed_before_physics() {
    let (mut request, execution) = parsed_fixture();
    request.last_separated_state.time_s = 8_000_000_000_000_000.0;
    request.first_penetrating_state.time_s = 8_000_000_000_000_002.0;
    request.max_time_s = 2_000_000_000_000_000.0;
    request.output_interval_s = 1_000_000_000_000_000.0;
    let mut checks = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        checks += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "time_resolution");
    assert_eq!(checks, 0);
}

fn assert_representable_large_epoch(restitution: f64) {
    let (mut request, mut execution) = parsed_fixture();
    request.last_separated_state.time_s = 1_000_000_000_000.0;
    request.first_penetrating_state.time_s = 1_000_000_000_002.0;
    request.surface.normal_restitution = restitution;
    request.output_interval_s = 0.00125;
    execution.skid_roll_settings.integration_step_s = 0.01;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    canonical_result_v1_json(&result).unwrap();
    assert!(result
        .trajectory
        .windows(2)
        .all(|pair| pair[0].time_s < pair[1].time_s));
}

#[test]
fn representable_trillion_second_bounce_completes_monotonically() {
    let (request, _) = parsed_fixture();
    assert_representable_large_epoch(request.surface.normal_restitution);
}

#[test]
fn representable_trillion_second_immediate_capture_completes_monotonically() {
    assert_representable_large_epoch(0.0);
}

#[test]
fn direct_native_request_uses_one_normalized_authority() {
    let (mut baseline, mut execution) = parsed_fixture();
    baseline.max_time_s = 0.1;
    baseline.output_interval_s = 0.01;
    execution.skid_roll_settings.integration_step_s = 0.01;
    execution.skid_roll_settings.max_steps = 100;
    let mut subcanonical = baseline.clone();
    subcanonical.max_time_s += 0.000_000_000_004;

    let expected = run_ground_reference_v1(&baseline, &execution, || false).unwrap();
    let actual = run_ground_reference_v1(&subcanonical, &execution, || false).unwrap();

    assert_eq!(
        canonical_result_v1_json(&actual).unwrap(),
        canonical_result_v1_json(&expected).unwrap()
    );
}
