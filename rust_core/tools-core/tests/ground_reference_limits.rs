use tools_core::flight_ground::{
    parse_ground_reference_execution_v1_json, parse_request_v1_json, run_ground_reference_v1,
    GroundReferenceExecutionV1, GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1,
};

const FIXTURE: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
);

fn parsed_fixture() -> (
    tools_core::flight_ground::FlightToGroundRequestV1,
    GroundReferenceExecutionV1,
) {
    let value: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut execution = value["execution"].clone();
    execution["schema_version"] = value["execution_schema_version"].clone();
    (
        parse_request_v1_json(&value["request"].to_string()).unwrap(),
        parse_ground_reference_execution_v1_json(&execution.to_string()).unwrap(),
    )
}

#[test]
fn untrusted_step_limit_cannot_expand_the_absolute_output_cap() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 200_001.0;
    request.output_interval_s = 1.0;
    execution.skid_roll_settings.max_steps = 1_000_000;
    let mut calls = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        calls += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "output_point_limit");
    assert_eq!(calls, 0);
}

#[test]
fn absolute_output_cap_accepts_its_exact_endpoint_inclusive_boundary() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 200_000.0;
    request.output_interval_s = 1.0;
    execution.skid_roll_settings.integration_step_s = 0.2;
    execution.skid_roll_settings.max_steps = 1_000_000;

    run_ground_reference_v1(&request, &execution, || false).unwrap();
}

#[test]
fn integration_step_budget_has_an_independent_absolute_preflight_cap() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 1.0;
    request.output_interval_s = 1.0;
    execution.skid_roll_settings.integration_step_s = 0.000_000_000_01;
    execution.skid_roll_settings.max_steps = 1_000_001;
    let mut calls = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        calls += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::SkidRoll);
    assert_eq!(error.native_reason, "integration_step_limit");
    assert_eq!(calls, 0);
}

#[test]
fn output_points_are_not_limited_by_the_integration_step_allowance() {
    let (mut request, mut execution) = parsed_fixture();
    request.max_time_s = 0.01;
    request.output_interval_s = 0.000_01;
    execution.skid_roll_settings.integration_step_s = 0.01;
    execution.skid_roll_settings.max_steps = 3;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    assert!(!result.trajectory.is_empty());
}

#[test]
fn admitted_small_step_budget_preserves_runtime_step_limit() {
    let (mut request, mut execution) = parsed_fixture();
    request.surface.normal_restitution = 0.0;
    request.max_time_s = 0.1;
    request.output_interval_s = 0.01;
    execution.skid_roll_settings.integration_step_s = 0.001;
    execution.skid_roll_settings.max_steps = 1;

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::SkidRoll);
    assert_eq!(error.native_reason, "step_limit");
}

#[test]
fn declared_event_budget_cannot_expand_absolute_event_work() {
    let (mut request, execution) = parsed_fixture();
    request.max_events = 10_001;
    let mut calls = 0;

    let error = run_ground_reference_v1(&request, &execution, || {
        calls += 1;
        false
    })
    .unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "event_count_limit");
    assert_eq!(calls, 0);
}

#[test]
fn surface_derived_state_overflow_returns_typed_failure() {
    let (mut request, execution) = parsed_fixture();
    request.surface.normal_restitution = 0.0;
    request.last_separated_state.position_m[0] = 9_000_000_000_000_000.0;
    request.first_penetrating_state.position_m[0] = 9_000_000_000_000_000.0;
    request.last_separated_state.velocity_m_s[0] = 9_000_000_000_000_000.0;
    request.first_penetrating_state.velocity_m_s[0] = 9_000_000_000_000_000.0;
    request.max_time_s = 0.01;
    request.output_interval_s = 0.001;

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::NumericalFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::SkidRoll);
    assert_eq!(error.native_reason, "numeric_range");
}

#[test]
fn bounce_derived_state_overflow_returns_typed_failure() {
    let (mut request, execution) = parsed_fixture();
    request.ball_radius_m = 1_000_000.0;
    request.surface.normal_restitution = 1.0;
    request.surface.static_friction = 5.0;
    request.surface.kinetic_friction = 5.0;
    request.last_separated_state.position_m[1] = 1_000_001.0;
    request.first_penetrating_state.position_m[1] = 999_999.0;
    for state in [
        &mut request.last_separated_state,
        &mut request.first_penetrating_state,
    ] {
        state.velocity_m_s = [0.0, -9_000_000_000_000_000.0, 0.0];
        state.angular_velocity_rad_s = [0.0, 0.0, 9_000_000_000_000_000.0];
    }

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::NumericalFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "numeric_range");
}

#[test]
fn composition_derived_distance_overflow_returns_typed_failure() {
    let (mut request, execution) = parsed_fixture();
    for state in [
        &mut request.last_separated_state,
        &mut request.first_penetrating_state,
    ] {
        state.position_m[0] = 6_500_000_000_000_000.0;
        state.position_m[2] = 6_500_000_000_000_000.0;
    }

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::NumericalFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Composition);
    assert_eq!(error.native_reason, "numeric_range");
}

#[test]
fn nonzero_restitution_event_limit_remains_a_bounce_failure() {
    let (mut request, execution) = parsed_fixture();
    request.max_events = 1;

    let error = run_ground_reference_v1(&request, &execution, || false).unwrap_err();

    assert_eq!(error.code, GroundReferenceRuntimeCodeV1::ExecutionFailure);
    assert_eq!(error.phase, GroundReferencePhaseV1::Bounce);
    assert_eq!(error.native_reason, "event_limit");
}
