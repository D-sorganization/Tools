use tools_core::flight_ground::{
    parse_ground_reference_execution_v1_json, parse_request_v1_json, run_ground_reference_v1,
    GroundPhaseV1, GroundTerminationReasonV1,
};

const FIXTURE: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
);

#[test]
fn stationary_holding_projection_reports_rest_before_the_time_horizon() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut execution_value = fixture["execution"].clone();
    execution_value["schema_version"] = fixture["execution_schema_version"].clone();
    let mut request = parse_request_v1_json(&fixture["request"].to_string()).unwrap();
    let mut execution =
        parse_ground_reference_execution_v1_json(&execution_value.to_string()).unwrap();
    let speed = 9.0e-10;
    let rolling_spin = -speed / request.ball_radius_m;
    request.surface.normal_restitution = 0.0;
    request.last_separated_state.velocity_m_s = [speed, -0.04, 0.0];
    request.first_penetrating_state.velocity_m_s = [speed, -0.04, 0.0];
    request.last_separated_state.angular_velocity_rad_s = [0.0, 0.0, rolling_spin];
    request.first_penetrating_state.angular_velocity_rad_s = [0.0, 0.0, rolling_spin];
    request.max_time_s = 0.01;
    request.output_interval_s = 0.001;
    execution.skid_roll_settings.integration_step_s = 0.01;

    let result = run_ground_reference_v1(&request, &execution, || false).unwrap();

    assert_eq!(result.termination.reason, GroundTerminationReasonV1::Rest);
    let final_point = result.trajectory.last().unwrap();
    assert_eq!(final_point.phase, GroundPhaseV1::Rest);
    assert_eq!(final_point.velocity_m_s, [0.0; 3]);
    assert_eq!(final_point.angular_velocity_rad_s, [0.0; 3]);
}
