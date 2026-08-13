use math_primitives::types::Vector3;
use tools_core::ball_flight::{BallProperties, EnvironmentalConditions, LaunchConditions};
use tools_core::flight_ground::{
    canonical_request_v1_json, parse_request_v1_json, simulate_flight_to_ground,
    FlightGroundConfig, GroundRequestV1Error, LaunchGeometry, PlanarGround,
    TransferUnavailableReason,
};

const GOLDEN_REQUEST: &str = concat!(
    "{\"ball_mass_kg\":0.04593,\"ball_radius_m\":0.02135,\"calibration\":{",
    "\"calibration_id\":\"literature-default-2026-08\",\"confidence\":0.6,",
    "\"kind\":\"literature\",\"source\":\"documented literature basis\"},",
    "\"first_penetrating_state\":{\"angular_velocity_rad_s\":[0,260,-4],",
    "\"frame\":\"target_frame:x_downrange,y_up,z_right\",",
    "\"position_m\":[210,0.019,-3],\"time_s\":5.2,",
    "\"velocity_m_s\":[31,-12,1.5]},\"last_separated_state\":{",
    "\"angular_velocity_rad_s\":[0,260,-4],",
    "\"frame\":\"target_frame:x_downrange,y_up,z_right\",",
    "\"position_m\":[209.7,0.024,-3.01],\"time_s\":5.19,",
    "\"velocity_m_s\":[31,-12,1.5]},\"max_events\":64,\"max_time_s\":12,",
    "\"output_interval_s\":0.01,\"provenance\":{",
    "\"input_sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",",
    "\"producer\":\"tools.rate_of_closure\",\"producer_version\":\"1.0.0\",",
    "\"source_revision\":\"60ac5b46\"},\"request_id\":\"ground-run-001\",",
    "\"rotational_inertia_factor\":0.4,",
    "\"schema_version\":\"flight-to-ground-request/v1\",\"surface\":{",
    "\"compressibility_fraction\":0.2,\"compression_damping_fraction\":0.25,",
    "\"firmness_pa\":1200000,",
    "\"frame\":\"target_frame:x_downrange,y_up,z_right\",",
    "\"grass_height_m\":0.012,\"hardness_fraction\":0.7,\"height_m\":0,",
    "\"kinetic_friction\":0.28,\"moisture_fraction\":0.3,",
    "\"normal_restitution\":0.42,\"normal_unit\":[0,1,0],",
    "\"provider_id\":\"tools.planar-surface\",\"provider_version\":\"1.0.0\",",
    "\"rolling_resistance\":0.04,\"static_friction\":0.35,",
    "\"surface_id\":\"firm-fairway\",\"surface_velocity_m_s\":[0,0,0],",
    "\"turf_density_kg_m3\":180},\"unit_system\":\"SI\"}"
);

fn fixture_request() -> String {
    let fixture = include_str!(
        "../../../src/rate_of_closure/web/src/model/__fixtures__/flight_to_ground_golden_v1.json"
    )
    .replace("\\ud800", "rejected-surrogate");
    let value: serde_json::Value = serde_json::from_str(&fixture).unwrap();
    serde_json::to_string(&value["request"]).unwrap()
}

#[test]
fn request_emits_exact_cross_runtime_golden_tokens() {
    let request = parse_request_v1_json(&fixture_request()).unwrap();
    assert_eq!(canonical_request_v1_json(&request).unwrap(), GOLDEN_REQUEST);
}

#[test]
fn canonical_boundary_uses_half_away_fixed_point_and_integer_normalization() {
    let input = fixture_request()
        .replace("\"max_events\":64", "\"max_events\":64.0")
        .replace("\"confidence\":0.6", "\"confidence\":0.1234567890123")
        .replace(
            "\"angular_velocity_rad_s\":[0.0,260.0,-4.0]",
            "\"angular_velocity_rad_s\":[1.000000000005,-1.000000000005,1e-7]",
        );
    let request = parse_request_v1_json(&input).unwrap();
    let output = canonical_request_v1_json(&request).unwrap();
    assert!(output.contains("\"max_events\":64"));
    assert!(output.contains("\"confidence\":0.12345678901"));
    assert!(output.contains("\"angular_velocity_rad_s\":[1.00000000001,-1.00000000001,0.0000001]"));
    assert!(!output.contains("1e-7"));
}

#[test]
fn strict_boundary_rejects_duplicate_unknown_surrogate_and_unsafe_integer() {
    let input = fixture_request();
    let duplicate = input.replace(
        "\"request_id\":\"ground-run-001\"",
        "\"request_id\":\"ground-run-001\",\"request_id\":\"ground-run-001\"",
    );
    let unknown = format!("{},\"unknown\":true}}", &input[..input.len() - 1]);
    let surrogate = input.replace("ground-run-001", "\\ud800");
    let unsafe_integer = input.replace("\"max_events\":64", "\"max_events\":9007199254740992");
    for rejected in [duplicate, unknown, surrogate, unsafe_integer] {
        assert_eq!(
            parse_request_v1_json(&rejected),
            Err(GroundRequestV1Error::InvalidJson)
        );
    }
}

#[test]
fn canonical_emit_revalidates_the_post_quantization_contact_bracket() {
    let mut request = parse_request_v1_json(&fixture_request()).unwrap();
    request.last_separated_state.position_m[1] = request.ball_radius_m + 4.0e-12;
    assert_eq!(
        canonical_request_v1_json(&request),
        Err(GroundRequestV1Error::InvalidField("contact_bracket_gap"))
    );
}

fn no_contact_config(max_time: f64, dt: f64) -> FlightGroundConfig {
    FlightGroundConfig {
        max_time,
        dt,
        launch_geometry: LaunchGeometry::ground(),
        ground: PlanarGround::new(Vector3::new(0.0, -100.0, 0.0), Vector3::new(0.0, 1.0, 0.0))
            .unwrap(),
    }
}

#[test]
fn fixed_step_uses_an_exact_final_partial_step_without_exceeding_max_time() {
    let run = simulate_flight_to_ground(
        &BallProperties::default(),
        &EnvironmentalConditions::default(),
        &LaunchConditions::default(),
        &no_contact_config(0.015, 0.01),
    )
    .unwrap();
    assert_eq!(run.trajectory.last().unwrap().time, 0.015);
    assert!(run.trajectory.iter().all(|state| state.time <= 0.015));
}

#[test]
fn fixed_step_rejects_requests_over_the_step_budget() {
    let result = simulate_flight_to_ground(
        &BallProperties::default(),
        &EnvironmentalConditions::default(),
        &LaunchConditions::default(),
        &no_contact_config(1.000_001, 0.000_001),
    );
    assert_eq!(
        result,
        Err(TransferUnavailableReason::InvalidSimulationConfiguration)
    );
}
