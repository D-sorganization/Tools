use math_primitives::types::Vector3;
use tools_core::ball_flight::{BallProperties, EnvironmentalConditions, LaunchConditions};
use tools_core::flight_ground::{
    adapt_samples_to_request_v1, legacy_to_target, parse_request_v1_json, qualify_ground_transfer,
    simulate_flight_to_ground, FlightGroundConfig, FlightState, GroundTransferOutcome,
    LaunchGeometry, LaunchSupport, PlanarGround, TransferUnavailableReason,
};

const EPSILON: f64 = 1.0e-9;

fn state(time: f64, height: f64, vertical_speed: f64, spin: Vector3) -> FlightState {
    FlightState::new(
        time,
        Vector3::new(time, height, 0.0),
        Vector3::new(1.0, vertical_speed, 0.0),
        spin,
    )
}

#[test]
fn legacy_vectors_transform_into_the_target_frame() {
    let transformed = legacy_to_target(Vector3::new(12.0, 3.0, 7.0));
    assert_eq!(transformed, Vector3::new(12.0, 7.0, -3.0));
}

#[test]
fn launch_geometry_places_the_sphere_on_ground_or_tee() {
    let ball = BallProperties::default();
    let ground = PlanarGround::horizontal(0.0);
    let ground_center = LaunchGeometry::ground().initial_center(&ball, &ground);
    let tee_center = LaunchGeometry::tee(0.0381)
        .expect("valid tee")
        .initial_center(&ball, &ground);

    assert!((ground.signed_gap(ground_center, ball.radius())).abs() < EPSILON);
    assert!((ground.signed_gap(tee_center, ball.radius()) - 0.0381).abs() < EPSILON);

    let ground_request = LaunchGeometry::ground().request_ground(&ball, &ground);
    let tee_request = LaunchGeometry::tee(0.0381)
        .unwrap()
        .request_ground(&ball, &ground);
    assert!((ground_request.point().y + ball.radius()).abs() < EPSILON);
    assert!((tee_request.point().y + ball.radius() + 0.0381).abs() < EPSILON);
}

#[test]
fn planar_ground_rejects_a_non_upward_normal() {
    let downward = PlanarGround::new(Vector3::zero(), Vector3::new(0.0, -1.0, 0.0));
    let vertical = PlanarGround::new(Vector3::zero(), Vector3::new(1.0, 0.0, 0.0));
    assert!(downward.is_err());
    assert!(vertical.is_err());
}

#[test]
fn planar_ground_rejects_non_unit_normals_instead_of_normalizing() {
    let result = PlanarGround::new(Vector3::zero(), Vector3::new(0.0, 2.0, 0.0));
    assert_eq!(result, Err(TransferUnavailableReason::InvalidGroundPlane));
}

#[test]
fn request_geometry_preserves_terrain_elevation_and_uses_vertical_tee_height() {
    let ball = BallProperties::default();
    let normal = Vector3::new(0.0, 0.8, 0.6);
    for elevation in [-0.15, 0.2] {
        let terrain = PlanarGround::new(Vector3::new(0.0, elevation, 0.0), normal).unwrap();
        let tee = LaunchGeometry::tee(0.03).unwrap();
        let request_ground = tee.request_ground(&ball, &terrain);
        assert!((request_ground.point().y - (elevation - ball.radius() - 0.03)).abs() < EPSILON);
        assert_eq!(
            tee.initial_center(&ball, &terrain),
            Vector3::new(0.0, ball.radius() + 0.03, 0.0)
        );
        let contact_center = request_ground.point() + normal * ball.radius();
        assert!(
            request_ground
                .signed_gap(contact_center, ball.radius())
                .abs()
                < EPSILON
        );
    }
}

#[test]
fn first_sphere_contact_is_bracketed_and_interpolated() {
    let samples = [
        state(1.0, 0.03, -2.0, Vector3::new(10.0, -20.0, 30.0)),
        state(1.2, 0.01, -2.0, Vector3::new(8.0, -16.0, 24.0)),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);

    let GroundTransferOutcome::Contact(event) = outcome else {
        panic!("expected a contact event");
    };
    assert_eq!(event.last_separated, samples[0]);
    assert_eq!(event.first_penetrating, samples[1]);
    assert!((event.contact.time - 1.1).abs() < EPSILON);
    assert!((event.contact.position.y - 0.02).abs() < EPSILON);
    assert_eq!(
        event.contact.angular_velocity,
        Vector3::new(9.0, -18.0, 27.0)
    );
}

#[test]
fn separated_trajectory_without_contact_is_typed_no_crossing() {
    let samples = [
        state(0.0, 1.0, 2.0, Vector3::zero()),
        state(0.1, 1.1, 1.0, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
    assert!(matches!(outcome, GroundTransferOutcome::NoCrossing { .. }));
}

#[test]
fn positive_subnanometre_gap_is_still_strictly_separated() {
    let radius = 0.02;
    let samples = [
        state(0.0, radius + 5.0e-10, -1.0, Vector3::zero()),
        state(0.1, radius, -1.0, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), radius);
    assert!(matches!(outcome, GroundTransferOutcome::Contact(_)));
}

#[test]
fn unordered_or_duplicate_sample_times_are_typed_malformed() {
    for times in [[0.1, 0.1], [0.2, 0.1]] {
        let samples = [
            state(times[0], 0.03, -1.0, Vector3::zero()),
            state(times[1], 0.01, -1.0, Vector3::zero()),
        ];
        let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
        assert_eq!(
            outcome,
            GroundTransferOutcome::Unavailable(TransferUnavailableReason::MalformedSamples)
        );
    }
}

#[test]
fn tangent_contact_is_typed_grazing() {
    let samples = [
        state(0.0, 0.03, -0.2, Vector3::zero()),
        state(0.1, 0.02, 0.0, Vector3::zero()),
        state(0.2, 0.03, 0.2, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
    assert!(matches!(outcome, GroundTransferOutcome::Grazing { .. }));
}

#[test]
fn bounce_restart_does_not_retrigger_at_zero_time() {
    let samples = [
        state(0.0, 0.02, 3.0, Vector3::zero()),
        state(0.1, 0.20, 1.0, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
    assert!(matches!(outcome, GroundTransferOutcome::NoCrossing { .. }));
}

#[test]
fn zero_time_contact_or_grazing_is_not_a_transfer() {
    for vertical_speed in [-1.0, 0.0] {
        let initial = state(0.0, 0.02, vertical_speed, Vector3::zero());
        let separated = state(0.1, 0.03, 1.0, Vector3::zero());
        let outcome =
            qualify_ground_transfer(&[initial, separated], &PlanarGround::horizontal(0.0), 0.02);
        assert!(matches!(outcome, GroundTransferOutcome::NoCrossing { .. }));
    }
}

#[test]
fn coarse_nonincoming_bracket_is_typed_unavailable() {
    let samples = [
        state(0.0, 0.03, 2.0, Vector3::zero()),
        state(0.2, 0.01, -2.0, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
    assert!(matches!(outcome, GroundTransferOutcome::Unavailable(_)));
}

#[test]
fn bounce_restart_can_detect_a_later_return() {
    let samples = [
        state(0.0, 0.02, 3.0, Vector3::zero()),
        state(0.1, 0.20, 1.0, Vector3::zero()),
        state(0.2, 0.03, -1.0, Vector3::zero()),
        state(0.3, 0.01, -3.0, Vector3::zero()),
    ];
    let outcome = qualify_ground_transfer(&samples, &PlanarGround::horizontal(0.0), 0.02);
    let GroundTransferOutcome::Contact(event) = outcome else {
        panic!("expected later return contact");
    };
    assert!(event.contact.time > 0.2);
}

#[test]
fn invalid_runtime_input_returns_typed_unavailable() {
    let bad = state(0.0, f64::INFINITY, 0.0, Vector3::zero());
    let outcome = qualify_ground_transfer(&[bad], &PlanarGround::horizontal(0.0), 0.02);
    assert!(matches!(outcome, GroundTransferOutcome::Unavailable(_)));
}

#[test]
fn flight_run_preserves_signed_angular_velocity_and_physical_contact_height() {
    let ball = BallProperties::default();
    let launch = LaunchConditions {
        velocity: 12.0,
        launch_angle: 35.0,
        azimuth_angle: 0.0,
        spin_rate: 1200.0,
        spin_axis: Vector3::new(0.0, -1.0, 0.0),
    };
    let config = FlightGroundConfig {
        max_time: 4.0,
        dt: 0.01,
        launch_geometry: LaunchGeometry::ground(),
        ground: PlanarGround::horizontal(0.0),
    };
    let run =
        simulate_flight_to_ground(&ball, &EnvironmentalConditions::default(), &launch, &config)
            .expect("valid simulation");

    assert!(run.trajectory[0].angular_velocity.y < 0.0);
    let GroundTransferOutcome::Contact(event) = run.outcome else {
        panic!("expected simulated ground contact");
    };
    assert!(run.trajectory[0].position.magnitude() < EPSILON);
    assert!(event.contact.position.y.abs() < 1.0e-6);
    assert!(event.contact.angular_velocity.y < 0.0);
}

#[test]
fn tee_flight_lands_below_request_origin_by_tee_height() {
    let ball = BallProperties::default();
    let tee_height = 0.0381;
    let config = FlightGroundConfig {
        max_time: 4.0,
        dt: 0.01,
        launch_geometry: LaunchGeometry::tee(tee_height).unwrap(),
        ground: PlanarGround::horizontal(0.0),
    };
    let run = simulate_flight_to_ground(
        &ball,
        &EnvironmentalConditions::default(),
        &LaunchConditions {
            velocity: 12.0,
            launch_angle: 35.0,
            ..LaunchConditions::default()
        },
        &config,
    )
    .unwrap();
    let GroundTransferOutcome::Contact(event) = run.outcome else {
        panic!("expected tee flight contact");
    };
    assert!((event.contact.position.y + tee_height).abs() < 1.0e-6);
    assert!((run.trajectory.last().unwrap().position.y + tee_height).abs() < 1.0e-6);
}

#[test]
fn support_mode_is_explicitly_typed() {
    assert_eq!(LaunchGeometry::ground().support, LaunchSupport::Ground);
    assert_eq!(
        LaunchGeometry::tee(0.04).unwrap().support,
        LaunchSupport::Tee
    );
}

#[test]
fn negative_spin_magnitude_is_rejected() {
    let launch = LaunchConditions {
        spin_rate: -1.0,
        ..LaunchConditions::default()
    };
    let config = FlightGroundConfig {
        max_time: 1.0,
        dt: 0.01,
        launch_geometry: LaunchGeometry::ground(),
        ground: PlanarGround::horizontal(0.0),
    };
    let result = simulate_flight_to_ground(
        &BallProperties::default(),
        &EnvironmentalConditions::default(),
        &launch,
        &config,
    );
    assert!(result.is_err());
}

#[test]
fn v1_fixture_round_trips_and_sample_adapter_preserves_all_context() {
    let fixture_text = include_str!(
        "../../../src/rate_of_closure/web/src/model/__fixtures__/flight_to_ground_golden_v1.json"
    )
    .replace("\\ud800", "rejected-surrogate");
    let fixture: serde_json::Value = serde_json::from_str(&fixture_text).unwrap();
    let request_json = serde_json::to_string(&fixture["request"]).unwrap();
    let request = parse_request_v1_json(&request_json).expect("valid shared fixture");
    assert_eq!(serde_json::to_value(&request).unwrap(), fixture["request"]);

    let samples = [
        FlightState::new(
            5.19,
            Vector3::new(209.7, 0.024, -3.01),
            Vector3::new(31.0, -12.0, 1.5),
            Vector3::new(0.0, 260.0, -4.0),
        ),
        FlightState::new(
            5.2,
            Vector3::new(210.0, 0.019, -3.0),
            Vector3::new(31.0, -12.0, 1.5),
            Vector3::new(0.0, 260.0, -4.0),
        ),
    ];
    let adapted = adapt_samples_to_request_v1(&samples, request).unwrap();
    let adapted_value = serde_json::to_value(adapted).unwrap();
    for field in [
        "surface",
        "calibration",
        "provenance",
        "ball_radius_m",
        "ball_mass_kg",
        "rotational_inertia_factor",
        "schema_version",
        "unit_system",
        "request_id",
    ] {
        assert_eq!(adapted_value[field], fixture["request"][field]);
    }
}
