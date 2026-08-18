use super::*;

fn holding_case(residual: f64) -> (Body, GroundSurfaceV1, State, State) {
    let body = Body {
        radius: 0.02135,
        mass: 0.04593,
        factor: 0.4,
    };
    let angle = 0.05_f64;
    let mut moving_surface = surface();
    moving_surface.normal_unit = [0.0, angle.cos(), angle.sin()];
    moving_surface.surface_velocity_m_s = [0.04, 0.0, 0.0];
    moving_surface.rolling_resistance = 0.04;
    let uphill = [0.0, angle.sin(), -angle.cos()];
    let before = State {
        time: 0.0,
        position: scale(moving_surface.normal_unit, body.radius),
        velocity: add(moving_surface.surface_velocity_m_s, scale(uphill, residual)),
        spin: [-residual / body.radius, 0.0, 0.0],
    };
    let after = holding_state(before, &moving_surface).unwrap();
    (body, moving_surface, before, after)
}

fn assert_projection(residual: f64, correction_tolerance: f64) {
    let (body, moving_surface, before, after) = holding_case(residual);
    SurfaceLedger::default()
        .record_projection(
            before,
            after,
            &moving_surface,
            body,
            1.0e-9,
            correction_tolerance,
        )
        .unwrap();
    assert_eq!(after.velocity, moving_surface.surface_velocity_m_s);
    assert_eq!(after.spin, ZERO);
}

#[test]
fn holding_projection_removes_a_sub_tolerance_uphill_residual() {
    assert_projection(9.0e-10, 1.0e-9);
}

#[test]
fn holding_projection_uses_the_independent_velocity_tolerance() {
    assert_projection(5.0e-7, 1.0e-6);
}
