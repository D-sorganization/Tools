//! Passive restitution and Coulomb impulse used by the compiled reference run.

use super::request_v1::GroundSurfaceV1;
use super::runtime_math::{
    add, cross, dot, kinetic_energy, norm, scale, sub, tangent, unit, Body, State,
};

const VELOCITY_TOLERANCE: f64 = 1.0e-12;
const IMPULSE_TOLERANCE: f64 = 1.0e-12;
const ENERGY_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
pub(super) struct Impact {
    pub before: State,
    pub after: State,
    pub restitution: f64,
}

pub(super) fn resolve_impact(
    state: State,
    surface: &GroundSurfaceV1,
    body: Body,
    restitution: f64,
) -> Result<Impact, &'static str> {
    let normal = surface.normal_unit;
    let arm = scale(normal, -body.radius);
    let contact_before = contact_velocity(state, surface, arm);
    let incoming_normal = dot(contact_before, normal);
    if incoming_normal.abs() <= VELOCITY_TOLERANCE {
        return Err("grazing");
    }
    if incoming_normal > 0.0 {
        return Err("outgoing");
    }
    let normal_impulse = -(1.0 + restitution) * body.mass * incoming_normal;
    let (tangential_impulse, sticking) =
        tangent_impulse(contact_before, surface, body, normal_impulse);
    let total_impulse = add(scale(normal, normal_impulse), tangential_impulse);
    let after = State {
        velocity: add(state.velocity, scale(total_impulse, body.mass.recip())),
        spin: add(
            state.spin,
            scale(cross(arm, tangential_impulse), body.inertia().recip()),
        ),
        ..state
    }
    .normalized()
    .map_err(|error| error.reason())?;
    validate_postconditions(Postconditions {
        before: contact_before,
        after: contact_velocity(after, surface, arm),
        normal,
        restitution,
        tangential_impulse,
        normal_impulse,
        static_friction: surface.static_friction,
        sticking,
    })?;
    validate_passivity(
        state,
        after,
        body,
        total_impulse,
        surface.surface_velocity_m_s,
    )?;
    Ok(Impact {
        before: state,
        after,
        restitution,
    })
}

fn contact_velocity(state: State, surface: &GroundSurfaceV1, arm: [f64; 3]) -> [f64; 3] {
    sub(
        add(state.velocity, cross(state.spin, arm)),
        surface.surface_velocity_m_s,
    )
}

fn tangent_impulse(
    contact_velocity: [f64; 3],
    surface: &GroundSurfaceV1,
    body: Body,
    normal_impulse: f64,
) -> ([f64; 3], bool) {
    let slip = tangent(contact_velocity, surface.normal_unit);
    if norm(slip) <= VELOCITY_TOLERANCE {
        return ([0.0; 3], true);
    }
    let desired = scale(slip, -body.tangential_mass());
    if norm(desired) <= surface.static_friction * normal_impulse + IMPULSE_TOLERANCE {
        return (desired, true);
    }
    (
        scale(
            unit(slip, VELOCITY_TOLERANCE).unwrap_or([0.0; 3]),
            -surface.kinetic_friction * normal_impulse,
        ),
        false,
    )
}

struct Postconditions {
    before: [f64; 3],
    after: [f64; 3],
    normal: [f64; 3],
    restitution: f64,
    tangential_impulse: [f64; 3],
    normal_impulse: f64,
    static_friction: f64,
    sticking: bool,
}

fn validate_postconditions(values: Postconditions) -> Result<(), &'static str> {
    let before_normal = dot(values.before, values.normal);
    let after_normal = dot(values.after, values.normal);
    let expected = -values.restitution * before_normal;
    if (after_normal - expected).abs() > ENERGY_TOLERANCE.max(ENERGY_TOLERANCE * expected.abs()) {
        return Err("restitution_postcondition");
    }
    if norm(values.tangential_impulse)
        > values.static_friction * values.normal_impulse + IMPULSE_TOLERANCE
    {
        return Err("friction_cone");
    }
    let tangent_before = tangent(values.before, values.normal);
    let tangent_after = tangent(values.after, values.normal);
    if values.sticking && norm(tangent_after) > 1.0e-9 {
        return Err("sticking_slip");
    }
    if !values.sticking && dot(tangent_before, tangent_after) < -ENERGY_TOLERANCE {
        return Err("slip_reversal");
    }
    Ok(())
}

fn validate_passivity(
    before: State,
    after: State,
    body: Body,
    impulse: [f64; 3],
    surface_velocity: [f64; 3],
) -> Result<(), &'static str> {
    let energy_before = kinetic_energy(before, body);
    let energy_after = kinetic_energy(after, body);
    let boundary_work = dot(impulse, surface_velocity);
    let scale = energy_before.max(energy_after).max(boundary_work.abs());
    if energy_before + boundary_work - energy_after < -(ENERGY_TOLERANCE + ENERGY_TOLERANCE * scale)
    {
        return Err("impact_passivity");
    }
    Ok(())
}
