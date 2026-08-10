//! Rigid-sphere skid and rolling kinematics on an arbitrary immutable plane.

use super::execution_v1::SkidRollExecutionSettingsV1;
use super::request_v1::GroundSurfaceV1;
use super::result_v1::GroundPhaseV1;
use super::runtime_math::{
    add, cross, dot, ensure_numeric_range, kinetic_energy, norm, path_distance, scale, sub,
    tangent, unit, Body, MathResult, Motion, State,
};

const ZERO: [f64; 3] = [0.0; 3];
const FEASIBILITY_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Phase {
    Skid,
    Roll,
}

impl Phase {
    pub fn wire(self) -> GroundPhaseV1 {
        match self {
            Self::Skid => GroundPhaseV1::Skid,
            Self::Roll => GroundPhaseV1::Roll,
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct SurfaceLedger {
    pub skid_distance: f64,
    pub roll_distance: f64,
    gravity_work: f64,
    surface_work: f64,
}

pub(super) struct LedgerStep<'a> {
    pub phase: Phase,
    pub start: State,
    pub end: State,
    pub motion: Motion,
    pub duration: f64,
    pub surface: &'a GroundSurfaceV1,
    pub body: Body,
    pub gravity: [f64; 3],
}

impl SurfaceLedger {
    pub fn record(&mut self, step: LedgerStep<'_>) -> MathResult<()> {
        let distance = path_distance(
            center_relative_velocity(step.start, step.surface),
            step.motion.acceleration,
            step.duration,
        );
        match step.phase {
            Phase::Skid => {
                let updated = self.skid_distance + distance;
                ensure_numeric_range(updated)?;
                self.skid_distance = updated;
            }
            Phase::Roll => {
                let updated = self.roll_distance + distance;
                ensure_numeric_range(updated)?;
                self.roll_distance = updated;
            }
        }
        let displacement = sub(step.end.position, step.start.position);
        let gravity_work = self.gravity_work + step.body.mass * dot(step.gravity, displacement);
        ensure_numeric_range(gravity_work)?;
        self.gravity_work = gravity_work;
        let surface_work = self.surface_work
            + dot(step.motion.contact_force, step.surface.surface_velocity_m_s) * step.duration;
        ensure_numeric_range(surface_work)?;
        self.surface_work = surface_work;
        Ok(())
    }

    pub fn validate_passivity(
        &self,
        initial: State,
        final_state: State,
        body: Body,
    ) -> Result<(), &'static str> {
        let initial_energy = kinetic_energy(initial, body);
        let final_energy = kinetic_energy(final_state, body);
        let dissipation = initial_energy + self.gravity_work + self.surface_work - final_energy;
        let tolerance = 1.0e-9
            + 1.0e-9
                * initial_energy
                    .max(final_energy)
                    .max(self.surface_work.abs());
        (dissipation >= -tolerance)
            .then_some(())
            .ok_or("surface_passivity")
    }
}

pub(super) fn contact_slip(state: State, surface: &GroundSurfaceV1, body: Body) -> [f64; 3] {
    let arm = scale(surface.normal_unit, -body.radius);
    tangent(
        sub(
            add(state.velocity, cross(state.spin, arm)),
            surface.surface_velocity_m_s,
        ),
        surface.normal_unit,
    )
}

pub(super) fn center_relative_velocity(state: State, surface: &GroundSurfaceV1) -> [f64; 3] {
    tangent(
        sub(state.velocity, surface.surface_velocity_m_s),
        surface.normal_unit,
    )
}

pub(super) fn skid_motion(
    state: State,
    surface: &GroundSurfaceV1,
    body: Body,
    gravity: [f64; 3],
) -> Option<Motion> {
    let slip = contact_slip(state, surface, body);
    let direction = unit(slip, 1.0e-15)?;
    let normal_force = body.mass * normal_gravity(surface, gravity);
    let friction = scale(direction, -surface.kinetic_friction * normal_force);
    let gravity_tangent = tangent(gravity, surface.normal_unit);
    let acceleration = add(gravity_tangent, scale(friction, body.mass.recip()));
    let arm = scale(surface.normal_unit, -body.radius);
    let angular_acceleration = scale(cross(arm, friction), body.inertia().recip());
    Some(Motion {
        acceleration,
        angular_acceleration,
        slip_acceleration: add(acceleration, cross(angular_acceleration, arm)),
        contact_force: friction,
    })
}

pub(super) fn rolling_feasible(surface: &GroundSurfaceV1, body: Body, gravity: [f64; 3]) -> bool {
    let normal = normal_gravity(surface, gravity);
    let gravity_tangent = tangent(gravity, surface.normal_unit);
    let required = body.mass * body.factor / (1.0 + body.factor) * norm(gravity_tangent);
    required <= surface.static_friction * body.mass * normal + FEASIBILITY_TOLERANCE
}

pub(super) fn rolling_state(
    state: State,
    surface: &GroundSurfaceV1,
    body: Body,
) -> MathResult<State> {
    let normal = surface.normal_unit;
    let relative = center_relative_velocity(state, surface);
    let axial_spin = dot(state.spin, normal);
    State {
        velocity: add(surface.surface_velocity_m_s, relative),
        spin: add(
            scale(cross(normal, relative), body.radius.recip()),
            scale(normal, axial_spin),
        ),
        ..state
    }
    .normalized()
}

pub(super) fn rolling_motion(
    state: State,
    surface: &GroundSurfaceV1,
    body: Body,
    gravity: [f64; 3],
) -> Motion {
    let gravity_tangent = tangent(gravity, surface.normal_unit);
    let drive = scale(gravity_tangent, (1.0 + body.factor).recip());
    let relative = center_relative_velocity(state, surface);
    let direction = unit(relative, 1.0e-15)
        .or_else(|| unit(drive, 1.0e-15))
        .unwrap_or(ZERO);
    let resistance = surface.rolling_resistance * normal_gravity(surface, gravity);
    let acceleration = add(drive, scale(direction, -resistance));
    let contact_force = scale(sub(acceleration, gravity_tangent), body.mass);
    Motion {
        acceleration,
        angular_acceleration: scale(
            cross(surface.normal_unit, acceleration),
            body.radius.recip(),
        ),
        slip_acceleration: ZERO,
        contact_force,
    }
}

pub(super) fn stable_at_zero(surface: &GroundSurfaceV1, body: Body, gravity: [f64; 3]) -> bool {
    let drive = norm(tangent(gravity, surface.normal_unit)) / (1.0 + body.factor);
    let resistance = surface.rolling_resistance * normal_gravity(surface, gravity);
    rolling_feasible(surface, body, gravity) && drive <= resistance + FEASIBILITY_TOLERANCE
}

pub(super) fn can_rest(
    state: State,
    surface: &GroundSurfaceV1,
    body: Body,
    settings: &SkidRollExecutionSettingsV1,
) -> bool {
    surface.surface_velocity_m_s == ZERO
        && stable_at_zero(surface, body, settings.gravity_m_s2)
        && norm(state.velocity) <= settings.velocity_tolerance_m_s
        && norm(state.spin) <= settings.angular_tolerance_rad_s
}

fn normal_gravity(surface: &GroundSurfaceV1, gravity: [f64; 3]) -> f64 {
    -dot(gravity, surface.normal_unit)
}
