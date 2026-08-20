//! Rigid-sphere skid and rolling kinematics on an arbitrary immutable plane.

use super::execution_v1::SkidRollExecutionSettingsV1;
use super::request_v1::GroundSurfaceV1;
use super::result_v1::GroundPhaseV1;
use super::runtime_math::{
    add, advance_raw, cross, dot, ensure_numeric_range, kinetic_energy, norm, path_distance, scale,
    sub, tangent, unit, Body, MathResult, Motion, State,
};

const ZERO: [f64; 3] = [0.0; 3];
const FEASIBILITY_TOLERANCE: f64 = 1.0e-12;
const CANONICAL_QUANTUM: f64 = 1.0e-11;
const FLOATING_ERROR_MULTIPLIER: f64 = 64.0;

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
    surface_work: f64,
    physical_dissipation: f64,
    canonical_energy_budget: f64,
    operation_count: u64,
}

pub(super) struct LedgerStep<'a> {
    pub phase: Phase,
    pub start: State,
    pub motion: Motion,
    pub duration: f64,
    pub surface: &'a GroundSurfaceV1,
    pub body: Body,
    pub gravity: [f64; 3],
    pub end: State,
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
        let surface_work = self.surface_work
            + dot(step.motion.contact_force, step.surface.surface_velocity_m_s) * step.duration;
        ensure_numeric_range(surface_work)?;
        self.surface_work = surface_work;
        let raw_end = advance_raw(step.start, step.motion, step.duration);
        let gravity_work =
            step.body.mass * dot(step.gravity, sub(raw_end.position, step.start.position));
        let segment_surface_work =
            dot(step.motion.contact_force, step.surface.surface_velocity_m_s) * step.duration;
        let segment_dissipation =
            kinetic_energy(step.start, step.body) + gravity_work + segment_surface_work
                - kinetic_energy(raw_end, step.body);
        let segment_scale = 1.0_f64
            .max(kinetic_energy(step.start, step.body))
            .max(kinetic_energy(raw_end, step.body))
            .max(gravity_work.abs())
            .max(segment_surface_work.abs());
        if segment_dissipation < -floating_tolerance(segment_scale, 1) {
            return Err(super::runtime_math::RuntimeMathError::Passivity);
        }
        let physical_dissipation = self.physical_dissipation + segment_dissipation.max(0.0);
        ensure_numeric_range(physical_dissipation)?;
        self.physical_dissipation = physical_dissipation;
        let snap_budget = canonical_snap_energy_budget(raw_end, step.end, step.body, step.gravity)?;
        let canonical_energy_budget = self.canonical_energy_budget + snap_budget;
        ensure_numeric_range(canonical_energy_budget)?;
        self.canonical_energy_budget = canonical_energy_budget;
        self.operation_count = self
            .operation_count
            .checked_add(1)
            .ok_or(super::runtime_math::RuntimeMathError::CanonicalRange)?;
        Ok(())
    }

    pub fn record_projection(
        &mut self,
        before: State,
        after: State,
        surface: &GroundSurfaceV1,
        body: Body,
        slip_tolerance: f64,
        correction_tolerance: f64,
    ) -> MathResult<()> {
        let slip = norm(contact_slip(before, surface, body));
        if slip > slip_tolerance + floating_tolerance(slip.max(1.0), 1) {
            return Err(super::runtime_math::RuntimeMathError::Passivity);
        }
        let velocity_change = norm(sub(after.velocity, before.velocity));
        let spin_change = norm(sub(after.spin, before.spin));
        let rounding = 3.0_f64.sqrt() * CANONICAL_QUANTUM;
        if velocity_change > correction_tolerance + rounding
            || spin_change > correction_tolerance / body.radius + rounding
        {
            return Err(super::runtime_math::RuntimeMathError::Passivity);
        }
        let energy_creation = (kinetic_energy(after, body) - kinetic_energy(before, body)).max(0.0);
        let updated = self.canonical_energy_budget + energy_creation;
        ensure_numeric_range(updated)?;
        self.canonical_energy_budget = updated;
        self.operation_count = self
            .operation_count
            .checked_add(1)
            .ok_or(super::runtime_math::RuntimeMathError::CanonicalRange)?;
        Ok(())
    }

    pub fn validate_passivity(
        &self,
        initial: State,
        final_state: State,
        body: Body,
        gravity: [f64; 3],
    ) -> Result<(), &'static str> {
        let initial_energy = kinetic_energy(initial, body);
        let final_energy = kinetic_energy(final_state, body);
        let displacement = sub(final_state.position, initial.position);
        let gravity_work = body.mass * dot(gravity, displacement);
        ensure_numeric_range(gravity_work).map_err(|_| "numeric_range")?;
        let dissipation = initial_energy + gravity_work + self.surface_work - final_energy;
        let scale = 1.0_f64
            .max(initial_energy)
            .max(final_energy)
            .max(gravity_work.abs())
            .max(self.surface_work.abs());
        let tolerance = self.canonical_energy_budget
            + floating_tolerance(scale, self.operation_count.saturating_add(1));
        (dissipation >= -tolerance)
            .then_some(())
            .ok_or("surface_passivity")
    }
}

fn floating_tolerance(scale: f64, operations: u64) -> f64 {
    FLOATING_ERROR_MULTIPLIER * f64::EPSILON * scale.max(1.0) * operations as f64
}

fn component_error_bound(value: f64) -> f64 {
    0.5 * CANONICAL_QUANTUM + 4.0 * f64::EPSILON * value.abs().max(1.0)
}

fn vector_error_bound(raw: [f64; 3], canonical: [f64; 3]) -> MathResult<f64> {
    let bounds = raw.map(component_error_bound);
    if (0..3).any(|index| (canonical[index] - raw[index]).abs() > bounds[index]) {
        return Err(super::runtime_math::RuntimeMathError::Passivity);
    }
    Ok(norm(bounds))
}

fn canonical_snap_energy_budget(
    raw: State,
    canonical: State,
    body: Body,
    gravity: [f64; 3],
) -> MathResult<f64> {
    let position_error = vector_error_bound(raw.position, canonical.position)?;
    let velocity_error = vector_error_bound(raw.velocity, canonical.velocity)?;
    let spin_error = vector_error_bound(raw.spin, canonical.spin)?;
    let budget = body.mass * norm(gravity) * position_error
        + body.mass * (norm(raw.velocity) * velocity_error + 0.5 * velocity_error.powi(2))
        + body.inertia() * (norm(raw.spin) * spin_error + 0.5 * spin_error.powi(2));
    ensure_numeric_range(budget)?;
    Ok(budget)
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

pub(super) fn holding_motion(surface: &GroundSurfaceV1, body: Body, gravity: [f64; 3]) -> Motion {
    let gravity_tangent = tangent(gravity, surface.normal_unit);
    Motion {
        acceleration: ZERO,
        angular_acceleration: ZERO,
        slip_acceleration: ZERO,
        contact_force: scale(gravity_tangent, -body.mass),
    }
}

pub(super) fn holding_state(state: State, surface: &GroundSurfaceV1) -> MathResult<State> {
    let axial_spin = dot(state.spin, surface.normal_unit);
    State {
        velocity: surface.surface_velocity_m_s,
        spin: scale(surface.normal_unit, axial_spin),
        ..state
    }
    .normalized()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn surface() -> GroundSurfaceV1 {
        GroundSurfaceV1 {
            surface_id: "test".to_owned(),
            provider_id: "test".to_owned(),
            provider_version: "1".to_owned(),
            frame: "target_frame:x_downrange,y_up,z_right".to_owned(),
            height_m: 0.0,
            normal_unit: [0.0, 1.0, 0.0],
            surface_velocity_m_s: ZERO,
            normal_restitution: 0.0,
            static_friction: 0.3,
            kinetic_friction: 0.2,
            rolling_resistance: 0.0,
            firmness_pa: 1.0,
            hardness_fraction: 0.5,
            grass_height_m: 0.0,
            compressibility_fraction: 0.0,
            compression_damping_fraction: 0.0,
            turf_density_kg_m3: 1.0,
            moisture_fraction: 0.0,
        }
    }

    #[test]
    fn each_segment_rejects_energy_creation_without_masking() {
        let body = Body {
            radius: 0.02135,
            mass: 0.04593,
            factor: 0.4,
        };
        let start = State {
            time: 0.0,
            position: ZERO,
            velocity: [1.0, 0.0, 0.0],
            spin: [0.0, 0.0, -1.0 / body.radius],
        };
        let motion = Motion {
            acceleration: [10.0, 0.0, 0.0],
            angular_acceleration: ZERO,
            slip_acceleration: [10.0, 0.0, 0.0],
            contact_force: ZERO,
        };
        let duration = 0.01;
        let end = advance_raw(start, motion, duration).normalized().unwrap();
        let mut ledger = SurfaceLedger {
            physical_dissipation: 0.01,
            ..SurfaceLedger::default()
        };

        assert_eq!(
            ledger.record(LedgerStep {
                phase: Phase::Roll,
                start,
                motion,
                duration,
                surface: &surface(),
                body,
                gravity: [0.0, -9.80665, 0.0],
                end,
            }),
            Err(super::super::runtime_math::RuntimeMathError::Passivity)
        );
    }

    #[test]
    fn unbudgeted_endpoint_energy_creation_is_rejected() {
        let body = Body {
            radius: 0.02135,
            mass: 0.04593,
            factor: 0.4,
        };
        let initial = State {
            time: 0.0,
            position: ZERO,
            velocity: ZERO,
            spin: ZERO,
        };
        let final_state = State {
            velocity: [1.0, 0.0, 0.0],
            ..initial
        };

        assert_eq!(
            SurfaceLedger::default().validate_passivity(
                initial,
                final_state,
                body,
                [0.0, -9.80665, 0.0]
            ),
            Err("surface_passivity")
        );
    }

    #[test]
    fn rolling_projection_rejects_an_off_manifold_state() {
        let body = Body {
            radius: 0.02135,
            mass: 0.04593,
            factor: 0.4,
        };
        let before = State {
            time: 0.0,
            position: [0.0, body.radius, 0.0],
            velocity: [1.0, 0.0, 0.0],
            spin: ZERO,
        };
        let after = rolling_state(before, &surface(), body).unwrap();

        assert_eq!(
            SurfaceLedger::default().record_projection(
                before,
                after,
                &surface(),
                body,
                1.0e-9,
                1.0e-9,
            ),
            Err(super::super::runtime_math::RuntimeMathError::Passivity)
        );
    }

    #[test]
    fn canonical_snap_rejects_a_component_outside_the_wire_quantum() {
        let body = Body {
            radius: 0.02135,
            mass: 0.04593,
            factor: 0.4,
        };
        let raw = State {
            time: 0.0,
            position: ZERO,
            velocity: ZERO,
            spin: ZERO,
        };
        let forged = State {
            position: [1.0e-6, 0.0, 0.0],
            ..raw
        };

        assert_eq!(
            canonical_snap_energy_budget(raw, forged, body, [0.0, -9.80665, 0.0]),
            Err(super::super::runtime_math::RuntimeMathError::Passivity)
        );
    }

    #[path = "surface_dynamics_holding_tests.rs"]
    mod holding;
}
