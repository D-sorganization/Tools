//! Small deterministic vector and rigid-body primitives for the ground runtime.

use super::canonical_json::normalize_f64;
use super::request_v1::{FlightToGroundRequestV1, GroundContactStateV1, GroundSurfaceV1};
use super::result_v1::{GroundPhaseV1, GroundTrajectoryPointV1};

pub(super) type Vec3 = [f64; 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeMathError {
    CanonicalRange,
    Passivity,
}

impl RuntimeMathError {
    pub fn reason(self) -> &'static str {
        match self {
            Self::CanonicalRange => "numeric_range",
            Self::Passivity => "surface_passivity",
        }
    }
}

pub(super) type MathResult<T> = Result<T, RuntimeMathError>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct State {
    pub time: f64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub spin: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Body {
    pub radius: f64,
    pub mass: f64,
    pub factor: f64,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Motion {
    pub acceleration: Vec3,
    pub angular_acceleration: Vec3,
    pub slip_acceleration: Vec3,
    pub contact_force: Vec3,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OutputSchedule {
    interval: f64,
    next_index: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TimelineAppend {
    Push,
    Replace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TimelineError {
    Order,
    Resolution,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct WireTimeline {
    last_elapsed: Option<f64>,
    last_wire: Option<f64>,
}

impl OutputSchedule {
    pub fn new(interval: f64) -> Self {
        Self {
            interval,
            next_index: 1,
        }
    }

    pub fn skip_through(&mut self, elapsed: f64, tolerance: f64) -> Option<()> {
        let covered = ((elapsed + tolerance) / self.interval).floor();
        if !covered.is_finite() || covered < 0.0 || covered >= u64::MAX as f64 {
            return None;
        }
        let target = (covered as u64).checked_add(1)?;
        self.next_index = self.next_index.max(target);
        Some(())
    }

    pub fn next_elapsed(self) -> MathResult<f64> {
        canonical(self.interval * self.next_index as f64)
    }

    pub fn advance(&mut self) -> Option<()> {
        self.next_index = self.next_index.checked_add(1)?;
        Some(())
    }
}

impl WireTimeline {
    pub fn anchored(elapsed: f64, wire: f64) -> Self {
        Self {
            last_elapsed: Some(elapsed),
            last_wire: Some(wire),
        }
    }

    pub fn classify(
        &mut self,
        elapsed: f64,
        wire: f64,
        tolerance: f64,
    ) -> Result<TimelineAppend, TimelineError> {
        let Some((last_elapsed, last_wire)) = self.last_elapsed.zip(self.last_wire) else {
            self.last_elapsed = Some(elapsed);
            self.last_wire = Some(wire);
            return Ok(TimelineAppend::Push);
        };
        let delta = elapsed - last_elapsed;
        let action = if delta.abs() <= tolerance {
            TimelineAppend::Replace
        } else if delta < 0.0 {
            return Err(TimelineError::Order);
        } else if wire <= last_wire {
            return Err(TimelineError::Resolution);
        } else {
            TimelineAppend::Push
        };
        self.last_elapsed = Some(elapsed);
        self.last_wire = Some(wire);
        Ok(action)
    }
}

impl Body {
    pub fn from_request(request: &FlightToGroundRequestV1) -> Self {
        Self {
            radius: request.ball_radius_m,
            mass: request.ball_mass_kg,
            factor: request.rotational_inertia_factor,
        }
    }

    pub fn inertia(self) -> f64 {
        self.factor * self.mass * self.radius.powi(2)
    }

    pub fn tangential_mass(self) -> f64 {
        self.mass * self.factor / (self.factor + 1.0)
    }
}

impl State {
    pub fn from_wire(value: &GroundContactStateV1) -> Self {
        Self {
            time: value.time_s,
            position: value.position_m,
            velocity: value.velocity_m_s,
            spin: value.angular_velocity_rad_s,
        }
    }

    pub fn point(
        self,
        frame: &str,
        phase: GroundPhaseV1,
        time_origin: f64,
    ) -> MathResult<GroundTrajectoryPointV1> {
        Ok(GroundTrajectoryPointV1 {
            time_s: wire_time(time_origin, self.time)?,
            frame: frame.to_owned(),
            position_m: self.position,
            velocity_m_s: self.velocity,
            angular_velocity_rad_s: self.spin,
            phase,
        })
    }

    pub fn normalized(mut self) -> MathResult<Self> {
        self.time = canonical(self.time)?;
        self.position = normalized_vector(self.position)?;
        self.velocity = normalized_vector(self.velocity)?;
        self.spin = normalized_vector(self.spin)?;
        Ok(self)
    }
}

pub(super) fn wire_time(origin: f64, elapsed: f64) -> MathResult<f64> {
    normalize_f64(origin + elapsed).map_err(|_| RuntimeMathError::CanonicalRange)
}

pub(super) fn canonical_scalar(value: f64) -> MathResult<f64> {
    canonical(value)
}

pub(super) fn ensure_numeric_range(value: f64) -> MathResult<()> {
    canonical(value).map(|_| ())
}

pub(super) fn add(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(super) fn sub(left: Vec3, right: Vec3) -> Vec3 {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(super) fn scale(value: Vec3, factor: f64) -> Vec3 {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

pub(super) fn dot(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(super) fn cross(left: Vec3, right: Vec3) -> Vec3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(super) fn norm(value: Vec3) -> f64 {
    dot(value, value).sqrt()
}

pub(super) fn unit(value: Vec3, tolerance: f64) -> Option<Vec3> {
    let magnitude = norm(value);
    (magnitude > tolerance).then(|| scale(value, magnitude.recip()))
}

pub(super) fn tangent(value: Vec3, normal: Vec3) -> Vec3 {
    sub(value, scale(normal, dot(value, normal)))
}

pub(super) fn surface_point(surface: &GroundSurfaceV1) -> Vec3 {
    [0.0, surface.height_m, 0.0]
}

pub(super) fn signed_gap(state: State, request: &FlightToGroundRequestV1) -> f64 {
    dot(
        sub(state.position, surface_point(&request.surface)),
        request.surface.normal_unit,
    ) - request.ball_radius_m
}

pub(super) fn project_contact(
    mut state: State,
    request: &FlightToGroundRequestV1,
) -> MathResult<State> {
    let gap = signed_gap(state, request);
    state.position = sub(state.position, scale(request.surface.normal_unit, gap));
    state.normalized()
}

pub(super) fn interpolate_contact(request: &FlightToGroundRequestV1) -> MathResult<State> {
    let left = State::from_wire(&request.last_separated_state);
    let right = State::from_wire(&request.first_penetrating_state);
    let left_gap = signed_gap(left, request);
    let right_gap = signed_gap(right, request);
    let fraction = left_gap / (left_gap - right_gap);
    project_contact(interpolate(left, right, fraction)?, request)
}

fn interpolate(left: State, right: State, fraction: f64) -> MathResult<State> {
    State {
        time: left.time + fraction * (right.time - left.time),
        position: add(
            left.position,
            scale(sub(right.position, left.position), fraction),
        ),
        velocity: add(
            left.velocity,
            scale(sub(right.velocity, left.velocity), fraction),
        ),
        spin: add(left.spin, scale(sub(right.spin, left.spin), fraction)),
    }
    .normalized()
}

pub(super) fn advance(state: State, motion: Motion, duration: f64) -> MathResult<State> {
    advance_raw(state, motion, duration).normalized()
}

pub(super) fn advance_raw(state: State, motion: Motion, duration: f64) -> State {
    State {
        time: state.time + duration,
        position: add(
            state.position,
            add(
                scale(state.velocity, duration),
                scale(motion.acceleration, 0.5 * duration.powi(2)),
            ),
        ),
        velocity: add(state.velocity, scale(motion.acceleration, duration)),
        spin: add(state.spin, scale(motion.angular_acceleration, duration)),
    }
}

pub(super) fn ballistic(state: State, gravity: Vec3, duration: f64) -> MathResult<State> {
    advance(
        state,
        Motion {
            acceleration: gravity,
            angular_acceleration: [0.0; 3],
            slip_acceleration: [0.0; 3],
            contact_force: [0.0; 3],
        },
        duration,
    )
}

pub(super) fn time_to_zero(value: Vec3, rate: Vec3, tolerance: f64) -> Option<f64> {
    let denominator = dot(rate, rate);
    if denominator <= tolerance.powi(2) {
        return None;
    }
    let time = -dot(value, rate) / denominator;
    if time <= 0.0 || norm(add(value, scale(rate, time))) > tolerance {
        return None;
    }
    Some(time)
}

pub(super) fn closing_duration(value: Vec3, rate: Vec3, requested: f64) -> f64 {
    let magnitude = norm(value);
    let rate_magnitude = norm(rate);
    if magnitude == 0.0 || rate_magnitude == 0.0 || dot(value, rate) >= 0.0 {
        return requested;
    }
    requested.min(0.25 * magnitude / rate_magnitude)
}

pub(super) fn path_distance(velocity: Vec3, acceleration: Vec3, duration: f64) -> f64 {
    let final_velocity = add(velocity, scale(acceleration, duration));
    let midpoint = add(velocity, scale(acceleration, 0.5 * duration));
    if norm(cross(velocity, final_velocity)) <= 1.0e-12 {
        return 0.5 * (norm(velocity) + norm(final_velocity)) * duration;
    }
    duration * (norm(velocity) + 4.0 * norm(midpoint) + norm(final_velocity)) / 6.0
}

pub(super) fn kinetic_energy(state: State, body: Body) -> f64 {
    0.5 * body.mass * dot(state.velocity, state.velocity)
        + 0.5 * body.inertia() * dot(state.spin, state.spin)
}

fn normalized_vector(value: Vec3) -> MathResult<Vec3> {
    Ok([
        canonical(value[0])?,
        canonical(value[1])?,
        canonical(value[2])?,
    ])
}

fn canonical(value: f64) -> MathResult<f64> {
    normalize_f64(value).map_err(|_| RuntimeMathError::CanonicalRange)
}
