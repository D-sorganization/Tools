use math_primitives::types::Vector3;

use super::{
    qualify_ground_transfer, FlightGroundConfig, FlightGroundRun, FlightState,
    GroundTransferOutcome, TransferUnavailableReason,
};
use crate::ball_flight::{
    apply_spin_decay, rk4_step, BallProperties, EnvironmentalConditions, LaunchConditions,
    MIN_SPEED_THRESHOLD, SPIN_DECAY_RATE,
};

const MAX_STEPS: usize = 1_000_000;

/// Simulate request-relative flight from explicit ground/tee geometry to transfer.
pub fn simulate_flight_to_ground(
    ball: &BallProperties,
    env: &EnvironmentalConditions,
    launch: &LaunchConditions,
    config: &FlightGroundConfig,
) -> Result<FlightGroundRun, TransferUnavailableReason> {
    validate_simulation(ball, launch, config)?;
    let context = IntegrationContext::new(ball, env, launch)?;
    let request_ground = config.launch_geometry.request_ground(ball, &config.ground);
    let mut state = initial_state(launch, context.spin_axis);
    let mut trajectory = vec![state];
    for _ in 0..MAX_STEPS {
        if state.time >= config.max_time {
            break;
        }
        let next_time = (state.time + config.dt).min(config.max_time);
        state = context.advance(state, next_time - state.time);
        state.time = next_time;
        trajectory.push(state);
        let pair = &trajectory[trajectory.len() - 2..];
        let outcome = qualify_ground_transfer(pair, &request_ground, ball.radius());
        if is_terminal(&outcome) {
            replace_terminal_sample(&mut trajectory, &outcome);
            return Ok(FlightGroundRun {
                trajectory,
                outcome,
            });
        }
        if state.velocity.magnitude() < MIN_SPEED_THRESHOLD {
            break;
        }
    }
    let outcome = qualify_ground_transfer(&trajectory, &request_ground, ball.radius());
    Ok(FlightGroundRun {
        trajectory,
        outcome,
    })
}

fn replace_terminal_sample(trajectory: &mut [FlightState], outcome: &GroundTransferOutcome) {
    let GroundTransferOutcome::Contact(event) = outcome else {
        return;
    };
    if let Some(last) = trajectory.last_mut() {
        *last = event.contact;
    }
}

fn is_terminal(outcome: &GroundTransferOutcome) -> bool {
    matches!(
        outcome,
        GroundTransferOutcome::Contact(_)
            | GroundTransferOutcome::Grazing { .. }
            | GroundTransferOutcome::Unavailable(_)
    )
}

fn validate_simulation(
    ball: &BallProperties,
    launch: &LaunchConditions,
    config: &FlightGroundConfig,
) -> Result<(), TransferUnavailableReason> {
    let valid_time = config.dt.is_finite()
        && config.dt > 0.0
        && config.max_time.is_finite()
        && config.max_time > 0.0
        && config.max_time <= config.dt * MAX_STEPS as f64;
    let valid_ball = ball.radius().is_finite() && ball.radius() > 0.0 && ball.mass > 0.0;
    if !valid_time
        || !valid_ball
        || !config.ground.is_valid()
        || !config.launch_geometry.is_valid()
        || !launch.velocity.is_finite()
        || launch.velocity <= 0.0
    {
        return Err(TransferUnavailableReason::InvalidSimulationConfiguration);
    }
    Ok(())
}

fn initial_state(launch: &LaunchConditions, spin_axis: Vector3) -> FlightState {
    let angle = launch.launch_angle.to_radians();
    let azimuth = launch.azimuth_angle.to_radians();
    let velocity = Vector3::new(
        launch.velocity * angle.cos() * azimuth.cos(),
        launch.velocity * angle.sin(),
        launch.velocity * angle.cos() * azimuth.sin(),
    );
    let omega = launch.spin_rate * 2.0 * std::f64::consts::PI / 60.0;
    FlightState::new(0.0, Vector3::zero(), velocity, spin_axis * omega)
}

struct IntegrationContext {
    gravity: Vector3,
    wind: Vector3,
    radius: f64,
    force_scale: f64,
    coeffs: (f64, f64, f64, f64, f64, f64),
    spin_axis: Vector3,
}

impl IntegrationContext {
    fn new(
        ball: &BallProperties,
        env: &EnvironmentalConditions,
        launch: &LaunchConditions,
    ) -> Result<Self, TransferUnavailableReason> {
        if !environment_is_finite(env) || !launch_is_finite(launch) {
            return Err(TransferUnavailableReason::InvalidSimulationConfiguration);
        }
        let spin_axis = normalized_spin_axis(launch)?;
        Ok(Self {
            gravity: Vector3::new(0.0, -env.gravity, 0.0),
            wind: env.wind_velocity,
            radius: ball.radius(),
            force_scale: 0.5 * env.air_density * ball.cross_sectional_area() / ball.mass,
            coeffs: (ball.cd0, ball.cd1, ball.cd2, ball.cl0, ball.cl1, ball.cl2),
            spin_axis,
        })
    }

    fn advance(&self, state: FlightState, dt: f64) -> FlightState {
        let raw = [
            state.position.x,
            state.position.y,
            state.position.z,
            state.velocity.x,
            state.velocity.y,
            state.velocity.z,
        ];
        let omega = state.angular_velocity.dot(&self.spin_axis);
        let delta = rk4_step(
            &raw,
            dt,
            &self.gravity,
            &self.wind,
            self.radius,
            self.force_scale,
            &self.coeffs,
            omega,
            &self.spin_axis,
        );
        let next: [f64; 6] = std::array::from_fn(|index| raw[index] + delta[index]);
        FlightState::new(
            state.time + dt,
            Vector3::new(next[0], next[1], next[2]),
            Vector3::new(next[3], next[4], next[5]),
            self.spin_axis * apply_spin_decay(omega, SPIN_DECAY_RATE, dt),
        )
    }
}

fn environment_is_finite(env: &EnvironmentalConditions) -> bool {
    env.gravity.is_finite()
        && env.air_density.is_finite()
        && env.air_density > 0.0
        && env.wind_velocity.x.is_finite()
        && env.wind_velocity.y.is_finite()
        && env.wind_velocity.z.is_finite()
}

fn launch_is_finite(launch: &LaunchConditions) -> bool {
    launch.launch_angle.is_finite()
        && launch.azimuth_angle.is_finite()
        && launch.spin_rate.is_finite()
        && launch.spin_rate >= 0.0
        && launch.spin_axis.x.is_finite()
        && launch.spin_axis.y.is_finite()
        && launch.spin_axis.z.is_finite()
}

fn normalized_spin_axis(launch: &LaunchConditions) -> Result<Vector3, TransferUnavailableReason> {
    if launch.spin_rate.abs() <= f64::EPSILON {
        return Ok(Vector3::new(0.0, 1.0, 0.0));
    }
    launch
        .spin_axis
        .normalized()
        .map_err(|_| TransferUnavailableReason::InvalidSpinAxis)
}
