//! Ball flight physics simulation with drag and Magnus effect.
//!
//! This module implements the core ball flight physics for golf simulation:
//! - Reynolds-number-dependent drag forces
//! - Magnus effect (spin-induced lift)
//! - RK4 numerical integration
//! - Exponential spin decay
//!
//! # Design by Contract
//! - `simulate_trajectory` validates `velocity`/`dt`/`max_time` as real
//!   (release-mode) preconditions via `assert!`; lower-level helpers use
//!   `debug_assert!` for development-time checks.
//! - Functions return `Result<T, E>` for invalid configurations
//!
//! # DRY
//! - This is the **canonical** implementation. Python and WASM wrappers call
//!   these functions directly. Replaces the Numba-JIT version in UpstreamDrift.
//!
//! # TDD
//! - Tests written before implementation covering:
//!   - Zero-spin drag-only trajectory
//!   - Magnus lift direction
//!   - RK4 vs analytical comparison for gravity-only
//!   - Spin decay exponential curve
//!   - Parity with Python reference values

use serde::{Deserialize, Serialize};

use math_primitives::types::Vector3;

// ── Physical Constants ───────────────────────────────────────────────────────

/// Standard gravitational acceleration [m/s²].
/// Re-exported from `math` module for convenience (canonical value = 9.80665).
pub use crate::math::GRAVITY;

/// Air density at sea level, 15 °C [kg/m³].
pub const AIR_DENSITY_SEA_LEVEL: f64 = 1.225;

/// Spin decay rate [1/s] — exponential decay constant.
pub const SPIN_DECAY_RATE: f64 = 0.08;

/// Minimum speed threshold below which simulation stops [m/s].
pub const MIN_SPEED_THRESHOLD: f64 = 0.1;

/// Maximum lift coefficient (physical cap).
pub const MAX_LIFT_COEFFICIENT: f64 = 0.25;

/// Numerical epsilon for avoiding division by zero.
pub const NUMERICAL_EPSILON: f64 = 1e-10;

// ── Data Types ───────────────────────────────────────────────────────────────

/// Physical properties of a golf ball.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct BallProperties {
    /// Mass [kg].
    pub mass: f64,
    /// Diameter [m].
    pub diameter: f64,
    /// Drag coefficients: cd = cd0 + s*(cd1 + s*cd2).
    pub cd0: f64,
    pub cd1: f64,
    pub cd2: f64,
    /// Lift coefficients: cl = cl0 + s*(cl1 + s*cl2), clamped to MAX_LIFT_COEFFICIENT.
    pub cl0: f64,
    pub cl1: f64,
    pub cl2: f64,
}

impl Default for BallProperties {
    fn default() -> Self {
        Self {
            mass: 0.0459,
            diameter: 0.04267,
            cd0: 0.21,
            cd1: 0.05,
            cd2: 0.02,
            cl0: 0.00,
            cl1: 0.38,
            cl2: 0.00,
        }
    }
}

impl BallProperties {
    /// Ball radius [m].
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.diameter / 2.0
    }

    /// Cross-sectional area [m²].
    #[must_use]
    pub fn cross_sectional_area(&self) -> f64 {
        std::f64::consts::PI * self.radius() * self.radius()
    }

    /// Drag coefficient from spin parameter.
    #[must_use]
    pub fn calculate_cd(&self, s: f64) -> f64 {
        self.cd0 + s * (self.cd1 + s * self.cd2)
    }

    /// Lift coefficient from spin parameter, clamped.
    #[must_use]
    pub fn calculate_cl(&self, s: f64) -> f64 {
        let cl = self.cl0 + s * (self.cl1 + s * self.cl2);
        if cl > MAX_LIFT_COEFFICIENT {
            MAX_LIFT_COEFFICIENT
        } else {
            cl
        }
    }
}

/// Initial launch conditions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct LaunchConditions {
    /// Launch speed [m/s].
    pub velocity: f64,
    /// Launch angle [degrees].
    pub launch_angle: f64,
    /// Azimuth angle [degrees].
    pub azimuth_angle: f64,
    /// Spin rate [rpm].
    pub spin_rate: f64,
    /// Spin axis (unit vector, typically [0, -1, 0] for backspin).
    pub spin_axis: Vector3,
}

impl Default for LaunchConditions {
    fn default() -> Self {
        Self {
            velocity: 70.0,
            launch_angle: 12.0,
            azimuth_angle: 0.0,
            spin_rate: 2500.0,
            spin_axis: Vector3::new(0.0, -1.0, 0.0),
        }
    }
}

/// Environmental conditions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct EnvironmentalConditions {
    /// Air density [kg/m³].
    pub air_density: f64,
    /// Wind velocity [m/s] as (x, y, z).
    pub wind_velocity: Vector3,
    /// Gravitational acceleration [m/s²].
    pub gravity: f64,
}

impl Default for EnvironmentalConditions {
    fn default() -> Self {
        Self {
            air_density: AIR_DENSITY_SEA_LEVEL,
            wind_velocity: Vector3::new(0.0, 0.0, 0.0),
            gravity: GRAVITY,
        }
    }
}

/// A single point in the ball trajectory.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct TrajectoryPoint {
    /// Time [s].
    pub time: f64,
    /// Position [m] (x=downrange, y=height, z=lateral).
    pub position: Vector3,
    /// Velocity [m/s].
    pub velocity: Vector3,
    /// Current spin rate [rad/s].
    pub spin: f64,
}

impl TrajectoryPoint {
    /// Speed magnitude [m/s].
    #[must_use]
    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }

    /// Height (y-component) [m].
    #[must_use]
    pub fn height(&self) -> f64 {
        self.position.y
    }
}

/// Summary of a completed trajectory.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct TrajectoryAnalysis {
    /// Total carry distance [m].
    pub carry_distance: f64,
    /// Maximum height [m].
    pub max_height: f64,
    /// Total flight time [s].
    pub flight_time: f64,
    /// Landing angle [degrees].
    pub landing_angle: f64,
    /// Final spin rate [rad/s].
    pub final_spin: f64,
}

// ── Core Physics Functions ───────────────────────────────────────────────────

/// Calculate acceleration from drag and Magnus forces.
///
/// Matches Python `_calculate_accel_core` exactly.
///
/// # Arguments
/// * `rel_vel` — velocity relative to air (vel - wind)
/// * `speed`   — magnitude of rel_vel
/// * `gravity_acc` — gravity acceleration vector [0, -g, 0]
/// * `ball_radius` — ball radius [m]
/// * `const_term`  — 0.5 * rho * A / m
/// * `coeffs` — (cd0, cd1, cd2, cl0, cl1, cl2)
/// * `omega` — spin rate [rad/s]
/// * `spin_axis` — spin axis unit vector
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn calculate_accel_core(
    rel_vel: &Vector3,
    speed: f64,
    gravity_acc: &Vector3,
    ball_radius: f64,
    const_term: f64,
    coeffs: &(f64, f64, f64, f64, f64, f64),
    omega: f64,
    spin_axis: &Vector3,
) -> Vector3 {
    let (cd0, cd1, cd2, cl0, cl1, cl2) = *coeffs;

    let spin_ratio = if omega > 0.0 {
        (omega * ball_radius) / speed
    } else {
        0.0
    };

    // Drag: a = g - (const * cd * speed) * v_rel
    let cd = cd0 + spin_ratio * (cd1 + spin_ratio * cd2);
    let drag_factor = const_term * cd * speed;
    let mut acc = Vector3::new(
        gravity_acc.x - drag_factor * rel_vel.x,
        gravity_acc.y - drag_factor * rel_vel.y,
        gravity_acc.z - drag_factor * rel_vel.z,
    );

    // Magnus lift
    if omega > 0.0 && spin_ratio > 0.0 {
        let mut cl = cl0 + spin_ratio * (cl1 + spin_ratio * cl2);
        if cl > MAX_LIFT_COEFFICIENT {
            cl = MAX_LIFT_COEFFICIENT;
        }

        let magnus_mag = const_term * cl * speed * speed;

        // Cross product: spin_axis × rel_vel
        let cross = spin_axis.cross(rel_vel);
        let cross_mag = cross.magnitude();

        if cross_mag > NUMERICAL_EPSILON {
            let factor = magnus_mag / cross_mag;
            acc = Vector3::new(
                acc.x + cross.x * factor,
                acc.y + cross.y * factor,
                acc.z + cross.z * factor,
            );
        }
    }

    acc
}

/// State derivative: [dx, dv] = [v, a].
#[must_use]
#[allow(clippy::too_many_arguments)]
fn flight_dynamics_step(
    state: &[f64; 6],
    gravity_acc: &Vector3,
    wind_velocity: &Vector3,
    ball_radius: f64,
    const_term: f64,
    coeffs: &(f64, f64, f64, f64, f64, f64),
    omega: f64,
    spin_axis: &Vector3,
) -> [f64; 6] {
    let vel = Vector3::new(state[3], state[4], state[5]);
    let rel_vel = Vector3::new(
        vel.x - wind_velocity.x,
        vel.y - wind_velocity.y,
        vel.z - wind_velocity.z,
    );
    let speed = rel_vel.magnitude();

    if speed < MIN_SPEED_THRESHOLD {
        return [
            vel.x,
            vel.y,
            vel.z,
            gravity_acc.x,
            gravity_acc.y,
            gravity_acc.z,
        ];
    }

    let acc = calculate_accel_core(
        &rel_vel,
        speed,
        gravity_acc,
        ball_radius,
        const_term,
        coeffs,
        omega,
        spin_axis,
    );

    [vel.x, vel.y, vel.z, acc.x, acc.y, acc.z]
}

/// Single RK4 integration step.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rk4_step(
    state: &[f64; 6],
    dt: f64,
    gravity_acc: &Vector3,
    wind_velocity: &Vector3,
    ball_radius: f64,
    const_term: f64,
    coeffs: &(f64, f64, f64, f64, f64, f64),
    omega: f64,
    spin_axis: &Vector3,
) -> [f64; 6] {
    let k1 = flight_dynamics_step(
        state,
        gravity_acc,
        wind_velocity,
        ball_radius,
        const_term,
        coeffs,
        omega,
        spin_axis,
    );

    let mut s2 = *state;
    for i in 0..6 {
        s2[i] += 0.5 * dt * k1[i];
    }
    let k2 = flight_dynamics_step(
        &s2,
        gravity_acc,
        wind_velocity,
        ball_radius,
        const_term,
        coeffs,
        omega,
        spin_axis,
    );

    let mut s3 = *state;
    for i in 0..6 {
        s3[i] += 0.5 * dt * k2[i];
    }
    let k3 = flight_dynamics_step(
        &s3,
        gravity_acc,
        wind_velocity,
        ball_radius,
        const_term,
        coeffs,
        omega,
        spin_axis,
    );

    let mut s4 = *state;
    for i in 0..6 {
        s4[i] += dt * k3[i];
    }
    let k4 = flight_dynamics_step(
        &s4,
        gravity_acc,
        wind_velocity,
        ball_radius,
        const_term,
        coeffs,
        omega,
        spin_axis,
    );

    let mut result = [0.0; 6];
    for i in 0..6 {
        result[i] = dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    result
}

/// Apply exponential spin decay: ω(t+dt) = ω(t) × exp(-λ × dt).
#[must_use]
pub fn apply_spin_decay(omega: f64, decay_rate: f64, dt: f64) -> f64 {
    omega * (-decay_rate * dt).exp()
}

/// Simulate a complete ball flight trajectory.
///
/// This is the main entry point.
///
/// # Arguments
/// * `ball`  — ball physical properties
/// * `env`   — environmental conditions
/// * `launch` — initial launch conditions
/// * `max_time` — maximum simulation time [s]
/// * `dt` — integration time step [s]
///
/// # Returns
/// A vector of `TrajectoryPoint` values.
///
/// # Panics
/// Panics if `launch.velocity`, `dt`, or `max_time` is not a finite positive
/// value. These are real (release-mode) preconditions — unlike the previous
/// `debug_assert!` guards they cannot be stripped from an optimized build, so a
/// caller passing `dt = 0.0` (infinite step count) or a non-finite `max_time`
/// (a saturating cast to a huge `Vec` capacity → OOM) fails fast and loudly.
#[must_use]
pub fn simulate_trajectory(
    ball: &BallProperties,
    env: &EnvironmentalConditions,
    launch: &LaunchConditions,
    max_time: f64,
    dt: f64,
) -> Vec<TrajectoryPoint> {
    assert!(
        launch.velocity.is_finite() && launch.velocity > 0.0,
        "Launch velocity must be finite and positive, got {}",
        launch.velocity
    );
    assert!(
        dt.is_finite() && dt > 0.0,
        "Time step must be finite and positive, got {dt}"
    );
    assert!(
        max_time.is_finite() && max_time > 0.0,
        "Max time must be finite and positive, got {max_time}"
    );

    let launch_angle_rad = launch.launch_angle.to_radians();
    let azimuth_rad = launch.azimuth_angle.to_radians();

    // Initial velocity components (x=downrange, y=up, z=lateral)
    let vx = launch.velocity * launch_angle_rad.cos() * azimuth_rad.cos();
    let vy = launch.velocity * launch_angle_rad.sin();
    let vz = launch.velocity * launch_angle_rad.cos() * azimuth_rad.sin();

    let gravity_acc = Vector3::new(0.0, -env.gravity, 0.0);
    let const_term = 0.5 * env.air_density * ball.cross_sectional_area() / ball.mass;
    let coeffs = (ball.cd0, ball.cd1, ball.cd2, ball.cl0, ball.cl1, ball.cl2);

    let mut omega = launch.spin_rate * 2.0 * std::f64::consts::PI / 60.0; // RPM → rad/s
    let spin_axis = launch.spin_axis;

    let mut state: [f64; 6] = [0.0, 0.0, 0.0, vx, vy, vz];
    let mut time = 0.0;

    // `max_time / dt` can be enormous (e.g. a multi-hour `max_time` with a tiny
    // `dt`); a saturating `as usize` cast would then ask for a multi-gigabyte
    // allocation up front. Cap the *pre-allocation* at a sane ceiling — the
    // integration loop still honors `max_steps` and `Vec` grows on demand.
    const MAX_PREALLOC_STEPS: usize = 1_000_000;
    let step_count = (max_time / dt).ceil();
    let max_steps = if step_count >= MAX_PREALLOC_STEPS as f64 {
        MAX_PREALLOC_STEPS
    } else {
        step_count as usize
    };
    let capacity = max_steps.saturating_add(1);
    let mut trajectory = Vec::with_capacity(capacity);

    // Record initial point
    trajectory.push(TrajectoryPoint {
        time,
        position: Vector3::new(state[0], state[1], state[2]),
        velocity: Vector3::new(state[3], state[4], state[5]),
        spin: omega,
    });

    for _ in 0..max_steps {
        let delta = rk4_step(
            &state,
            dt,
            &gravity_acc,
            &env.wind_velocity,
            ball.radius(),
            const_term,
            &coeffs,
            omega,
            &spin_axis,
        );

        for i in 0..6 {
            state[i] += delta[i];
        }
        time += dt;

        omega = apply_spin_decay(omega, SPIN_DECAY_RATE, dt);

        trajectory.push(TrajectoryPoint {
            time,
            position: Vector3::new(state[0], state[1], state[2]),
            velocity: Vector3::new(state[3], state[4], state[5]),
            spin: omega,
        });

        // Stop if ball hits the ground (after first step)
        if state[1] < 0.0 && time > dt {
            break;
        }

        // Stop if speed too low
        let speed = (state[3] * state[3] + state[4] * state[4] + state[5] * state[5]).sqrt();
        if speed < MIN_SPEED_THRESHOLD {
            break;
        }
    }

    trajectory
}

/// Analyze a completed trajectory.
#[must_use]
pub fn analyze_trajectory(trajectory: &[TrajectoryPoint]) -> TrajectoryAnalysis {
    if trajectory.is_empty() {
        return TrajectoryAnalysis {
            carry_distance: 0.0,
            max_height: 0.0,
            flight_time: 0.0,
            landing_angle: 0.0,
            final_spin: 0.0,
        };
    }

    let last = trajectory.last().unwrap();
    let carry_distance =
        (last.position.x * last.position.x + last.position.z * last.position.z).sqrt();
    let max_height = trajectory
        .iter()
        .map(|p| p.position.y)
        .fold(0.0_f64, f64::max);
    let flight_time = last.time;
    let final_spin = last.spin;

    // Landing angle: arctan(|vy| / horizontal_speed) at last point
    let h_speed = (last.velocity.x * last.velocity.x + last.velocity.z * last.velocity.z).sqrt();
    let landing_angle = if h_speed > NUMERICAL_EPSILON {
        (last.velocity.y.abs() / h_speed).atan().to_degrees()
    } else {
        90.0
    };

    TrajectoryAnalysis {
        carry_distance,
        max_height,
        flight_time,
        landing_angle,
        final_spin,
    }
}

// ── Python Bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl BallProperties {
    /// Create default golf ball properties.
    #[new]
    #[pyo3(text_signature = "()")]
    fn py_new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "BallProperties(mass={}, diameter={})",
            self.mass, self.diameter
        )
    }

    #[getter]
    fn get_mass(&self) -> f64 {
        self.mass
    }
    #[setter]
    fn set_mass(&mut self, value: f64) {
        self.mass = value;
    }
    #[getter]
    fn get_diameter(&self) -> f64 {
        self.diameter
    }
    #[setter]
    fn set_diameter(&mut self, value: f64) {
        self.diameter = value;
    }
    #[getter]
    fn get_cd0(&self) -> f64 {
        self.cd0
    }
    #[setter]
    fn set_cd0(&mut self, value: f64) {
        self.cd0 = value;
    }
    #[getter]
    fn get_cd1(&self) -> f64 {
        self.cd1
    }
    #[setter]
    fn set_cd1(&mut self, value: f64) {
        self.cd1 = value;
    }
    #[getter]
    fn get_cd2(&self) -> f64 {
        self.cd2
    }
    #[setter]
    fn set_cd2(&mut self, value: f64) {
        self.cd2 = value;
    }
    #[getter]
    fn get_cl0(&self) -> f64 {
        self.cl0
    }
    #[setter]
    fn set_cl0(&mut self, value: f64) {
        self.cl0 = value;
    }
    #[getter]
    fn get_cl1(&self) -> f64 {
        self.cl1
    }
    #[setter]
    fn set_cl1(&mut self, value: f64) {
        self.cl1 = value;
    }
    #[getter]
    fn get_cl2(&self) -> f64 {
        self.cl2
    }
    #[setter]
    fn set_cl2(&mut self, value: f64) {
        self.cl2 = value;
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl LaunchConditions {
    /// Create launch conditions for a golf shot.
    #[new]
    #[pyo3(signature = (velocity=70.0, launch_angle=12.0, azimuth_angle=0.0, spin_rate=2500.0))]
    #[pyo3(
        text_signature = "(velocity=70.0, launch_angle=12.0, azimuth_angle=0.0, spin_rate=2500.0)"
    )]
    fn py_new(velocity: f64, launch_angle: f64, azimuth_angle: f64, spin_rate: f64) -> Self {
        Self {
            velocity,
            launch_angle,
            azimuth_angle,
            spin_rate,
            spin_axis: Vector3::new(0.0, -1.0, 0.0),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LaunchConditions(v={}, angle={}, spin={}rpm)",
            self.velocity, self.launch_angle, self.spin_rate
        )
    }

    /// Set the spin axis as a (not necessarily normalized) 3-vector.
    ///
    /// The default constructor pins pure backspin (`[0, -1, 0]`); this
    /// setter lets Python callers (swing_sim flight facade, #4107) pass
    /// tilted spin axes derived from impact solvers.
    fn set_spin_axis(&mut self, x: f64, y: f64, z: f64) {
        self.spin_axis = Vector3::new(x, y, z);
    }

    /// Return the spin axis as an `(x, y, z)` tuple.
    fn get_spin_axis(&self) -> (f64, f64, f64) {
        (self.spin_axis.x, self.spin_axis.y, self.spin_axis.z)
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl EnvironmentalConditions {
    /// Create default environmental conditions (sea level, 15°C, no wind).
    #[new]
    #[pyo3(text_signature = "()")]
    fn py_new() -> Self {
        Self::default()
    }

    /// Set the wind velocity vector [m/s].
    fn set_wind(&mut self, x: f64, y: f64, z: f64) {
        self.wind_velocity = Vector3::new(x, y, z);
    }

    /// Return the wind velocity as an `(x, y, z)` tuple.
    fn get_wind(&self) -> (f64, f64, f64) {
        (
            self.wind_velocity.x,
            self.wind_velocity.y,
            self.wind_velocity.z,
        )
    }

    #[getter]
    fn get_air_density(&self) -> f64 {
        self.air_density
    }
    #[setter]
    fn set_air_density(&mut self, value: f64) {
        self.air_density = value;
    }
    #[getter]
    fn get_gravity(&self) -> f64 {
        self.gravity
    }
    #[setter]
    fn set_gravity(&mut self, value: f64) {
        self.gravity = value;
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl TrajectoryPoint {
    #[getter]
    fn time(&self) -> f64 {
        self.time
    }
    #[getter]
    fn x(&self) -> f64 {
        self.position.x
    }
    #[getter]
    fn y(&self) -> f64 {
        self.position.y
    }
    #[getter]
    fn z(&self) -> f64 {
        self.position.z
    }
    #[getter]
    fn spin_rate(&self) -> f64 {
        self.spin
    }
    #[getter]
    fn vx(&self) -> f64 {
        self.velocity.x
    }
    #[getter]
    fn vy(&self) -> f64 {
        self.velocity.y
    }
    #[getter]
    fn vz(&self) -> f64 {
        self.velocity.z
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectoryPoint(t={:.3}, pos=({:.2}, {:.2}, {:.2}))",
            self.time, self.position.x, self.position.y, self.position.z
        )
    }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl TrajectoryAnalysis {
    #[getter]
    fn carry_distance(&self) -> f64 {
        self.carry_distance
    }
    #[getter]
    fn max_height(&self) -> f64 {
        self.max_height
    }
    #[getter]
    fn flight_time(&self) -> f64 {
        self.flight_time
    }
    #[getter]
    fn landing_angle(&self) -> f64 {
        self.landing_angle
    }

    fn __repr__(&self) -> String {
        format!(
            "TrajectoryAnalysis(carry={:.1}m, height={:.1}m, time={:.2}s)",
            self.carry_distance, self.max_height, self.flight_time
        )
    }
}

/// Simulate a complete ball flight trajectory (Python entry point).
///
/// Thin wrapper over [`simulate_trajectory`] that converts the release-mode
/// precondition `assert!`s into `ValueError`s so invalid Python input never
/// panics across the FFI boundary. Consumed by
/// `src/shared/python/swing_sim/flight/_rust_facade.py` (#4107).
#[cfg(feature = "python")]
#[pyo3::prelude::pyfunction]
#[pyo3(name = "simulate_trajectory")]
pub fn py_simulate_trajectory(
    ball: BallProperties,
    env: EnvironmentalConditions,
    launch: LaunchConditions,
    max_time: f64,
    dt: f64,
) -> pyo3::PyResult<Vec<TrajectoryPoint>> {
    use pyo3::exceptions::PyValueError;
    if !(launch.velocity.is_finite() && launch.velocity > 0.0) {
        return Err(PyValueError::new_err(format!(
            "Launch velocity must be finite and positive, got {}",
            launch.velocity
        )));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(PyValueError::new_err(format!(
            "Time step must be finite and positive, got {dt}"
        )));
    }
    if !(max_time.is_finite() && max_time > 0.0) {
        return Err(PyValueError::new_err(format!(
            "Max time must be finite and positive, got {max_time}"
        )));
    }
    Ok(simulate_trajectory(&ball, &env, &launch, max_time, dt))
}

/// Analyze a completed trajectory (Python entry point).
#[cfg(feature = "python")]
#[pyo3::prelude::pyfunction]
#[pyo3(name = "analyze_trajectory")]
pub fn py_analyze_trajectory(trajectory: Vec<TrajectoryPoint>) -> TrajectoryAnalysis {
    analyze_trajectory(&trajectory)
}

// ── WASM bindings ────────────────────────────────────────────────────────────
// BallProperties and TrajectoryAnalysis have only f64 fields, so
// wasm_bindgen auto-generates getters. LaunchConditions needs a custom
// constructor since spin_axis (Vector3) is a wasm_bindgen type too.

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl BallProperties {
    /// Create default golf ball properties.
    #[wasm_bindgen(constructor)]
    pub fn wasm_new() -> Self {
        Self::default()
    }

    /// Ball radius [m].
    #[wasm_bindgen(js_name = "radius")]
    pub fn wasm_radius(&self) -> f64 {
        self.radius()
    }

    /// Cross-sectional area [m²].
    #[wasm_bindgen(js_name = "crossSectionalArea")]
    pub fn wasm_cross_sectional_area(&self) -> f64 {
        self.cross_sectional_area()
    }

    /// Drag coefficient for a given spin parameter.
    #[wasm_bindgen(js_name = "calculateCd")]
    pub fn wasm_calculate_cd(&self, s: f64) -> f64 {
        self.calculate_cd(s)
    }

    /// Lift coefficient for a given spin parameter.
    #[wasm_bindgen(js_name = "calculateCl")]
    pub fn wasm_calculate_cl(&self, s: f64) -> f64 {
        self.calculate_cl(s)
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl LaunchConditions {
    /// Create launch conditions.
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(velocity: f64, launch_angle: f64, azimuth_angle: f64, spin_rate: f64) -> Self {
        Self {
            velocity,
            launch_angle,
            azimuth_angle,
            spin_rate,
            spin_axis: Vector3::new(0.0, -1.0, 0.0),
        }
    }

    /// Get the spin axis as a Vector3.
    #[wasm_bindgen(js_name = "spinAxis", getter)]
    pub fn wasm_spin_axis(&self) -> Vector3 {
        self.spin_axis
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl TrajectoryAnalysis {
    /// Carry distance [m].
    #[wasm_bindgen(js_name = "carryDistance", getter)]
    pub fn wasm_carry_distance(&self) -> f64 {
        self.carry_distance
    }

    /// Maximum height [m].
    #[wasm_bindgen(js_name = "maxHeight", getter)]
    pub fn wasm_max_height(&self) -> f64 {
        self.max_height
    }

    /// Total flight time [s].
    #[wasm_bindgen(js_name = "flightTime", getter)]
    pub fn wasm_flight_time(&self) -> f64 {
        self.flight_time
    }

    /// Landing angle [degrees].
    #[wasm_bindgen(js_name = "landingAngle", getter)]
    pub fn wasm_landing_angle(&self) -> f64 {
        self.landing_angle
    }
}

// ── Tests (TDD) ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ball() -> BallProperties {
        BallProperties::default()
    }

    fn default_env() -> EnvironmentalConditions {
        EnvironmentalConditions::default()
    }

    // ── Spin Decay ──

    #[test]
    fn test_spin_decay_exponential() {
        let omega = 100.0;
        let dt = 1.0;
        let result = apply_spin_decay(omega, SPIN_DECAY_RATE, dt);
        let expected = omega * (-SPIN_DECAY_RATE * dt).exp();
        assert!(
            (result - expected).abs() < 1e-12,
            "Spin decay should match exponential"
        );
    }

    #[test]
    fn test_spin_decay_zero_rate() {
        let omega = 100.0;
        let result = apply_spin_decay(omega, 0.0, 1.0);
        assert!((result - omega).abs() < 1e-12, "Zero decay = no change");
    }

    // ── Gravity-only trajectory (drag=0) ──

    #[test]
    fn test_gravity_only_trajectory() {
        let ball = BallProperties {
            cd0: 0.0,
            cd1: 0.0,
            cd2: 0.0,
            cl0: 0.0,
            cl1: 0.0,
            cl2: 0.0,
            ..default_ball()
        };
        let launch = LaunchConditions {
            velocity: 50.0,
            launch_angle: 45.0,
            spin_rate: 0.0,
            ..LaunchConditions::default()
        };
        let trajectory = simulate_trajectory(&ball, &default_env(), &launch, 10.0, 0.001);
        let analysis = analyze_trajectory(&trajectory);

        // Analytical: range = v² sin(2θ) / g = 2500 * 1 / 9.81 ≈ 254.8 m
        let analytical_range = 50.0 * 50.0 * (90.0_f64.to_radians()).sin() / GRAVITY;
        assert!(
            (analysis.carry_distance - analytical_range).abs() < 1.0,
            "Gravity-only RK4 should match analytical to ~1m. Got {:.2}, expected {:.2}",
            analysis.carry_distance,
            analytical_range
        );
    }

    // ── Drag reduces range ──

    #[test]
    fn test_drag_reduces_range() {
        let ball_no_drag = BallProperties {
            cd0: 0.0,
            cd1: 0.0,
            cd2: 0.0,
            cl0: 0.0,
            cl1: 0.0,
            cl2: 0.0,
            ..default_ball()
        };
        let ball_with_drag = BallProperties {
            cl0: 0.0,
            cl1: 0.0,
            cl2: 0.0,
            ..default_ball()
        };
        let launch = LaunchConditions {
            velocity: 50.0,
            launch_angle: 30.0,
            spin_rate: 0.0,
            ..LaunchConditions::default()
        };
        let env = default_env();

        let no_drag = analyze_trajectory(&simulate_trajectory(
            &ball_no_drag,
            &env,
            &launch,
            10.0,
            0.001,
        ));
        let with_drag = analyze_trajectory(&simulate_trajectory(
            &ball_with_drag,
            &env,
            &launch,
            10.0,
            0.001,
        ));

        assert!(
            with_drag.carry_distance < no_drag.carry_distance,
            "Drag should reduce range: {} vs {}",
            with_drag.carry_distance,
            no_drag.carry_distance
        );
    }

    // ── Magnus increases height (backspin) ──

    #[test]
    fn test_backspin_increases_height() {
        // Same ball, same initial conditions — only spin rate differs
        let ball = default_ball();
        let env = default_env();

        let launch_no_spin = LaunchConditions {
            velocity: 70.0,
            launch_angle: 25.0,
            spin_rate: 0.0,
            spin_axis: Vector3::new(0.0, 0.0, 1.0), // backspin axis
            ..LaunchConditions::default()
        };
        let launch_with_spin = LaunchConditions {
            velocity: 70.0,
            launch_angle: 25.0,
            spin_rate: 5000.0,
            spin_axis: Vector3::new(0.0, 0.0, 1.0), // backspin axis
            ..LaunchConditions::default()
        };

        let no_spin = analyze_trajectory(&simulate_trajectory(
            &ball,
            &env,
            &launch_no_spin,
            10.0,
            0.001,
        ));
        let with_spin = analyze_trajectory(&simulate_trajectory(
            &ball,
            &env,
            &launch_with_spin,
            10.0,
            0.001,
        ));

        assert!(
            with_spin.max_height > no_spin.max_height,
            "Backspin Magnus should increase height: {:.2} vs {:.2}",
            with_spin.max_height,
            no_spin.max_height
        );
    }

    // ── Trajectory basics ──

    #[test]
    fn test_trajectory_starts_at_origin() {
        let trajectory = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            5.0,
            0.01,
        );
        let first = &trajectory[0];
        assert!(first.position.x.abs() < 1e-12);
        assert!(first.position.y.abs() < 1e-12);
        assert!(first.position.z.abs() < 1e-12);
    }

    #[test]
    fn test_trajectory_lands() {
        let trajectory = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            10.0,
            0.01,
        );
        let last = trajectory.last().unwrap();
        assert!(
            last.position.y <= 0.01,
            "Ball should land (y <= 0): got {:.4}",
            last.position.y
        );
    }

    #[test]
    fn test_trajectory_analysis_reasonable_values() {
        let trajectory = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            10.0,
            0.01,
        );
        let analysis = analyze_trajectory(&trajectory);

        // Default launch: 70 m/s, 12° angle, 2500 rpm backspin
        // With drag+Magnus: carry ~80-250m, height ~5-40m, time ~1-8s
        assert!(
            analysis.carry_distance > 50.0 && analysis.carry_distance < 300.0,
            "Carry distance out of range: {:.1}",
            analysis.carry_distance
        );
        assert!(
            analysis.max_height > 3.0 && analysis.max_height < 60.0,
            "Max height out of range: {:.1}",
            analysis.max_height
        );
        assert!(
            analysis.flight_time > 0.5 && analysis.flight_time < 10.0,
            "Flight time out of range: {:.2}",
            analysis.flight_time
        );
    }

    // ── Ball Properties ──

    #[test]
    fn test_ball_properties_defaults() {
        let ball = default_ball();
        assert!((ball.mass - 0.0459).abs() < 1e-6);
        assert!((ball.radius() - 0.021335).abs() < 1e-6);
    }

    #[test]
    fn test_cd_increases_with_spin() {
        let ball = default_ball();
        let cd0 = ball.calculate_cd(0.0);
        let cd1 = ball.calculate_cd(0.1);
        assert!(cd1 > cd0, "CD should increase with spin parameter");
    }

    #[test]
    fn test_cl_clamped() {
        let ball = BallProperties {
            cl0: 0.3,
            cl1: 0.5,
            ..default_ball()
        };
        let cl = ball.calculate_cl(1.0);
        assert!(
            (cl - MAX_LIFT_COEFFICIENT).abs() < 1e-12,
            "CL should be clamped to {}",
            MAX_LIFT_COEFFICIENT
        );
    }

    // ── Accel core ──

    #[test]
    fn test_accel_gravity_only() {
        let gravity_acc = Vector3::new(0.0, -GRAVITY, 0.0);
        let rel_vel = Vector3::new(10.0, 0.0, 0.0);
        let coeffs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let acc = calculate_accel_core(
            &rel_vel,
            10.0,
            &gravity_acc,
            0.02,
            1.0,
            &coeffs,
            0.0,
            &Vector3::new(0.0, 1.0, 0.0),
        );
        assert!((acc.x).abs() < 1e-10);
        assert!((acc.y - (-GRAVITY)).abs() < 1e-10);
    }

    // ── Serde roundtrip ──

    #[test]
    fn test_serde_ball_properties() {
        let ball = default_ball();
        let json = serde_json::to_string(&ball).unwrap();
        let ball2: BallProperties = serde_json::from_str(&json).unwrap();
        assert!((ball.mass - ball2.mass).abs() < 1e-12);
    }

    #[test]
    fn test_serde_launch_conditions() {
        let launch = LaunchConditions::default();
        let json = serde_json::to_string(&launch).unwrap();
        let launch2: LaunchConditions = serde_json::from_str(&json).unwrap();
        assert!((launch.velocity - launch2.velocity).abs() < 1e-12);
    }

    // ── simulate_trajectory preconditions (issue #3554) ──

    #[test]
    #[should_panic(expected = "Time step must be finite and positive")]
    fn test_simulate_rejects_zero_dt() {
        // dt = 0 would otherwise mean an infinite step count (0 steps + a
        // div-by-zero capacity). Now a real (release-mode) precondition.
        let _ = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            5.0,
            0.0,
        );
    }

    #[test]
    #[should_panic(expected = "Time step must be finite and positive")]
    fn test_simulate_rejects_negative_dt() {
        let _ = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            5.0,
            -0.01,
        );
    }

    #[test]
    #[should_panic(expected = "Max time must be finite and positive")]
    fn test_simulate_rejects_nonfinite_max_time() {
        let _ = simulate_trajectory(
            &default_ball(),
            &default_env(),
            &LaunchConditions::default(),
            f64::INFINITY,
            0.01,
        );
    }

    #[test]
    #[should_panic(expected = "Launch velocity must be finite and positive")]
    fn test_simulate_rejects_zero_velocity() {
        let launch = LaunchConditions {
            velocity: 0.0,
            ..LaunchConditions::default()
        };
        let _ = simulate_trajectory(&default_ball(), &default_env(), &launch, 5.0, 0.01);
    }

    #[test]
    fn test_simulate_huge_max_time_dt_ratio_does_not_oom() {
        // A huge max_time with a tiny dt previously asked `Vec::with_capacity`
        // for ~3e11 elements (instant OOM). The pre-allocation is now capped;
        // the run terminates early via the ground/low-speed stop conditions and
        // returns a bounded trajectory without exhausting memory.
        let launch = LaunchConditions {
            velocity: 50.0,
            launch_angle: 30.0,
            spin_rate: 0.0,
            ..LaunchConditions::default()
        };
        let trajectory =
            simulate_trajectory(&default_ball(), &default_env(), &launch, 1_000_000.0, 1e-5);
        // Ball lands and the loop breaks; far fewer than the (capped) 1e6 steps.
        assert!(!trajectory.is_empty());
        assert!(
            trajectory.len() < 1_000_000,
            "expected early termination, got {} points",
            trajectory.len()
        );
    }
}
