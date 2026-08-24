//! Double-pendulum swing dynamics.
//!
//! Port of UpstreamDrift's
//! `src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py`
//! (mass matrix, Coriolis/centripetal vector, gravity vector, viscous
//! damping, RK4 stepping), generalised so gravity enters as an **in-plane
//! 2-vector** computed from the swing-plane pose (see [`crate::swing::plane`])
//! instead of a projected scalar.
//!
//! # Angle convention
//! - `theta1`: angle of the upper segment measured from the in-plane
//!   downward vertical (positive counter-clockwise).
//! - `theta2`: angle of the lower segment **relative to** the upper segment.
//!
//! In-plane coordinates: local x = in-plane horizontal, local y = in-plane
//! up. A segment at angle `theta` from downward vertical points along
//! `(sin theta, -cos theta)`.
//!
//! # Design by Contract
//! - `PendulumParameters::validate` rejects non-positive masses/lengths and
//!   non-finite values.
//! - The mass matrix is symmetric positive-definite for valid parameters
//!   (unit tested).
//! - Undamped, unforced dynamics conserve total energy (unit tested:
//!   relative drift < 1e-6 over 1000 RK4 steps).

use serde::{Deserialize, Serialize};

/// Numerical tolerance for detecting singular mass matrices.
pub const MASS_MATRIX_SINGULAR_TOLERANCE: f64 = 1e-12;

// Defaults ported from UpstreamDrift double_pendulum.py (documented there).
const DEFAULT_ARM_LENGTH_M: f64 = 0.75;
const DEFAULT_ARM_MASS_KG: f64 = 7.5;
const DEFAULT_ARM_COM_RATIO: f64 = 0.45;
const DEFAULT_ARM_INERTIA_SCALING: f64 = 1.0 / 12.0;
const DEFAULT_SHAFT_LENGTH_M: f64 = 1.0;
const DEFAULT_SHAFT_MASS_KG: f64 = 0.15;
const DEFAULT_CLUBHEAD_MASS_KG: f64 = 0.20;
const DEFAULT_SHAFT_COM_RATIO: f64 = 0.43;
const DEFAULT_DAMPING_SHOULDER: f64 = 0.4;
const DEFAULT_DAMPING_WRIST: f64 = 0.25;

/// Physical parameters of the two-segment (shoulder + wrist) pendulum.
///
/// Inertias are about the **proximal joint** (parallel-axis already applied),
/// matching the cached quantities in the UpstreamDrift reference.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all, set_all))]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct PendulumParameters {
    /// Upper segment mass [kg].
    pub m1: f64,
    /// Upper segment length [m].
    pub l1: f64,
    /// Upper segment COM distance from shoulder [m].
    pub lc1: f64,
    /// Upper segment inertia about the shoulder [kg·m²].
    pub i1: f64,
    /// Lower segment (shaft + clubhead) total mass [kg].
    pub m2: f64,
    /// Lower segment length [m].
    pub l2: f64,
    /// Lower segment COM distance from wrist [m].
    pub lc2: f64,
    /// Lower segment inertia about the wrist [kg·m²].
    pub i2: f64,
    /// Viscous damping at the shoulder [N·m·s/rad].
    pub d1: f64,
    /// Viscous damping at the wrist [N·m·s/rad].
    pub d2: f64,
}

impl PendulumParameters {
    /// Default golf-swing parameters, computed from the same segment
    /// constants as the UpstreamDrift reference (arm rod + shaft/clubhead
    /// composite). Kept as formulas — not hard-coded decimals — so the
    /// Python reference and this crate agree bit-for-bit.
    #[must_use]
    pub fn golf_default() -> Self {
        let m1 = DEFAULT_ARM_MASS_KG;
        let l1 = DEFAULT_ARM_LENGTH_M;
        let lc1 = l1 * DEFAULT_ARM_COM_RATIO;
        let i1_com = DEFAULT_ARM_INERTIA_SCALING * m1 * l1 * l1;
        let i1 = i1_com + m1 * lc1 * lc1;

        let l2 = DEFAULT_SHAFT_LENGTH_M;
        let ms = DEFAULT_SHAFT_MASS_KG;
        let mh = DEFAULT_CLUBHEAD_MASS_KG;
        let m2 = ms + mh;
        let shaft_com = l2 * DEFAULT_SHAFT_COM_RATIO;
        let lc2 = (shaft_com * ms + l2 * mh) / m2;
        let shaft_inertia_com = (1.0 / 12.0) * ms * l2 * l2;
        let parallel_axis =
            ms * (shaft_com - lc2) * (shaft_com - lc2) + mh * (l2 - lc2) * (l2 - lc2);
        let i2_com = shaft_inertia_com + parallel_axis;
        let i2 = i2_com + m2 * lc2 * lc2;

        Self {
            m1,
            l1,
            lc1,
            i1,
            m2,
            l2,
            lc2,
            i2,
            d1: DEFAULT_DAMPING_SHOULDER,
            d2: DEFAULT_DAMPING_WRIST,
        }
    }

    /// Validate physical plausibility.
    ///
    /// # Errors
    /// Returns a description of the first violated precondition.
    pub fn validate(&self) -> Result<(), String> {
        let fields = [
            ("m1", self.m1),
            ("l1", self.l1),
            ("lc1", self.lc1),
            ("i1", self.i1),
            ("m2", self.m2),
            ("l2", self.l2),
            ("lc2", self.lc2),
            ("i2", self.i2),
        ];
        for (name, value) in fields {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and > 0, got {value}"));
            }
        }
        for (name, value) in [("d1", self.d1), ("d2", self.d2)] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and >= 0, got {value}"));
            }
        }
        Ok(())
    }
}

/// Dynamic state of the pendulum (planar).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(get_all, set_all))]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct PendulumState {
    /// Upper segment angle from in-plane downward vertical [rad].
    pub theta1: f64,
    /// Lower segment angle relative to the upper segment [rad].
    pub theta2: f64,
    /// Upper segment angular velocity [rad/s].
    pub omega1: f64,
    /// Lower segment angular velocity [rad/s].
    pub omega2: f64,
}

/// Compute the symmetric 2x2 mass matrix `[[m11, m12], [m12, m22]]`.
#[must_use]
pub fn mass_matrix(p: &PendulumParameters, theta2: f64) -> [[f64; 2]; 2] {
    debug_assert!(theta2.is_finite(), "mass_matrix: theta2 must be finite");
    let cos_theta2 = theta2.cos();
    let m11 = p.i1 + p.i2 + p.m2 * p.l1 * p.l1 + 2.0 * p.m2 * p.l1 * p.lc2 * cos_theta2;
    let m12 = p.i2 + p.m2 * p.l1 * p.lc2 * cos_theta2;
    let m22 = p.i2;
    [[m11, m12], [m12, m22]]
}

/// Coriolis and centripetal generalized-force vector.
#[must_use]
pub fn coriolis_vector(p: &PendulumParameters, theta2: f64, omega1: f64, omega2: f64) -> [f64; 2] {
    let h = -p.m2 * p.l1 * p.lc2 * theta2.sin();
    let c1 = h * (2.0 * omega1 * omega2 + omega2 * omega2);
    let c2 = -h * omega1 * omega1;
    [c1, c2]
}

/// Gravitational generalized-force vector for an in-plane gravity 2-vector.
///
/// `g_inplane = (gx, gy)` are the gravity components along the plane's local
/// horizontal and local up axes (see [`crate::swing::plane::in_plane_gravity`]).
/// For the flat plane `(0, -g)` this reduces exactly to the classic scalar
/// form `g1 = (m1·lc1 + m2·l1)·g·sin θ1 + m2·lc2·g·sin(θ1+θ2)`.
#[must_use]
pub fn gravity_vector(
    p: &PendulumParameters,
    theta1: f64,
    theta2: f64,
    g_inplane: (f64, f64),
) -> [f64; 2] {
    let (gx, gy) = g_inplane;
    let t12 = theta1 + theta2;
    // Segment direction derivative: d/dθ (sin θ, -cos θ) = (cos θ, sin θ).
    // Generalized gravity torque Q_i = Σ_k m_k g_vec · ∂p_k/∂q_i; the EOM
    // uses G = -Q (restoring convention, matching the scalar reference).
    let a1 = p.m1 * p.lc1 + p.m2 * p.l1;
    let a2 = p.m2 * p.lc2;
    let g1 = -a1 * (gx * theta1.cos() + gy * theta1.sin()) - a2 * (gx * t12.cos() + gy * t12.sin());
    let g2 = -a2 * (gx * t12.cos() + gy * t12.sin());
    [g1, g2]
}

/// Viscous damping torques.
#[must_use]
pub fn damping_vector(p: &PendulumParameters, omega1: f64, omega2: f64) -> [f64; 2] {
    [p.d1 * omega1, p.d2 * omega2]
}

fn invert_mass_matrix(p: &PendulumParameters, theta2: f64) -> Result<[[f64; 2]; 2], String> {
    let m = mass_matrix(p, theta2);
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() <= MASS_MATRIX_SINGULAR_TOLERANCE {
        return Err("mass matrix determinant too close to zero; check pendulum parameters".into());
    }
    Ok([
        [m[1][1] / det, -m[0][1] / det],
        [-m[1][0] / det, m[0][0] / det],
    ])
}

/// State derivatives `(dθ1, dθ2, dω1, dω2)` for unforced dynamics.
///
/// # Errors
/// Returns an error if the mass matrix is numerically singular.
pub fn derivatives(
    p: &PendulumParameters,
    state: &PendulumState,
    g_inplane: (f64, f64),
) -> Result<[f64; 4], String> {
    let [c1, c2] = coriolis_vector(p, state.theta2, state.omega1, state.omega2);
    let [g1, g2] = gravity_vector(p, state.theta1, state.theta2, g_inplane);
    let [d1, d2] = damping_vector(p, state.omega1, state.omega2);
    let inv_m = invert_mass_matrix(p, state.theta2)?;
    let rhs1 = -(c1 + g1 + d1);
    let rhs2 = -(c2 + g2 + d2);
    let acc1 = inv_m[0][0] * rhs1 + inv_m[0][1] * rhs2;
    let acc2 = inv_m[1][0] * rhs1 + inv_m[1][1] * rhs2;
    Ok([state.omega1, state.omega2, acc1, acc2])
}

/// Advance the state by one classical RK4 step of size `dt`.
///
/// # Errors
/// Returns an error if the mass matrix is numerically singular at any stage.
pub fn rk4_step(
    p: &PendulumParameters,
    state: &PendulumState,
    g_inplane: (f64, f64),
    dt: f64,
) -> Result<PendulumState, String> {
    debug_assert!(
        dt.is_finite() && dt > 0.0,
        "rk4_step: dt must be finite and > 0"
    );
    let y = [state.theta1, state.theta2, state.omega1, state.omega2];
    let f = |v: [f64; 4]| -> Result<[f64; 4], String> {
        derivatives(
            p,
            &PendulumState {
                theta1: v[0],
                theta2: v[1],
                omega1: v[2],
                omega2: v[3],
            },
            g_inplane,
        )
    };
    let add = |a: [f64; 4], s: f64, b: [f64; 4]| -> [f64; 4] {
        [
            a[0] + s * b[0],
            a[1] + s * b[1],
            a[2] + s * b[2],
            a[3] + s * b[3],
        ]
    };
    let k1 = f(y)?;
    let k2 = f(add(y, dt / 2.0, k1))?;
    let k3 = f(add(y, dt / 2.0, k2))?;
    let k4 = f(add(y, dt, k3))?;
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = y[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    Ok(PendulumState {
        theta1: out[0],
        theta2: out[1],
        omega1: out[2],
        omega2: out[3],
    })
}

/// Total mechanical energy (kinetic + gravitational potential) [J].
///
/// Potential is `-Σ m_k g_vec · p_k` with COM positions measured from the
/// shoulder pivot in in-plane coordinates. Used by conservation tests.
#[must_use]
pub fn total_energy(p: &PendulumParameters, state: &PendulumState, g_inplane: (f64, f64)) -> f64 {
    let m = mass_matrix(p, state.theta2);
    let q = [state.omega1, state.omega2];
    let kinetic =
        0.5 * (m[0][0] * q[0] * q[0] + 2.0 * m[0][1] * q[0] * q[1] + m[1][1] * q[1] * q[1]);

    let (gx, gy) = g_inplane;
    let dir = |theta: f64| (theta.sin(), -theta.cos());
    let (e1x, e1y) = dir(state.theta1);
    let (e2x, e2y) = dir(state.theta1 + state.theta2);
    let p1 = (p.lc1 * e1x, p.lc1 * e1y);
    let p2 = (p.l1 * e1x + p.lc2 * e2x, p.l1 * e1y + p.lc2 * e2y);
    let potential = -(p.m1 * (gx * p1.0 + gy * p1.1) + p.m2 * (gx * p2.0 + gy * p2.1));
    kinetic + potential
}

/// Simulate `n_steps` RK4 steps, returning `n_steps + 1` states (including
/// the initial state).
///
/// # Errors
/// Returns an error if any step encounters a singular mass matrix.
pub fn simulate(
    p: &PendulumParameters,
    initial: &PendulumState,
    g_inplane: (f64, f64),
    dt: f64,
    n_steps: usize,
) -> Result<Vec<PendulumState>, String> {
    p.validate()?;
    let mut states = Vec::with_capacity(n_steps + 1);
    states.push(*initial);
    let mut current = *initial;
    for _ in 0..n_steps {
        current = rk4_step(p, &current, g_inplane, dt)?;
        states.push(current);
    }
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swing::plane::in_plane_gravity_from_tilts;

    const G: f64 = 9.80665;

    fn undamped_params() -> PendulumParameters {
        PendulumParameters {
            d1: 0.0,
            d2: 0.0,
            ..PendulumParameters::golf_default()
        }
    }

    #[test]
    fn golf_default_is_valid() {
        assert!(PendulumParameters::golf_default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_nonpositive_mass() {
        let p = PendulumParameters {
            m1: 0.0,
            ..PendulumParameters::golf_default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn mass_matrix_is_symmetric_positive_definite() {
        let p = PendulumParameters::golf_default();
        for theta2 in [-3.0, -1.5, 0.0, 0.8, 1.6, 3.1] {
            let m = mass_matrix(&p, theta2);
            assert!((m[0][1] - m[1][0]).abs() < 1e-15, "asymmetric at {theta2}");
            // Sylvester's criterion for 2x2 SPD.
            assert!(m[0][0] > 0.0, "m11 not positive at {theta2}");
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            assert!(det > 0.0, "determinant not positive at {theta2}: {det}");
        }
    }

    #[test]
    fn gravity_vector_flat_plane_matches_scalar_reference() {
        let p = PendulumParameters::golf_default();
        let (theta1, theta2) = (0.7, -0.4);
        let [g1, g2] = gravity_vector(&p, theta1, theta2, (0.0, -G));
        let expected_g1 = (p.m1 * p.lc1 + p.m2 * p.l1) * G * theta1.sin()
            + p.m2 * p.lc2 * G * (theta1 + theta2).sin();
        let expected_g2 = p.m2 * p.lc2 * G * (theta1 + theta2).sin();
        assert!((g1 - expected_g1).abs() < 1e-12, "g1 {g1} != {expected_g1}");
        assert!((g2 - expected_g2).abs() < 1e-12, "g2 {g2} != {expected_g2}");
    }

    #[test]
    fn undamped_unforced_energy_conserved_over_1000_steps() {
        let p = undamped_params();
        let g = in_plane_gravity_from_tilts(0.4, 0.6, -0.2, G);
        let initial = PendulumState {
            theta1: 1.2,
            theta2: -0.5,
            omega1: 0.0,
            omega2: 0.0,
        };
        let dt = 1e-4;
        let states = simulate(&p, &initial, g, dt, 1000).expect("simulation must succeed");
        let e0 = total_energy(&p, &initial, g);
        let scale = e0.abs().max(1.0);
        for (i, s) in states.iter().enumerate() {
            let e = total_energy(&p, s, g);
            let drift = (e - e0).abs() / scale;
            assert!(drift < 1e-6, "energy drift {drift} at step {i}");
        }
    }

    #[test]
    fn damping_dissipates_energy() {
        let p = PendulumParameters::golf_default();
        let g = (0.0, -G);
        let initial = PendulumState {
            theta1: 1.0,
            theta2: 0.3,
            omega1: 0.5,
            omega2: -0.2,
        };
        let states = simulate(&p, &initial, g, 1e-3, 2000).expect("simulation must succeed");
        let e0 = total_energy(&p, &initial, g);
        let e_end = total_energy(&p, states.last().expect("non-empty"), g);
        assert!(e_end < e0, "damped energy must decrease: {e_end} >= {e0}");
    }

    #[test]
    fn simulate_returns_n_plus_one_states() {
        let p = PendulumParameters::golf_default();
        let s0 = PendulumState {
            theta1: 0.1,
            theta2: 0.0,
            omega1: 0.0,
            omega2: 0.0,
        };
        let states = simulate(&p, &s0, (0.0, -G), 1e-3, 10).expect("simulation must succeed");
        assert_eq!(states.len(), 11);
        assert_eq!(states[0], s0);
    }
}
