//! wasm-bindgen bindings for `swing::pendulum` and `swing::plane`.
//!
//! Flat `Vec<f64>` payloads cross the JS boundary as `Float64Array`
//! (row-major); the web app reshapes them. Struct fields are `f64` and Copy,
//! so wasm-bindgen auto-generates getters/setters.

use wasm_bindgen::prelude::*;

use crate::swing::pendulum::{self, PendulumParameters, PendulumState};
use crate::swing::plane;

#[wasm_bindgen]
impl PendulumParameters {
    /// Construct parameters; throws on physically invalid values.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn wasm_new(
        m1: f64,
        l1: f64,
        lc1: f64,
        i1: f64,
        m2: f64,
        l2: f64,
        lc2: f64,
        i2: f64,
        d1: f64,
        d2: f64,
    ) -> Result<PendulumParameters, JsError> {
        let params = Self {
            m1,
            l1,
            lc1,
            i1,
            m2,
            l2,
            lc2,
            i2,
            d1,
            d2,
        };
        params.validate().map_err(|e| JsError::new(&e))?;
        Ok(params)
    }

    /// Default golf-swing parameters (UpstreamDrift reference constants).
    #[wasm_bindgen(js_name = "golfDefault")]
    pub fn wasm_golf_default() -> PendulumParameters {
        Self::golf_default()
    }
}

#[wasm_bindgen]
impl PendulumState {
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(theta1: f64, theta2: f64, omega1: f64, omega2: f64) -> PendulumState {
        Self {
            theta1,
            theta2,
            omega1,
            omega2,
        }
    }
}

/// Plane rotation matrix as a row-major `Float64Array` of length 9.
#[wasm_bindgen(js_name = "planeRotation")]
pub fn wasm_plane_rotation(yaw: f64, side_tilt: f64, fwd_tilt: f64) -> Vec<f64> {
    let r = plane::plane_rotation(yaw, side_tilt, fwd_tilt);
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| r[(i, j)])
        .collect()
}

/// In-plane gravity `[g_x, g_y]` from the three tilt angles and `g` [m/s²].
#[wasm_bindgen(js_name = "inPlaneGravity")]
pub fn wasm_in_plane_gravity(yaw: f64, side_tilt: f64, fwd_tilt: f64, g: f64) -> Vec<f64> {
    let (gx, gy) = plane::in_plane_gravity_from_tilts(yaw, side_tilt, fwd_tilt, g);
    vec![gx, gy]
}

/// One RK4 step; throws on a singular mass matrix.
#[wasm_bindgen(js_name = "step")]
pub fn wasm_step(
    params: &PendulumParameters,
    state: &PendulumState,
    gx: f64,
    gy: f64,
    dt: f64,
) -> Result<PendulumState, JsError> {
    pendulum::rk4_step(params, state, (gx, gy), dt).map_err(|e| JsError::new(&e))
}

/// Simulate `n_steps` RK4 steps.
///
/// Returns a row-major `Float64Array` of `(n_steps + 1) * 4` floats:
/// `[theta1, theta2, omega1, omega2]` per state, initial state included.
#[wasm_bindgen(js_name = "simulate")]
pub fn wasm_simulate(
    params: &PendulumParameters,
    initial: &PendulumState,
    gx: f64,
    gy: f64,
    dt: f64,
    n_steps: usize,
) -> Result<Vec<f64>, JsError> {
    let states =
        pendulum::simulate(params, initial, (gx, gy), dt, n_steps).map_err(|e| JsError::new(&e))?;
    let mut out = Vec::with_capacity(states.len() * 4);
    for s in states {
        out.extend_from_slice(&[s.theta1, s.theta2, s.omega1, s.omega2]);
    }
    Ok(out)
}

/// Total mechanical energy [J] under in-plane gravity `(gx, gy)`.
#[wasm_bindgen(js_name = "totalEnergy")]
pub fn wasm_total_energy(
    params: &PendulumParameters,
    state: &PendulumState,
    gx: f64,
    gy: f64,
) -> f64 {
    pendulum::total_energy(params, state, (gx, gy))
}
