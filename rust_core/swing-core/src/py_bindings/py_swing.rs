//! PyO3 bindings for `swing::pendulum` and `swing::plane`.
//!
//! Registered as the runtime submodule `swing` on the `swing_core`
//! extension module. Python consumers must import it via attribute access
//! (`from swing_core import swing`), never `import swing_core.swing`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::swing::pendulum::{self, PendulumParameters, PendulumState};
use crate::swing::plane;

#[pymethods]
impl PendulumParameters {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
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
    ) -> PyResult<Self> {
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
        params.validate().map_err(PyValueError::new_err)?;
        Ok(params)
    }

    /// Default golf-swing parameters (UpstreamDrift reference constants).
    #[staticmethod]
    fn golf_default_py() -> Self {
        Self::golf_default()
    }

    fn __repr__(&self) -> String {
        format!(
            "PendulumParameters(m1={}, l1={}, lc1={}, i1={}, m2={}, l2={}, lc2={}, i2={}, d1={}, d2={})",
            self.m1, self.l1, self.lc1, self.i1, self.m2, self.l2, self.lc2, self.i2, self.d1, self.d2
        )
    }
}

#[pymethods]
impl PendulumState {
    #[new]
    fn py_new(theta1: f64, theta2: f64, omega1: f64, omega2: f64) -> Self {
        Self {
            theta1,
            theta2,
            omega1,
            omega2,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PendulumState(theta1={}, theta2={}, omega1={}, omega2={})",
            self.theta1, self.theta2, self.omega1, self.omega2
        )
    }
}

/// Plane rotation matrix as a row-major nested list `[[f64; 3]; 3]`.
#[pyfunction]
fn plane_rotation(yaw: f64, side_tilt: f64, fwd_tilt: f64) -> [[f64; 3]; 3] {
    let r = plane::plane_rotation(yaw, side_tilt, fwd_tilt);
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = r[(i, j)];
        }
    }
    out
}

/// In-plane gravity `(g_x, g_y)` from the three tilt angles and `g` [m/s²].
#[pyfunction]
fn in_plane_gravity(yaw: f64, side_tilt: f64, fwd_tilt: f64, g: f64) -> (f64, f64) {
    plane::in_plane_gravity_from_tilts(yaw, side_tilt, fwd_tilt, g)
}

/// Symmetric 2x2 mass matrix as nested list.
#[pyfunction]
fn mass_matrix(params: &PendulumParameters, theta2: f64) -> [[f64; 2]; 2] {
    pendulum::mass_matrix(params, theta2)
}

/// One RK4 step of size `dt` under in-plane gravity `(gx, gy)`.
#[pyfunction]
fn step(
    params: &PendulumParameters,
    state: &PendulumState,
    gx: f64,
    gy: f64,
    dt: f64,
) -> PyResult<PendulumState> {
    pendulum::rk4_step(params, state, (gx, gy), dt).map_err(PyValueError::new_err)
}

/// Simulate `n_steps` RK4 steps.
///
/// Returns a flat row-major vector of `(n_steps + 1) * 4` floats:
/// `[theta1, theta2, omega1, omega2]` per state, initial state included.
/// The Python façade reshapes this into an `(n+1, 4)` NumPy array.
#[pyfunction]
fn simulate(
    params: &PendulumParameters,
    initial: &PendulumState,
    gx: f64,
    gy: f64,
    dt: f64,
    n_steps: usize,
) -> PyResult<Vec<f64>> {
    let states = pendulum::simulate(params, initial, (gx, gy), dt, n_steps)
        .map_err(PyValueError::new_err)?;
    let mut out = Vec::with_capacity(states.len() * 4);
    for s in states {
        out.extend_from_slice(&[s.theta1, s.theta2, s.omega1, s.omega2]);
    }
    Ok(out)
}

/// Total mechanical energy [J] under in-plane gravity `(gx, gy)`.
#[pyfunction]
fn total_energy(params: &PendulumParameters, state: &PendulumState, gx: f64, gy: f64) -> f64 {
    pendulum::total_energy(params, state, (gx, gy))
}

/// Register the `swing` runtime submodule on the parent extension module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "swing")?;
    m.add_class::<PendulumParameters>()?;
    m.add_class::<PendulumState>()?;
    m.add_function(wrap_pyfunction!(plane_rotation, &m)?)?;
    m.add_function(wrap_pyfunction!(in_plane_gravity, &m)?)?;
    m.add_function(wrap_pyfunction!(mass_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(step, &m)?)?;
    m.add_function(wrap_pyfunction!(simulate, &m)?)?;
    m.add_function(wrap_pyfunction!(total_energy, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
