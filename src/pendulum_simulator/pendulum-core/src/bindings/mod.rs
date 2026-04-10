//! FFI bindings (Python via PyO3, JavaScript via wasm-bindgen).
//!
//! The monolithic `lib.rs` FFI layer has been decomposed into focused
//! submodules mirroring the issue's prescribed physics split:
//!
//! - [`state`] — parameter and result wrapper structs
//! - [`dynamics`] — mass matrix, gravity, Coriolis, friction, equations of
//!   motion, forward kinematics
//! - [`integrator`] — full-trajectory simulation wrappers
//! - [`forces`] — constraint forces, Jacobians, ellipsoids, CMA-ES
//! - [`energy`] — power / work / impulse series
//!
//! Each submodule exposes `python` and/or `wasm` inner modules that are
//! compiled only when the corresponding feature is enabled.

pub mod dynamics;
pub mod energy;
pub mod forces;
pub mod integrator;
pub mod state;

// ---------------------------------------------------------------------------
// Python module registration
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod python_init {
    //! PyO3 module registration. The exported `pendulum_core` function is
    //! discovered by PyO3 through its `#[pymodule]` attribute regardless of
    //! its crate path.

    use crate::bindings::dynamics::python as dyn_py;
    use crate::bindings::energy::python as energy_py;
    use crate::bindings::forces::python as forces_py;
    use crate::bindings::integrator::python as integ_py;
    use crate::bindings::state::python::{
        PyCmaEsConfig, PyCmaEsResult, PyDoublePendulumParams, PyGolferParams,
        PyTriplePendulumParams,
    };
    use pyo3::prelude::*;

    /// Module initialization
    #[pymodule]
    pub fn pendulum_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyDoublePendulumParams>()?;
        m.add_class::<PyTriplePendulumParams>()?;
        m.add_class::<PyGolferParams>()?;
        m.add_class::<PyCmaEsConfig>()?;
        m.add_class::<PyCmaEsResult>()?;
        // Double pendulum dynamics
        m.add_function(wrap_pyfunction!(dyn_py::py_double_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_double_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_double_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_double_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_double_equations_of_motion, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_double_forward_kinematics, m)?)?;
        // Triple pendulum dynamics
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_equations_of_motion, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_triple_forward_kinematics, m)?)?;
        // Golfer dynamics
        m.add_function(wrap_pyfunction!(dyn_py::py_golfer_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_golfer_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_golfer_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(dyn_py::py_golfer_forward_kinematics, m)?)?;
        // Golfer constraint forces
        m.add_function(wrap_pyfunction!(forces_py::py_golfer_constraint_vector, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_golfer_constraint_jacobian, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_golfer_constrained_dynamics, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_golfer_project_to_constraints, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_golfer_project_velocity, m)?)?;
        // Integrators / simulation
        m.add_function(wrap_pyfunction!(integ_py::py_batch_evaluate_double, m)?)?;
        m.add_function(wrap_pyfunction!(integ_py::py_simulate_double, m)?)?;
        m.add_function(wrap_pyfunction!(integ_py::py_simulate_triple, m)?)?;
        m.add_function(wrap_pyfunction!(integ_py::py_simulate_golfer, m)?)?;
        // CMA-ES
        m.add_function(wrap_pyfunction!(forces_py::py_cmaes_optimize_torque, m)?)?;
        // Jacobians & ellipsoids
        m.add_function(wrap_pyfunction!(forces_py::py_double_jacobians, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_double_ellipsoids, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_triple_jacobians, m)?)?;
        m.add_function(wrap_pyfunction!(forces_py::py_triple_ellipsoids, m)?)?;
        // Energy / power / work / impulse series
        m.add_function(wrap_pyfunction!(energy_py::py_angular_power_series, m)?)?;
        m.add_function(wrap_pyfunction!(energy_py::py_linear_power_series, m)?)?;
        m.add_function(wrap_pyfunction!(energy_py::py_angular_work_series, m)?)?;
        m.add_function(wrap_pyfunction!(energy_py::py_linear_work_series, m)?)?;
        m.add_function(wrap_pyfunction!(energy_py::py_angular_impulse_series, m)?)?;
        m.add_function(wrap_pyfunction!(energy_py::py_linear_impulse_series, m)?)?;
        Ok(())
    }
}
