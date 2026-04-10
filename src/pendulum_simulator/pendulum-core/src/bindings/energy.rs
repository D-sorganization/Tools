//! FFI wrappers for energy-transfer quantities: power, work, and impulse
//! time series derived from torques/forces and motion.

#![allow(dead_code)]

#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    #[pyfunction]
    pub fn py_angular_power_series(torques: Vec<f64>, angular_velocities: Vec<f64>) -> Vec<f64> {
        crate::dynamics::angular_power_series(&torques, &angular_velocities)
    }

    #[pyfunction]
    pub fn py_linear_power_series(forces: Vec<f64>, velocities: Vec<f64>) -> Vec<f64> {
        crate::dynamics::linear_power_series(&forces, &velocities)
    }

    #[pyfunction]
    pub fn py_angular_work_series(
        torques: Vec<f64>,
        angular_velocities: Vec<f64>,
        time: Vec<f64>,
    ) -> Vec<f64> {
        crate::dynamics::angular_work_series(&torques, &angular_velocities, &time)
    }

    #[pyfunction]
    pub fn py_linear_work_series(
        forces: Vec<f64>,
        velocities: Vec<f64>,
        time: Vec<f64>,
    ) -> Vec<f64> {
        crate::dynamics::linear_work_series(&forces, &velocities, &time)
    }

    #[pyfunction]
    pub fn py_angular_impulse_series(torques: Vec<f64>, time: Vec<f64>) -> Vec<f64> {
        crate::dynamics::angular_impulse_series(&torques, &time)
    }

    #[pyfunction]
    pub fn py_linear_impulse_series(forces: Vec<f64>, time: Vec<f64>) -> Vec<f64> {
        crate::dynamics::linear_impulse_series(&forces, &time)
    }
}
