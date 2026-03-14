//! # tools-core — Shared Simulation Kernel
//!
//! This crate provides the canonical implementations of mathematical primitives,
//! physics solvers, and thermodynamic calculators used across all AffineDrift
//! repositories. It compiles to:

pub mod atmosphere;
pub mod ball_flight;
pub mod math;
// Re-export primary types
pub use math::{clamp, lerp, GRAVITY, R_GAS};
pub use math_primitives::matrix3::Matrix3;
pub use math_primitives::quaternion::Quaternion;
pub use math_primitives::types::Vector3;
// Re-export submodules (used by benchmarks)
pub use math_primitives::{matrix3, quaternion, types};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn tools_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<math_primitives::types::Vector3>()?;
    m.add_class::<math_primitives::quaternion::Quaternion>()?;
    m.add_class::<math_primitives::matrix3::Matrix3>()?;
    m.add_class::<ball_flight::BallProperties>()?;
    m.add_class::<ball_flight::LaunchConditions>()?;
    m.add_class::<ball_flight::EnvironmentalConditions>()?;
    m.add_class::<ball_flight::TrajectoryPoint>()?;
    m.add_class::<ball_flight::TrajectoryAnalysis>()?;
    m.add_class::<atmosphere::AtmosphereProperties>()?;
    m.add_function(wrap_pyfunction!(math::py_lerp, m)?)?;
    m.add_function(wrap_pyfunction!(math::py_clamp, m)?)?;
    m.add_function(wrap_pyfunction!(math::py_deg_to_rad, m)?)?;
    m.add_function(wrap_pyfunction!(math::py_rad_to_deg, m)?)?;

    // Register math_primitives submodule (rotation, quaternion, geometry, numpy bindings)
    math_primitives::py_bindings::py_math::register_module(m)?;

    Ok(())
}
