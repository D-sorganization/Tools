//! # tools-core — Shared Simulation Kernel
//!
//! This crate provides the canonical implementations of mathematical primitives,
//! physics solvers, and thermodynamic calculators used across all AffineDrift
//! repositories. It compiles to:

pub mod atmosphere;
pub mod ball_flight;
pub mod engineering;
pub mod math;
pub mod signal;
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

    // Register signal submodule
    let signal_mod = PyModule::new(m.py(), "signal")?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_sinusoid,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_cosine,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_exponential,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_linear,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(signal::py_bindings::py_step, &signal_mod)?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_square,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_triangle,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_chirp,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_polynomial,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_pulse,
        &signal_mod
    )?)?;
    m.add_submodule(&signal_mod)?;

    // Register engineering submodule
    let eng_mod = PyModule::new(m.py(), "engineering")?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_reynolds_number,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_churchill_friction_factor,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_darcy_weisbach,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_ideal_gas_density,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_isentropic_work,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_lmtd,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_celsius_to_kelvin,
        &eng_mod
    )?)?;
    eng_mod.add_function(wrap_pyfunction!(
        engineering::py_bindings::py_kelvin_to_celsius,
        &eng_mod
    )?)?;
    m.add_submodule(&eng_mod)?;

    Ok(())
}
