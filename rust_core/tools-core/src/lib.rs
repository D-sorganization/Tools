//! # tools-core — Shared Simulation Kernel
//!
//! This crate provides the canonical implementations of mathematical primitives,
//! physics solvers, and thermodynamic calculators used across all AffineDrift
//! repositories. It compiles to:

pub mod atmosphere;
pub mod ball_flight;
#[cfg(feature = "python")]
pub mod electrode_advisor;
pub mod engineering;
pub mod math;
pub mod reactor;
pub mod rrt;
#[cfg(feature = "python")]
pub mod scada;
pub mod signal;
pub mod swing_plane;
pub mod thermodynamics;
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
    m.add_class::<rrt::Obstacle>()?;
    m.add_class::<rrt::RRTPlanner>()?;
    m.add_function(wrap_pyfunction!(ball_flight::py_simulate_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(ball_flight::py_analyze_trajectory, m)?)?;
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
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_bilateral_filter,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_moving_average,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_exponential_smoothing,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_lms_filter,
        &signal_mod
    )?)?;
    signal_mod.add_function(wrap_pyfunction!(
        signal::py_bindings::py_rls_filter,
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

    // Register thermodynamics classes
    m.add_class::<thermodynamics::py_bindings::PyNasa7Species>()?;

    // Register reactor classes
    m.add_class::<reactor::py_bindings::PyTrc1DSystem>()?;

    // Register electrode advisor classes
    let electrode_mod = PyModule::new(m.py(), "electrode_advisor")?;
    electrode_advisor::py_bindings::register_module(&electrode_mod)?;
    m.add_submodule(&electrode_mod)?;

    // Register SCADA classes and functions
    let scada_mod = PyModule::new(m.py(), "scada")?;
    scada_mod.add_class::<scada::AlarmState>()?;
    scada_mod.add_class::<scada::TagLimits>()?;
    scada_mod.add_class::<scada::AlarmEngine>()?;
    scada_mod.add_class::<scada::InterlockMatrix>()?;
    scada_mod.add_class::<scada::GasificationSimulator>()?;
    scada_mod.add_function(wrap_pyfunction!(scada::py_moving_average, &scada_mod)?)?;
    scada_mod.add_function(wrap_pyfunction!(
        scada::py_exponential_smoothing,
        &scada_mod
    )?)?;
    scada_mod.add_function(wrap_pyfunction!(scada::py_savitzky_golay, &scada_mod)?)?;
    m.add_submodule(&scada_mod)?;

    Ok(())
}
