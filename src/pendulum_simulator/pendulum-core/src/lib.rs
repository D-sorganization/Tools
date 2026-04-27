//! Pendulum Core: A shared physics kernel for pendulum golf swing simulators.
//!
//! This library provides high-performance physics implementations for:
//! - Double pendulum (2-DOF)
//! - Triple pendulum (3-DOF)
//! - Golfer upper body (8-DOF with 4 constraints)
//!
//! Compiled as a native library (PyO3 for Python FFI) and WASM (for web apps).
//!
//! # Features
//!
//! - `python`: Compile with PyO3 bindings for Python FFI
//! - `wasm`: Compile with wasm-bindgen for WASM targets
//! - `serde`: Enable serialization support via serde
//!
//! # Module layout
//!
//! The Rust API is composed of per-model physics modules ([`double`],
//! [`triple`], [`golfer`], [`golfer_constraints`]), shared numerical
//! infrastructure ([`integrator`], [`jacobians`], [`cmaes`], [`batch`]),
//! and cross-cutting helpers ([`dynamics`], [`types`]).
//!
//! The FFI layer lives under [`bindings`] and is decomposed into focused
//! submodules (`state`, `dynamics`, `integrator`, `forces`, `energy`).

pub mod batch;
pub mod bindings;
pub mod cmaes;
pub mod double;
pub mod dynamics;
pub mod golfer;
pub mod golfer_constraints;
pub mod integrator;
pub mod jacobians;
pub mod triple;
pub mod types;

pub use double::{
    coriolis as double_coriolis, equations_of_motion as double_equations_of_motion,
    forward_kinematics as double_forward_kinematics, friction_torque as double_friction_torque,
    gravity_vector as double_gravity_vector, jacobian_club_tip, jacobian_wrist,
    mass_matrix as double_mass_matrix,
};
pub use triple::{
    coriolis as triple_coriolis, equations_of_motion as triple_equations_of_motion,
    forward_kinematics as triple_forward_kinematics, friction_torque as triple_friction_torque,
    gravity_vector as triple_gravity_vector, jacobian_joint1, jacobian_joint2, jacobian_joint3,
    mass_matrix as triple_mass_matrix,
};
pub use golfer::{
    analytical_fk_jacobians, constraint_jacobian, constraint_vector,
    forward_kinematics as golfer_forward_kinematics, friction_torque as golfer_friction_torque,
    gravity_vector as golfer_gravity_vector, mass_matrix as golfer_mass_matrix,
};
pub use golfer_constraints::{
    constrained_accelerations, constraint_acceleration_bias, project_to_constraints,
    project_velocity, BaumgarteGains,
};
pub use integrator::{
    integrate_double_pendulum, integrate_golfer, integrate_triple_pendulum, RK45Config,
};
pub use cmaes::{optimize, optimize_torque_coefficients, CmaEsConfig, CmaEsResult};
pub use jacobians::{
    ellipsoids_double, ellipsoids_triple, jacobian_double as jacobian_double_analytical,
    jacobian_triple as jacobian_triple_analytical, EllipsoidResult,
};
pub use dynamics::{
    angular_impulse_series, angular_power_series, angular_work_series, linear_impulse_series,
    linear_power_series, linear_work_series,
};
pub use types::{
    DoubleFKResult, DoublePendulumParams, GolferFKResult, GolferParams, TripleFKResult,
    TriplePendulumParams, Vec2,
};
pub use nalgebra::{SMatrix, SVector};

// FFI re-exports: preserve the pre-refactor `crate::py_bindings` /
// `crate::wasm_bindings` module paths for backward compatibility.
#[cfg(feature = "python")]
pub use bindings::python_init as py_bindings;

#[cfg(feature = "wasm")]
pub mod wasm_bindings {
    //! WASM bindings re-exported from the `bindings` submodules.
    pub use crate::bindings::dynamics::wasm::*;
    pub use crate::bindings::forces::wasm::*;
    pub use crate::bindings::state::wasm::*;
}
