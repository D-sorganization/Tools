//! # tools-core — Shared Simulation Kernel
//!
//! This crate provides the canonical implementations of mathematical primitives,
//! physics solvers, and thermodynamic calculators used across all AffineDrift
//! repositories. It compiles to:
//!
//! - A Python extension module (via PyO3 / Maturin)
//! - A WebAssembly module (via wasm-bindgen / wasm-pack)
//! - A native Rust library (for direct Rust consumers)
//!
//! ## Design Principles
//!
//! - **TDD**: Every public function has tests written before implementation.
//! - **DbC (Design by Contract)**: All public functions validate preconditions
//!   via `debug_assert!` and return `Result<T, E>` instead of panicking.
//! - **DRY**: Types and logic defined once here; Python/JS are thin wrappers.

pub mod math;
pub mod types;

// Re-export primary types at crate root for ergonomic imports.
pub use math::{clamp, lerp};
pub use types::Vector3;

// ── Python bindings (feature-gated) ──────────────────────────────────────────

/// Register the Python module when compiled with `--features python`.
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn tools_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::Vector3>()?;
    m.add_function(wrap_pyfunction!(math::py_lerp, m)?)?;
    m.add_function(wrap_pyfunction!(math::py_clamp, m)?)?;
    Ok(())
}
