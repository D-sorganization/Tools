//! # swing-core — Swing Simulation Kernel
//!
//! Canonical Rust implementation of the swing simulation physics shared by the
//! PyQt6 desktop app (via PyO3 wheel `swing_core`) and the web app (via
//! wasm-pack NPM package). Single-source physics: Python and WASM consumers
//! call these exact functions so the two UIs cannot drift.
//!
//! Phase 0 (issue #4104) walking skeleton:
//! - `swing::pendulum` — driven double-pendulum equations of motion ported
//!   from UpstreamDrift's `double_pendulum.py` (mass matrix, Coriolis,
//!   gravity, damping, RK4).
//! - `swing::plane` — swing-plane orientation (three sequential tilts) and
//!   in-plane gravity projection; the EOM consumes the projected 2-vector.
//!
//! # Architecture
//! - Core physics is plain Rust (no FFI types) in `swing/`.
//! - Python-specific glue lives in `py_bindings/` (feature `python`).
//! - WASM-specific glue lives in `wasm_bindings/` (feature `wasm`).

pub mod swing;

#[cfg(feature = "python")]
pub mod py_bindings;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

// Re-export primary types at the crate root for Rust consumers.
pub use swing::pendulum::{PendulumParameters, PendulumState};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module `swing_core`.
///
/// Registers submodules only; all classes/functions hang off runtime PyO3
/// submodules (e.g. `swing`). Note for Python consumers: runtime submodules
/// are attributes, not filesystem modules — use
/// `from swing_core import swing`, never `import swing_core.swing`.
#[cfg(feature = "python")]
#[pymodule]
fn swing_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_bindings::py_swing::register_module(m)?;
    Ok(())
}
