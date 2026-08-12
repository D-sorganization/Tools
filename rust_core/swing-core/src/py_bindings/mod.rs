//! Python (PyO3) bindings, split by concern.
//!
//! Only compiled with the `python` feature. Each submodule mirrors one
//! domain module under `swing/` and registers a runtime PyO3 submodule on
//! the `swing_core` extension module.

pub mod py_swing;
