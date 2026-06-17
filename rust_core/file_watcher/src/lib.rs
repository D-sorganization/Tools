//! Cross-platform file watcher built on `notify-rs`.
//!
//! Wraps `notify::RecommendedWatcher` with two value-adds the bare crate does
//! not provide:
//!
//! 1. **Debouncing** — rapid bursts of events (editor save → backup → atomic
//!    rename → temp-file create) are coalesced into a single flush after a
//!    configurable quiet period.
//! 2. **`.gitignore` filtering** — events for paths matched by any
//!    `.gitignore`, `.git/info/exclude`, or global excludes are dropped before
//!    they reach the consumer. This is what makes the watcher cheap enough to
//!    run on a full project tree (no flood from `node_modules/`, `target/`,
//!    `__pycache__/`, etc.).
//!
//! The pure-Rust `FileWatcher` is testable via `cargo test`. PyO3 bindings live
//! behind the `python` feature gate and are activated by maturin when building
//! the production wheel.
pub mod watcher;

#[cfg(feature = "python")]
mod python;

pub use watcher::{ChangeEvent, ChangeKind, FileWatcher, FileWatcherConfig, WatcherError};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// PyO3 module entry point. Exposes `FileWatcher` and `ChangeEvent` to Python.
///
/// The compiled module is named `file_watcher_rs` (not `file_watcher`) so it
/// does NOT collide with the pure-Python wrapper package
/// `shared.python.file_watcher`, which dispatches to this extension. A matching
/// name would make the wrapper import itself (issue #3520).
#[cfg(feature = "python")]
#[pymodule]
fn file_watcher_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python::PyFileWatcher>()?;
    m.add_class::<python::PyChangeEvent>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
