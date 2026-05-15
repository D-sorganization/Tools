//! PyO3 bindings for `FileWatcher`.
//!
//! Exposes a Python-friendly surface that mirrors the design in the issue:
//!
//! ```python
//! from file_watcher import FileWatcher
//! w = FileWatcher(root="/some/path", debounce_ms=100, respect_gitignore=True)
//! w.on_change(lambda events: ...)
//! w.start()
//! w.stop()
//! ```
//!
//! Callbacks are invoked from the debounce thread; we acquire the GIL each
//! time before calling the Python callable.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::watcher::{ChangeEvent, ChangeKind, FileWatcher, FileWatcherConfig};

#[pyclass(name = "ChangeEvent")]
#[derive(Clone)]
pub struct PyChangeEvent {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub kind: String,
}

impl From<ChangeEvent> for PyChangeEvent {
    fn from(ev: ChangeEvent) -> Self {
        Self {
            path: ev.path.to_string_lossy().into_owned(),
            kind: ev.kind.as_str().to_string(),
        }
    }
}

#[pymethods]
impl PyChangeEvent {
    fn __repr__(&self) -> String {
        format!("ChangeEvent(path={!r}, kind={!r})", self.path, self.kind)
    }
}

#[pyclass(name = "FileWatcher")]
pub struct PyFileWatcher {
    inner: Arc<FileWatcher>,
    callback: Arc<Mutex<Option<PyObject>>>,
}

#[pymethods]
impl PyFileWatcher {
    #[new]
    #[pyo3(signature = (root, debounce_ms = 100, respect_gitignore = true))]
    fn new(root: &str, debounce_ms: u64, respect_gitignore: bool) -> PyResult<Self> {
        let path = PathBuf::from(root);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "root path does not exist: {root}"
            )));
        }
        let config = FileWatcherConfig {
            root: path,
            debounce_ms,
            respect_gitignore,
        };
        Ok(Self {
            inner: Arc::new(FileWatcher::new(config)),
            callback: Arc::new(Mutex::new(None)),
        })
    }

    /// Register a callback. Pass a callable that accepts `list[ChangeEvent]`.
    /// Replaces any previous callback.
    fn on_change(&self, callback: PyObject) {
        *self.callback.lock().unwrap() = Some(callback);
        let cb_slot = self.callback.clone();
        self.inner.on_change(move |events| {
            Python::with_gil(|py| {
                let Some(cb) = cb_slot.lock().unwrap().clone() else {
                    return;
                };
                let py_events: Vec<PyChangeEvent> =
                    events.into_iter().map(PyChangeEvent::from).collect();
                let list = PyList::new(py, py_events).expect("PyList::new");
                let _ = cb.call1(py, (list,));
            });
        });
    }

    fn start(&self) -> PyResult<()> {
        self.inner
            .start()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn stop(&self) -> PyResult<()> {
        self.inner
            .stop()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn is_running(&self) -> bool {
        self.inner.is_running()
    }

    #[getter]
    fn root(&self) -> String {
        self.inner.root().to_string_lossy().into_owned()
    }

    fn __enter__(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<Self>> {
        slf.borrow(py).start()?;
        Ok(slf)
    }

    fn __exit__(
        &self,
        _exc_type: PyObject,
        _exc_val: PyObject,
        _exc_tb: PyObject,
    ) -> PyResult<bool> {
        // Ignore "not started" errors on exit so `__exit__` is idempotent.
        let _ = self.stop();
        Ok(false)
    }
}
