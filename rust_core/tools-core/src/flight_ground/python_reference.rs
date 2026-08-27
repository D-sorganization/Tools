//! PyO3 execution boundary for the compiled ground reference runtime.

use std::cell::RefCell;

use pyo3::exceptions::{PyInterruptedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::{run_ground_reference_v1_json, GroundReferenceBoundaryErrorV1};

#[pyfunction(
    name = "run_flight_to_ground_reference_v1",
    signature = (request_json, execution_json=None, is_cancelled=None)
)]
pub fn py_run_ground_reference_v1(
    py: Python<'_>,
    request_json: String,
    execution_json: Option<String>,
    is_cancelled: Option<Py<PyAny>>,
) -> PyResult<String> {
    if let Some(callback) = is_cancelled {
        return py
            .detach(move || run_with_callback(&request_json, execution_json.as_deref(), callback));
    }
    py.detach(|| run_ground_reference_v1_json(&request_json, execution_json.as_deref(), || false))
        .map_err(boundary_error)
}

fn run_with_callback(
    request_json: &str,
    execution_json: Option<&str>,
    callback: Py<PyAny>,
) -> PyResult<String> {
    let callback_error = RefCell::new(None);
    let result = run_ground_reference_v1_json(request_json, execution_json, || {
        let polled = Python::attach(|py| {
            callback
                .call0(py)
                .and_then(|value| value.extract::<bool>(py))
        });
        match polled {
            Ok(cancelled) => cancelled,
            Err(error) => {
                *callback_error.borrow_mut() = Some(error);
                true
            }
        }
    });
    if let Some(error) = callback_error.into_inner() {
        return Err(error);
    }
    result.map_err(boundary_error)
}

fn boundary_error(error: GroundReferenceBoundaryErrorV1) -> PyErr {
    let payload = error.payload();
    if error.is_cancelled() {
        return PyInterruptedError::new_err(payload);
    }
    match error {
        GroundReferenceBoundaryErrorV1::Request(_)
        | GroundReferenceBoundaryErrorV1::Execution(_) => PyValueError::new_err(payload),
        GroundReferenceBoundaryErrorV1::Runtime(_) | GroundReferenceBoundaryErrorV1::Result(_) => {
            PyRuntimeError::new_err(payload)
        }
    }
}
