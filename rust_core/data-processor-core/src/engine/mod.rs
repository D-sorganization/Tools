//! Bulk-I/O engine module — Phase 2 of issue #2989.
//!
//! Exposes the engine contract (`inspect`, `preview`, `convert`,
//! `scan_batch`, `filter_export`) plus optional PyO3 Python bindings
//! behind the `python` feature flag.

pub mod bulk_io;
pub mod schema;

pub use bulk_io::{convert, filter_export, inspect, preview, scan_batch, EngineError};
pub use schema::{ConversionReport, SchemaInfo};

// ── Python bindings ───────────────────────────────────────────────────────────
//
// Enabled only when building a maturin wheel (`--features python`).
// `cargo test` without the feature flag must compile and run cleanly.

#[cfg(feature = "python")]
mod python_bindings {
    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::path::Path;

    use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use pyo3::IntoPyObjectExt;

    use super::bulk_io::{
        convert as engine_convert, filter_export as engine_filter_export,
        inspect as engine_inspect, preview as engine_preview, scan_batch as engine_scan_batch,
        EngineError,
    };
    use super::schema::{ConversionReport, SchemaInfo};

    fn map_engine_error(e: EngineError) -> PyErr {
        match &e {
            EngineError::NotImplemented(_) => PyNotImplementedError::new_err(e.to_string()),
            EngineError::EmptyPath
            | EngineError::InvalidRowLimit
            | EngineError::UnsupportedFormat(_)
            | EngineError::UnsupportedOutputFormat(_)
            | EngineError::MissingColumn(_)
            | EngineError::Data(_) => PyValueError::new_err(e.to_string()),
            EngineError::FileNotFound(_) | EngineError::Io(_) | EngineError::Cancelled => {
                PyIOError::new_err(e.to_string())
            }
        }
    }

    /// Python-facing wrapper for [`engine::inspect`].
    ///
    /// Returns a dict with keys: `columns`, `column_types`, `row_count_estimate`,
    /// `file_size_bytes`, `format`.
    #[pyfunction]
    fn py_inspect(path: &str) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let info: SchemaInfo = engine_inspect(Path::new(path)).map_err(map_engine_error)?;
            let mut d = HashMap::new();
            d.insert("columns".to_owned(), info.columns.into_py_any(py)?);
            d.insert(
                "column_types".to_owned(),
                info.column_types
                    .into_iter()
                    .collect::<HashMap<_, _>>()
                    .into_py_any(py)?,
            );
            d.insert(
                "row_count_estimate".to_owned(),
                info.row_count_estimate.into_py_any(py)?,
            );
            d.insert(
                "file_size_bytes".to_owned(),
                info.file_size_bytes.into_py_any(py)?,
            );
            d.insert("format".to_owned(), info.format.into_py_any(py)?);
            Ok(d)
        })
    }

    /// Python-facing wrapper for [`engine::preview`].
    ///
    /// Returns a list of dicts (column-name → string value).
    #[pyfunction]
    #[pyo3(signature = (path, nrows=100, columns=None))]
    fn py_preview(
        path: &str,
        nrows: usize,
        columns: Option<Vec<String>>,
    ) -> PyResult<Vec<BTreeMap<String, String>>> {
        let col_refs: Option<Vec<&str>> = columns
            .as_ref()
            .map(|cs| cs.iter().map(String::as_str).collect());
        engine_preview(Path::new(path), nrows, col_refs.as_deref()).map_err(map_engine_error)
    }

    /// Python-facing wrapper for [`engine::convert`].
    ///
    /// Returns a dict with conversion statistics.
    #[pyfunction]
    fn py_convert(
        src: &str,
        dst: &str,
        output_format: &str,
    ) -> PyResult<HashMap<String, Py<PyAny>>> {
        Python::attach(|py| {
            let report: ConversionReport =
                engine_convert(Path::new(src), Path::new(dst), output_format)
                    .map_err(map_engine_error)?;
            let mut d = HashMap::new();
            d.insert("source".to_owned(), report.source.into_py_any(py)?);
            d.insert(
                "destination".to_owned(),
                report.destination.into_py_any(py)?,
            );
            d.insert(
                "output_format".to_owned(),
                report.output_format.into_py_any(py)?,
            );
            d.insert(
                "rows_written".to_owned(),
                report.rows_written.into_py_any(py)?,
            );
            d.insert("columns".to_owned(), report.columns.into_py_any(py)?);
            d.insert(
                "bytes_written".to_owned(),
                report.bytes_written.into_py_any(py)?,
            );
            Ok(d)
        })
    }

    /// Python-facing wrapper for [`engine::scan_batch`].
    ///
    /// Phase 2 scaffold — always raises `NotImplementedError`.
    #[pyfunction]
    #[pyo3(signature = (path, batch_size, columns=None))]
    fn py_scan_batch(path: &str, batch_size: usize, columns: Option<Vec<String>>) -> PyResult<()> {
        let col_refs: Option<Vec<&str>> = columns
            .as_ref()
            .map(|cs| cs.iter().map(String::as_str).collect());
        engine_scan_batch(Path::new(path), batch_size, col_refs.as_deref())
            .map_err(map_engine_error)
    }

    /// Python-facing wrapper for [`engine::filter_export`].
    ///
    /// Phase 2 scaffold — always raises `NotImplementedError`.
    #[pyfunction]
    #[pyo3(signature = (path, dst, predicate, columns=None))]
    fn py_filter_export(
        path: &str,
        dst: &str,
        predicate: &str,
        columns: Option<Vec<String>>,
    ) -> PyResult<u64> {
        let col_refs: Option<Vec<&str>> = columns
            .as_ref()
            .map(|cs| cs.iter().map(String::as_str).collect());
        engine_filter_export(
            Path::new(path),
            Path::new(dst),
            predicate,
            col_refs.as_deref(),
        )
        .map_err(map_engine_error)
    }

    /// Register all engine functions into the PyO3 module.
    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_inspect, m)?)?;
        m.add_function(wrap_pyfunction!(py_preview, m)?)?;
        m.add_function(wrap_pyfunction!(py_convert, m)?)?;
        m.add_function(wrap_pyfunction!(py_scan_batch, m)?)?;
        m.add_function(wrap_pyfunction!(py_filter_export, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::register;
