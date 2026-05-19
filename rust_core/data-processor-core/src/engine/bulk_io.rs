//! Engine scaffolding for the Rust/Polars bulk-I/O backend (issue #2989, Phase 2).
//!
//! Phase 2 defines the contract — function signatures, input validation, and
//! error types — without a full Polars implementation.  Each function either
//! delegates to the existing CSV helpers in the parent crate or returns a
//! typed `EngineError::NotImplemented` so the Python fallback can take over.
//!
//! The `python` feature gate (pyo3) is handled in `mod.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::{inspect as csv_inspect, preview as csv_preview};
use thiserror::Error;

use super::schema::{ConversionReport, SchemaInfo};

// ─── Error type ──────────────────────────────────────────────────────────────

/// Errors produced by the bulk-I/O engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("path must not be empty")]
    EmptyPath,
    #[error("file does not exist: {0}")]
    FileNotFound(String),
    #[error("unsupported format '{0}': only csv and parquet are supported")]
    UnsupportedFormat(String),
    #[error("unsupported output format '{0}': only csv and parquet are supported")]
    UnsupportedOutputFormat(String),
    #[error("nrows must be greater than zero")]
    InvalidRowLimit,
    #[error("column not found: {0}")]
    MissingColumn(String),
    #[error("operation cancelled")]
    Cancelled,
    #[error("not yet implemented: {0}")]
    NotImplemented(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("data error: {0}")]
    Data(String),
}

impl From<crate::DataProcessorError> for EngineError {
    fn from(e: crate::DataProcessorError) -> Self {
        match e {
            crate::DataProcessorError::EmptyPath => EngineError::EmptyPath,
            crate::DataProcessorError::FileNotFound(p) => {
                EngineError::FileNotFound(p.display().to_string())
            }
            crate::DataProcessorError::UnsupportedFormat(p) => {
                EngineError::UnsupportedFormat(p.display().to_string())
            }
            crate::DataProcessorError::UnsupportedOutputFormat(s) => {
                EngineError::UnsupportedOutputFormat(s)
            }
            crate::DataProcessorError::InvalidRowLimit => EngineError::InvalidRowLimit,
            crate::DataProcessorError::MissingColumn(s) => EngineError::MissingColumn(s),
            crate::DataProcessorError::Csv(e) => EngineError::Data(e.to_string()),
            crate::DataProcessorError::Io(e) => EngineError::Io(e),
        }
    }
}

// ─── Contract helpers ─────────────────────────────────────────────────────────

fn require_path(path: &Path) -> Result<(), EngineError> {
    if path.as_os_str().is_empty() {
        return Err(EngineError::EmptyPath);
    }
    if !path.is_file() {
        return Err(EngineError::FileNotFound(path.display().to_string()));
    }
    Ok(())
}

fn detect_format(path: &Path) -> Result<String, EngineError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("csv") => Ok("csv".to_owned()),
        Some("parquet") => Ok("parquet".to_owned()),
        _ => Err(EngineError::UnsupportedFormat(path.display().to_string())),
    }
}

fn validate_output_format(fmt: &str) -> Result<(), EngineError> {
    match fmt {
        "csv" | "parquet" => Ok(()),
        other => Err(EngineError::UnsupportedOutputFormat(other.to_owned())),
    }
}

// ─── Engine contract ──────────────────────────────────────────────────────────

/// Return column names, inferred types, estimated row count, and file size.
///
/// For CSV files, row count is exact (full scan).  Parquet will use footer
/// statistics once the Polars back-end is wired in (Phase 3).
pub fn inspect(path: &Path) -> Result<SchemaInfo, EngineError> {
    require_path(path)?;
    let format = detect_format(path)?;
    let file_size_bytes = fs::metadata(path)?.len();

    match format.as_str() {
        "csv" => {
            let meta = csv_inspect(path)?;
            // Infer types as "Utf8" (string) — Phase 3 will use Polars lazy schema scan.
            let column_types: BTreeMap<String, String> = meta
                .columns
                .iter()
                .map(|c| (c.clone(), "Utf8".to_owned()))
                .collect();
            Ok(SchemaInfo {
                columns: meta.columns,
                column_types,
                row_count_estimate: meta.row_count,
                file_size_bytes,
                format,
            })
        }
        "parquet" => {
            // Phase 3: use Polars `LazyFrame::scan_parquet` for zero-copy schema.
            // For now, surface a clear not-implemented signal so the Python
            // wrapper falls back to pandas.
            Err(EngineError::NotImplemented(
                "parquet inspect will be implemented in Phase 3 (Polars lazy scan)".to_owned(),
            ))
        }
        _ => unreachable!("detect_format already validated the extension"),
    }
}

/// Return up to `nrows` rows, optionally projecting to `columns`.
///
/// Returns a `Vec<BTreeMap<String, String>>` (column-name → string value) so
/// the Python layer can convert to a `pd.DataFrame` with correct dtypes.
pub fn preview(
    path: &Path,
    nrows: usize,
    columns: Option<&[&str]>,
) -> Result<Vec<BTreeMap<String, String>>, EngineError> {
    require_path(path)?;
    if nrows == 0 {
        return Err(EngineError::InvalidRowLimit);
    }
    let format = detect_format(path)?;

    match format.as_str() {
        "csv" => {
            let cols_str = columns.map(|cs| cs.join(","));
            let table = csv_preview(path, nrows, cols_str.as_deref())?;
            Ok(table.rows)
        }
        "parquet" => Err(EngineError::NotImplemented(
            "parquet preview will be implemented in Phase 3 (Polars lazy scan)".to_owned(),
        )),
        _ => unreachable!(),
    }
}

/// Convert `src` to `dst` in `output_format` (`"csv"` or `"parquet"`).
///
/// Phase 2 implements CSV→CSV only; Parquet paths return `NotImplemented` so
/// the Python fallback can handle them.
pub fn convert(
    src: &Path,
    dst: &Path,
    output_format: &str,
) -> Result<ConversionReport, EngineError> {
    require_path(src)?;
    validate_output_format(output_format)?;
    let src_format = detect_format(src)?;

    match (src_format.as_str(), output_format) {
        ("csv", "csv") => {
            let report = crate::convert(src, dst, output_format, None)?;
            Ok(ConversionReport {
                source: report.input,
                destination: report.output,
                output_format: report.output_format,
                rows_written: report.rows_written,
                columns: report.columns,
                bytes_written: report.bytes_written,
            })
        }
        _ => Err(EngineError::NotImplemented(format!(
            "{src_format} → {output_format} conversion will be implemented in Phase 3 (Polars)"
        ))),
    }
}

/// Scan `path` in batches of `batch_size` rows, yielding each batch.
///
/// Phase 2 scaffold: validates inputs and returns `NotImplemented`.
/// Phase 3 will wire up `Polars LazyFrame::scan_csv / scan_parquet` with
/// chunked materialisation.
pub fn scan_batch(
    path: &Path,
    batch_size: usize,
    _columns: Option<&[&str]>,
) -> Result<(), EngineError> {
    require_path(path)?;
    detect_format(path)?;
    if batch_size == 0 {
        return Err(EngineError::InvalidRowLimit);
    }
    Err(EngineError::NotImplemented(
        "scan_batch iterator will be implemented in Phase 3 (Polars streaming)".to_owned(),
    ))
}

/// Filter rows matching `predicate` and export to `dst`.  Returns row count.
///
/// Phase 2 scaffold: validates inputs and returns `NotImplemented`.
/// Phase 3 will use `Polars LazyFrame::filter` + `.sink_csv` / `.sink_parquet`.
pub fn filter_export(
    path: &Path,
    dst: &Path,
    predicate: &str,
    _columns: Option<&[&str]>,
) -> Result<u64, EngineError> {
    require_path(path)?;
    detect_format(path)?;
    if dst.as_os_str().is_empty() {
        return Err(EngineError::EmptyPath);
    }
    if predicate.trim().is_empty() {
        return Err(EngineError::Data("predicate must not be empty".to_owned()));
    }
    Err(EngineError::NotImplemented(
        "filter_export will be implemented in Phase 3 (Polars lazy filter)".to_owned(),
    ))
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_dir() -> std::path::PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("engine_scaffold_{ts}"));
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write_csv(path: &Path) {
        fs::write(path, "time,force,note\n0.0,10.5,start\n0.1,11.0,mid\n")
            .expect("write fixture csv");
    }

    #[test]
    fn inspect_csv_returns_schema_info() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);

        let info = inspect(&path).expect("inspect should succeed");

        assert_eq!(info.format, "csv");
        assert_eq!(info.columns, vec!["time", "force", "note"]);
        assert_eq!(info.row_count_estimate, 2);
        assert!(info.file_size_bytes > 0);
        // Phase 2: all types inferred as Utf8
        assert_eq!(
            info.column_types.get("time").map(String::as_str),
            Some("Utf8")
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn inspect_empty_path_returns_empty_path_error() {
        let result = inspect(Path::new(""));
        assert!(matches!(result, Err(EngineError::EmptyPath)));
    }

    #[test]
    fn inspect_missing_file_returns_file_not_found() {
        let result = inspect(Path::new("nonexistent_file_abc123.csv"));
        assert!(matches!(result, Err(EngineError::FileNotFound(_))));
    }

    #[test]
    fn inspect_parquet_returns_not_implemented() {
        let dir = temp_dir();
        // Create a fake .parquet file so require_path passes
        let path = dir.join("fake.parquet");
        fs::write(&path, b"PAR1").expect("write fake parquet");
        let result = inspect(&path);
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preview_csv_returns_rows() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);

        let rows = preview(&path, 1, None).expect("preview should succeed");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("time").map(String::as_str), Some("0.0"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preview_zero_rows_returns_invalid_row_limit() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);
        let result = preview(&path, 0, None);
        assert!(matches!(result, Err(EngineError::InvalidRowLimit)));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preview_column_projection() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);

        let rows = preview(&path, 10, Some(&["force", "note"])).expect("preview projected");

        assert!(rows[0].contains_key("force"));
        assert!(rows[0].contains_key("note"));
        assert!(!rows[0].contains_key("time"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn convert_csv_to_csv() {
        let dir = temp_dir();
        let src = dir.join("in.csv");
        let dst = dir.join("out.csv");
        write_csv(&src);

        let report = convert(&src, &dst, "csv").expect("csv→csv convert should succeed");

        assert_eq!(report.rows_written, 2);
        assert!(dst.is_file());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn convert_unsupported_output_format() {
        let dir = temp_dir();
        let src = dir.join("in.csv");
        write_csv(&src);
        let result = convert(&src, &dir.join("out.xlsx"), "xlsx");
        assert!(matches!(
            result,
            Err(EngineError::UnsupportedOutputFormat(_))
        ));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn scan_batch_returns_not_implemented() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);
        let result = scan_batch(&path, 100, None);
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn filter_export_returns_not_implemented() {
        let dir = temp_dir();
        let src = dir.join("sample.csv");
        write_csv(&src);
        let dst = dir.join("out.csv");
        let result = filter_export(&src, &dst, "force > 10.0", None);
        assert!(matches!(result, Err(EngineError::NotImplemented(_))));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn filter_export_empty_predicate_returns_data_error() {
        let dir = temp_dir();
        let src = dir.join("sample.csv");
        write_csv(&src);
        let dst = dir.join("out.csv");
        let result = filter_export(&src, &dst, "  ", None);
        assert!(matches!(result, Err(EngineError::Data(_))));
        fs::remove_dir_all(dir).ok();
    }
}
