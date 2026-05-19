//! Schema and report types for the bulk-I/O engine contract.
//!
//! These types are the stable API surface defined by Phase 2 of issue #2989.
//! They are serialisable so they can be round-tripped as JSON and, when the
//! `python` feature is enabled, exposed to Python via PyO3.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Column name → inferred type string (e.g. `"Float64"`, `"Utf8"`, `"Int64"`).
pub type ColumnTypes = BTreeMap<String, String>;

/// Metadata returned by [`super::engine::inspect`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchemaInfo {
    /// Ordered list of column names.
    pub columns: Vec<String>,
    /// Inferred dtype for each column (column-name → dtype string).
    pub column_types: ColumnTypes,
    /// Estimated row count.  May be exact (CSV) or approximate (Parquet stats).
    pub row_count_estimate: u64,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Detected file format (`"csv"` or `"parquet"`).
    pub format: String,
}

/// Report returned by [`super::engine::convert`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversionReport {
    /// Absolute or relative path of the source file.
    pub source: String,
    /// Absolute or relative path of the destination file.
    pub destination: String,
    /// Output format string (`"csv"` or `"parquet"`).
    pub output_format: String,
    /// Number of rows written to the destination.
    pub rows_written: u64,
    /// Columns present in the output.
    pub columns: Vec<String>,
    /// Size of the destination file in bytes.
    pub bytes_written: u64,
}
