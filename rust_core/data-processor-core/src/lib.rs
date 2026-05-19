// ── Bulk-I/O engine module (Phase 2 of issue #2989) ─────────────────────────
pub mod engine;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataProcessorError {
    #[error("path must be provided")]
    EmptyPath,
    #[error("file does not exist: {0}")]
    FileNotFound(PathBuf),
    #[error("Unsupported format for path: {0}")]
    UnsupportedFormat(PathBuf),
    #[error("Unsupported output format: {0}")]
    UnsupportedOutputFormat(String),
    #[error("rows must be greater than zero")]
    InvalidRowLimit,
    #[error("column not found: {0}")]
    MissingColumn(String),
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetMetadata {
    pub format: String,
    pub row_count: u64,
    pub columns: Vec<String>,
    pub byte_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreviewTable {
    pub columns: Vec<String>,
    pub rows: Vec<BTreeMap<String, String>>,
    pub rows_returned: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversionReport {
    pub input: String,
    pub output: String,
    pub output_format: String,
    pub rows_read: u64,
    pub rows_written: u64,
    pub columns: Vec<String>,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DataFormat {
    Csv,
}

fn require_file(path: &Path) -> Result<(), DataProcessorError> {
    if path.as_os_str().is_empty() {
        return Err(DataProcessorError::EmptyPath);
    }
    if !path.is_file() {
        return Err(DataProcessorError::FileNotFound(path.to_path_buf()));
    }
    Ok(())
}

fn detect_format(path: &Path) -> Result<DataFormat, DataProcessorError> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("csv") => Ok(DataFormat::Csv),
        _ => Err(DataProcessorError::UnsupportedFormat(path.to_path_buf())),
    }
}

fn parse_columns(columns: Option<&str>) -> Option<Vec<String>> {
    columns.map(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|column| !column.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    })
}

fn selected_indices(
    headers: &csv::StringRecord,
    columns: Option<&str>,
) -> Result<(Vec<usize>, Vec<String>), DataProcessorError> {
    let requested = parse_columns(columns);
    if let Some(requested_columns) = requested {
        let mut indices = Vec::with_capacity(requested_columns.len());
        for column in &requested_columns {
            let index = headers
                .iter()
                .position(|header| header == column)
                .ok_or_else(|| DataProcessorError::MissingColumn(column.clone()))?;
            indices.push(index);
        }
        Ok((indices, requested_columns))
    } else {
        Ok((
            (0..headers.len()).collect(),
            headers.iter().map(ToOwned::to_owned).collect(),
        ))
    }
}

pub fn inspect(path: &Path) -> Result<DatasetMetadata, DataProcessorError> {
    require_file(path)?;
    match detect_format(path)? {
        DataFormat::Csv => inspect_csv(path),
    }
}

pub fn preview(
    path: &Path,
    row_limit: usize,
    columns: Option<&str>,
) -> Result<PreviewTable, DataProcessorError> {
    require_file(path)?;
    if row_limit == 0 {
        return Err(DataProcessorError::InvalidRowLimit);
    }
    match detect_format(path)? {
        DataFormat::Csv => preview_csv(path, row_limit, columns),
    }
}

pub fn convert(
    input: &Path,
    output: &Path,
    output_format: &str,
    columns: Option<&str>,
) -> Result<ConversionReport, DataProcessorError> {
    require_file(input)?;
    if output_format != "csv" {
        return Err(DataProcessorError::UnsupportedOutputFormat(
            output_format.to_owned(),
        ));
    }
    match detect_format(input)? {
        DataFormat::Csv => convert_csv(input, output, output_format, columns),
    }
}

fn inspect_csv(path: &Path) -> Result<DatasetMetadata, DataProcessorError> {
    let mut reader = csv::Reader::from_path(path)?;
    let columns = reader
        .headers()?
        .iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut row_count = 0_u64;
    for record in reader.records() {
        record?;
        row_count += 1;
    }
    Ok(DatasetMetadata {
        format: "csv".to_owned(),
        row_count,
        columns,
        byte_size: fs::metadata(path)?.len(),
    })
}

fn preview_csv(
    path: &Path,
    row_limit: usize,
    columns: Option<&str>,
) -> Result<PreviewTable, DataProcessorError> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let (indices, selected_columns) = selected_indices(&headers, columns)?;
    let mut rows = Vec::with_capacity(row_limit);

    for record in reader.records().take(row_limit) {
        let record = record?;
        let mut row = BTreeMap::new();
        for (index, column) in indices.iter().zip(selected_columns.iter()) {
            row.insert(
                column.clone(),
                record.get(*index).unwrap_or_default().to_owned(),
            );
        }
        rows.push(row);
    }

    Ok(PreviewTable {
        columns: selected_columns,
        rows_returned: rows.len(),
        rows,
    })
}

fn convert_csv(
    input: &Path,
    output: &Path,
    output_format: &str,
    columns: Option<&str>,
) -> Result<ConversionReport, DataProcessorError> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut reader = csv::Reader::from_path(input)?;
    let headers = reader.headers()?.clone();
    let (indices, selected_columns) = selected_indices(&headers, columns)?;
    let mut writer = csv::Writer::from_path(output)?;
    writer.write_record(&selected_columns)?;

    let mut rows_read = 0_u64;
    let mut rows_written = 0_u64;
    for record in reader.records() {
        let record = record?;
        rows_read += 1;
        let selected = indices
            .iter()
            .map(|index| record.get(*index).unwrap_or_default())
            .collect::<Vec<_>>();
        writer.write_record(selected)?;
        rows_written += 1;
    }
    writer.flush()?;

    Ok(ConversionReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        output_format: output_format.to_owned(),
        rows_read,
        rows_written,
        columns: selected_columns,
        bytes_written: fs::metadata(output)?.len(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_dir() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("data_processor_core_{suffix}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }

    fn write_csv(path: &Path) {
        fs::write(path, "time,force,note\n0.0,10.5,start\n0.1,11.0,mid\n")
            .expect("fixture csv should be written");
    }

    #[test]
    fn inspect_counts_csv_rows() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);

        let metadata = inspect(&path).expect("csv should inspect");

        assert_eq!(metadata.format, "csv");
        assert_eq!(metadata.row_count, 2);
        assert_eq!(metadata.columns, vec!["time", "force", "note"]);
        assert!(metadata.byte_size > 0);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn preview_projects_columns() {
        let dir = temp_dir();
        let path = dir.join("sample.csv");
        write_csv(&path);

        let table = preview(&path, 1, Some("force,note")).expect("preview should work");

        assert_eq!(table.columns, vec!["force", "note"]);
        assert_eq!(table.rows_returned, 1);
        assert_eq!(table.rows[0].get("force").map(String::as_str), Some("10.5"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn convert_projects_columns_to_csv() {
        let dir = temp_dir();
        let input = dir.join("sample.csv");
        let output = dir.join("out.csv");
        write_csv(&input);

        let report =
            convert(&input, &output, "csv", Some("time,force")).expect("conversion should work");

        assert_eq!(report.rows_read, 2);
        assert_eq!(report.rows_written, 2);
        assert_eq!(report.columns, vec!["time", "force"]);
        assert_eq!(
            fs::read_to_string(output).expect("output should be readable"),
            "time,force\n0.0,10.5\n0.1,11.0\n"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn convert_allows_relative_output_without_parent_directory() {
        let dir = temp_dir();
        let input = dir.join("sample.csv");
        let output = PathBuf::from(format!(
            "data_processor_relative_{}.csv",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after epoch")
                .as_nanos()
        ));
        write_csv(&input);

        let report =
            convert(&input, &output, "csv", None).expect("relative conversion should work");

        assert_eq!(report.rows_written, 2);
        assert!(output.is_file());
        fs::remove_file(output).ok();
        fs::remove_dir_all(dir).ok();
    }
}
