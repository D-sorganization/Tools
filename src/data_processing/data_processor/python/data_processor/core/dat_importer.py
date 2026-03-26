"""DAT/DBF file importer for industrial data files.

Provides functions for:
- Reading DAT files (tab-separated industrial data)
- Parsing DBF tag files for signal names
- Converting DAT data to CSV/DataFrame format
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Optional DBF support
try:
    from simpledbf import Dbf5

    DBF_AVAILABLE = True
except ImportError:
    DBF_AVAILABLE = False


def read_dat_file(
    file_path: str | Path,
    delimiter: str = "\t",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a DAT file into a DataFrame.

    DAT files are typically tab-separated industrial data files.

    Args:
        file_path: Path to the DAT file
        delimiter: Field delimiter (default: tab)
        encoding: File encoding
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame with the data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"DAT file not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        low_memory=False,
        **kwargs,
    )


def read_dbf_tags(file_path: str | Path) -> list[str]:
    """Read tag names from a DBF file.

    DBF files often contain metadata including signal/tag names.

    Args:
        file_path: Path to the DBF file

    Returns:
        List of tag names

    Raises:
        ImportError: If simpledbf is not installed
        FileNotFoundError: If file doesn't exist
    """
    if not DBF_AVAILABLE:
        raise ImportError(
            "simpledbf is required for DBF file support. " "Install with: pip install simpledbf"
        )

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"DBF file not found: {file_path}")

    dbf = Dbf5(str(file_path))
    df = dbf.to_dataframe()

    # Extract tag names - common column names in industrial DBF files
    tag_columns = ["TAG", "TAGNAME", "NAME", "SIGNAL", "DESCRIPTION"]

    for col in tag_columns:
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()

    # If no known column found, return all values from first column
    if len(df.columns) > 0:
        return df.iloc[:, 0].dropna().astype(str).tolist()

    return []


def get_dat_columns(
    file_path: str | Path,
    delimiter: str = "\t",
    nrows: int = 5,
) -> list[str]:
    """Get column names from a DAT file.

    Args:
        file_path: Path to the DAT file
        delimiter: Field delimiter
        nrows: Number of rows to read for detection

    Returns:
        List of column names
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    df = read_dat_file(file_path, delimiter=delimiter, nrows=nrows)
    return df.columns.tolist()


def import_dat_with_tags(
    dat_path: str | Path,
    dbf_path: str | Path | None = None,
    selected_tags: list[str] | None = None,
    delimiter: str = "\t",
    time_column: str = "Time",
) -> pd.DataFrame:
    """Import DAT file with optional tag filtering from DBF.

    Args:
        dat_path: Path to the DAT data file
        dbf_path: Optional path to DBF tag file
        selected_tags: List of tags to include (None = all)
        delimiter: Field delimiter for DAT file
        time_column: Name of the time column to preserve

    Returns:
        DataFrame with imported data
    """
    # Read the DAT file
    df = read_dat_file(dat_path, delimiter=delimiter)

    # If no selection, return all
    if selected_tags is None:
        return df

    # Build column list
    columns = []

    # Always include time column if present
    if time_column in df.columns:
        columns.append(time_column)

    # Add selected tags that exist in the data
    for tag in selected_tags:
        if tag in df.columns and tag not in columns:
            columns.append(tag)

    if not columns:
        raise ValueError("No valid columns found for selection")

    return df[columns]


def export_dat_to_csv(
    dat_path: str | Path,
    output_path: str | Path,
    selected_tags: list[str] | None = None,
    delimiter: str = "\t",
    time_column: str = "Time",
) -> Path:
    """Export DAT file to CSV format.

    Args:
        dat_path: Path to the DAT data file
        output_path: Path for the output CSV
        selected_tags: List of tags to include (None = all)
        delimiter: Field delimiter for DAT file
        time_column: Name of the time column

    Returns:
        Path to the created CSV file
    """
    if not (dat_path is not None):
        raise ValueError("dat_path must be provided")
    df = import_dat_with_tags(
        dat_path,
        selected_tags=selected_tags,
        delimiter=delimiter,
        time_column=time_column,
    )

    output_path = Path(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def preview_dat_file(
    file_path: str | Path,
    delimiter: str = "\t",
    nrows: int = 100,
) -> pd.DataFrame:
    """Preview the first N rows of a DAT file.

    Args:
        file_path: Path to the DAT file
        delimiter: Field delimiter
        nrows: Number of rows to preview

    Returns:
        DataFrame with preview data
    """
    return read_dat_file(file_path, delimiter=delimiter, nrows=nrows)


def detect_dat_delimiter(file_path: str | Path) -> str:
    """Attempt to detect the delimiter used in a DAT file.

    Args:
        file_path: Path to the DAT file

    Returns:
        Detected delimiter character
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8", errors="ignore") as f:
        first_lines = [f.readline() for _ in range(5)]

    # Count potential delimiters
    delimiters = {"\t": 0, ",": 0, ";": 0, "|": 0}

    for line in first_lines:
        for delim in delimiters:
            delimiters[delim] += line.count(delim)

    # Return the most common delimiter
    return max(delimiters, key=delimiters.get)  # type: ignore[arg-type]


def get_dat_file_info(file_path: str | Path, delimiter: str = "\t") -> dict[str, Any]:
    """Get information about a DAT file.

    Args:
        file_path: Path to the DAT file
        delimiter: Field delimiter

    Returns:
        Dictionary with file info:
        - columns: List of column names
        - row_count: Approximate row count
        - file_size: File size in bytes
        - has_time_column: Whether a time column was detected
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    file_path = Path(file_path)

    # Get basic file info
    file_size = file_path.stat().st_size

    # Read header and sample
    df = read_dat_file(file_path, delimiter=delimiter, nrows=10)
    columns = df.columns.tolist()

    # Estimate row count
    # Read first chunk to get average line length
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        sample = f.read(10000)
    lines_in_sample = sample.count("\n")
    if lines_in_sample > 0:
        avg_line_length = len(sample) / lines_in_sample
        estimated_rows = int(file_size / avg_line_length)
    else:
        estimated_rows = 0

    # Check for time column
    time_columns = ["time", "timestamp", "date", "datetime"]
    has_time = any(col.lower() in time_columns for col in columns)

    return {
        "columns": columns,
        "column_count": len(columns),
        "estimated_row_count": estimated_rows,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "has_time_column": has_time,
        "detected_delimiter": delimiter,
    }


__all__ = [
    "read_dat_file",
    "read_dbf_tags",
    "get_dat_columns",
    "import_dat_with_tags",
    "export_dat_to_csv",
    "preview_dat_file",
    "detect_dat_delimiter",
    "get_dat_file_info",
    "DBF_AVAILABLE",
]
