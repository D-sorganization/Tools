# mypy: disable-error-code="no-any-return"
"""File utility functions for data processing operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional imports with availability flags
try:
    import scipy.io

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import openpyxl  # noqa: F401

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


class DataReader:
    """Class for reading data files in various formats."""

    @staticmethod
    def read_file(
        file_path: str | Path,
        format_type: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read a data file based on its format.

        Args:
            file_path: Path to the file to read
            format_type: Format of the file (csv, tsv, excel, parquet, etc.)
            **kwargs: Additional arguments passed to the underlying read function

        Returns:
            pd.DataFrame: The loaded data

        Raises:
            ValueError: If format is not supported
            ImportError: If required library is not available
        """
        file_path = Path(file_path)
        fmt = format_type.lower()

        if fmt == "csv":
            return pd.read_csv(file_path, **kwargs)
        if fmt == "tsv":
            return pd.read_csv(file_path, sep="\t", **kwargs)
        if fmt == "excel":
            return pd.read_excel(file_path, **kwargs)
        if fmt == "parquet":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for parquet files"
                raise ImportError(msg)
            return pd.read_parquet(file_path, **kwargs)
        if fmt == "json":
            return pd.read_json(file_path, **kwargs)
        if fmt == "pickle":
            return pd.read_pickle(file_path)
        if fmt == "hdf5":
            return pd.read_hdf(file_path, **kwargs)
        if fmt == "feather":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for feather files"
                raise ImportError(msg)
            return pd.read_feather(file_path, **kwargs)
        if fmt == "numpy":
            data = np.load(file_path, allow_pickle=False)
            if isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            return pd.DataFrame(data.item())
        if fmt == "matlab":
            if not SCIPY_AVAILABLE:
                msg = "SciPy is required for MATLAB files"
                raise ImportError(msg)
            data = scipy.io.loadmat(file_path)
            # Convert MATLAB struct to DataFrame
            # Filter out metadata keys
            data_keys = [k for k in data if not k.startswith("__")]
            if len(data_keys) == 1:
                return pd.DataFrame(data[data_keys[0]])
            return pd.DataFrame(
                {k: v for k, v in data.items() if not k.startswith("__")},
            )
        if fmt == "arrow":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for arrow files"
                raise ImportError(msg)
            table = pa.ipc.open_file(str(file_path)).read_all()
            return table.to_pandas()
        if fmt == "sqlite":
            import sqlite3

            conn = sqlite3.connect(str(file_path))
            df = pd.read_sql_query("SELECT * FROM data", conn)
            conn.close()
            return df

        msg = f"Unsupported format: {format_type}"
        raise ValueError(msg)

    @staticmethod
    def detect_format(file_path: str | Path) -> str:
        """Detect the format of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected format type, defaults to 'csv' if undetectable.
        """
        file_path = Path(file_path)

        # Check by extension first
        extension = file_path.suffix.lower()
        format_mapping = {
            ".csv": "csv",
            ".tsv": "tsv",
            ".txt": "tsv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".parquet": "parquet",
            ".pq": "parquet",
            ".json": "json",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".feather": "feather",
            ".npy": "numpy",
            ".mat": "matlab",
            ".arrow": "arrow",
            ".db": "sqlite",
            ".sqlite": "sqlite",
        }

        if extension in format_mapping:
            return format_mapping[extension]

        # Content-based detection for ambiguous extensions (only if file exists)
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    header = f.read(1024)

                # Check for CSV/TSV
                if b"," in header and b"\n" in header:
                    return "csv"
                elif b"\t" in header and b"\n" in header:
                    return "tsv"
                elif header.startswith(b"{") or header.startswith(b"["):
                    return "json"
                elif header.startswith(b"PK"):
                    return "excel"  # ZIP-based format
            except (PermissionError, OSError):
                pass

        return "csv"  # Default to csv for unrecognizable formats


class DataWriter:
    """Class for writing data files in various formats."""

    @staticmethod
    def write_file(
        data: pd.DataFrame,
        file_path: str | Path,
        format_type: str,
        **kwargs: Any,
    ) -> None:
        """Write data to a file in the specified format.

        Args:
            data: DataFrame to write
            file_path: Path where to save the file
            format_type: Format to save the file in
            **kwargs: Additional arguments passed to the underlying write function

        Raises:
            ValueError: If format is not supported
            ImportError: If required library is not available
        """
        if not (data is not None):
            raise ValueError("data must be provided")
        file_path = Path(file_path)
        fmt = format_type.lower()

        if fmt == "csv":
            data.to_csv(file_path, index=False, **kwargs)
        elif fmt == "tsv":
            data.to_csv(file_path, sep="\t", index=False, **kwargs)
        elif fmt == "excel":
            data.to_excel(file_path, index=False, **kwargs)
        elif fmt == "parquet":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for parquet files"
                raise ImportError(msg)
            data.to_parquet(file_path, **kwargs)
        elif fmt == "json":
            data.to_json(file_path, orient="records", indent=2, **kwargs)
        elif fmt == "pickle":
            data.to_pickle(file_path)
        elif fmt == "hdf5":
            data.to_hdf(file_path, key="data", mode="w", **kwargs)
        elif fmt == "feather":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for feather files"
                raise ImportError(msg)
            data.to_feather(file_path, **kwargs)
        elif fmt == "numpy":
            np.save(str(file_path), data.values)
        elif fmt == "matlab":
            if not SCIPY_AVAILABLE:
                msg = "SciPy is required for MATLAB files"
                raise ImportError(msg)
            scipy.io.savemat(
                str(file_path),
                {"data": data.values, "columns": data.columns.tolist()},
            )
        elif fmt == "arrow":
            if not PYARROW_AVAILABLE:
                msg = "PyArrow is required for arrow files"
                raise ImportError(msg)
            table = pa.Table.from_pandas(data)
            with pa.OSFile(str(file_path), "wb") as sink:
                with pa.ipc.new_file(sink, table.schema) as writer:
                    writer.write(table)
        elif fmt == "sqlite":
            import sqlite3

            conn = sqlite3.connect(str(file_path))
            data.to_sql("data", conn, if_exists="replace", index=False)
            conn.close()
        else:
            msg = f"Unsupported format: {format_type}"
            raise ValueError(msg)


class FileFormatDetector:
    """Class for detecting file formats and providing format information."""

    @staticmethod
    def detect_format(file_path: str | Path) -> str | None:
        """Detect the format of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            str | None: Detected format type, or None if detection fails
        """
        return DataReader.detect_format(file_path)

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported file formats.

        Returns:
            List[str]: List of supported format extensions
        """
        return [
            ".csv",
            ".tsv",
            ".txt",
            ".xlsx",
            ".xls",
            ".parquet",
            ".pq",
            ".json",
            ".pkl",
            ".pickle",
            ".h5",
            ".hdf5",
            ".feather",
            ".npy",
            ".mat",
            ".arrow",
            ".db",
            ".sqlite",
        ]

    @staticmethod
    def is_format_supported(file_path: str | Path) -> bool:
        """Check if a file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if format is supported, False otherwise
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        return extension in FileFormatDetector.get_supported_formats()


def get_file_info(file_path: str | Path) -> dict[str, Any]:
    """Get information for a file.

    Args:
        file_path: Path to the file

    Returns:
        Dict[str, Any]: Dictionary containing file information
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {"error": "File does not exist"}

    try:
        stat = file_path.stat()
        return {
            "name": file_path.name,
            "size": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "format": FileFormatDetector.detect_format(file_path),
            "is_supported": FileFormatDetector.is_format_supported(file_path),
        }
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return {"error": str(e)}


def validate_file_path(file_path: str | Path) -> bool:
    """Validate if a file path is valid and accessible.

    Args:
        file_path: Path to validate

    Returns:
        bool: True if path is valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except (PermissionError, OSError):
        return False
