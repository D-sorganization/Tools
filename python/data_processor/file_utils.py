"""File utility functions for data processing operations."""

from pathlib import Path
from typing import Any

import pandas as pd


class DataReader:
    """Class for reading data files in various formats."""

    @staticmethod
    def read_file(file_path: str | Path, format_type: str) -> pd.DataFrame:
        """Read a data file based on its format.

        Args:
            file_path: Path to the file to read
            format_type: Format of the file (csv, excel, parquet, etc.)

        Returns:
            pd.DataFrame: The loaded data

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)

        if format_type.lower() == "csv":
            return pd.read_csv(file_path)
        if format_type.lower() == "excel":
            return pd.read_excel(file_path)
        if format_type.lower() == "parquet":
            return pd.read_parquet(file_path)
        if format_type.lower() == "json":
            return pd.read_json(file_path)
        if format_type.lower() == "pickle":
            return pd.read_pickle(file_path)
        if format_type.lower() == "hdf5":
            return pd.read_hdf(file_path)
        if format_type.lower() == "feather":
            return pd.read_feather(file_path)
        raise ValueError(f"Unsupported format: {format_type}")

    @staticmethod
    def detect_format(file_path: str | Path) -> str:
        """Detect the format of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected format type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        format_mapping = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".parquet": "parquet",
            ".json": "json",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".feather": "feather",
        }

        return format_mapping.get(extension, "csv")


class DataWriter:
    """Class for writing data files in various formats."""

    @staticmethod
    def write_file(
        data: pd.DataFrame, file_path: str | Path, format_type: str,
    ) -> None:
        """Write data to a file in the specified format.

        Args:
            data: DataFrame to write
            file_path: Path where to save the file
            format_type: Format to save the file in

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)

        if format_type.lower() == "csv":
            data.to_csv(file_path, index=False)
        elif format_type.lower() == "excel":
            data.to_excel(file_path, index=False)
        elif format_type.lower() == "parquet":
            data.to_parquet(file_path, index=False)
        elif format_type.lower() == "json":
            data.to_json(file_path, orient="records", indent=2)
        elif format_type.lower() == "pickle":
            data.to_pickle(file_path)
        elif format_type.lower() == "hdf5":
            data.to_hdf(file_path, key="data", mode="w")
        elif format_type.lower() == "feather":
            data.to_feather(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


class FileFormatDetector:
    """Class for detecting file formats and providing format information."""

    @staticmethod
    def detect_format(file_path: str | Path) -> str:
        """Detect the format of a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            str: Detected format type
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
            ".xlsx",
            ".xls",
            ".parquet",
            ".json",
            ".pkl",
            ".pickle",
            ".h5",
            ".hdf5",
            ".feather",
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
    """Get information about a file.

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
    except Exception as e:
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
    except Exception:
        return False
