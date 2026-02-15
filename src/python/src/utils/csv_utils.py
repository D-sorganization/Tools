"""
CSV file operation utilities for consistent CSV handling.

This module provides reusable functions for CSV operations across
the repository, following DRY principles.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def safe_read_csv(
    file_path: Path | str,
    default: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Safely read a CSV file with error handling.

    Args:
        file_path: Path to CSV file
        default: Default DataFrame to return on error
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame or default value
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"CSV file not found: {path}, using default")
        return default if default is not None else pd.DataFrame()

    try:
        result: pd.DataFrame = pd.read_csv(path, **kwargs)
        return result
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.error(f"Error reading CSV file {path}: {e}")
        return default if default is not None else pd.DataFrame()


def safe_write_csv(
    df: pd.DataFrame,
    file_path: Path | str,
    create_parents: bool = True,
    **kwargs: Any,
) -> bool:
    """Safely write a DataFrame to CSV with error handling.

    Args:
        df: DataFrame to write
        file_path: Path to CSV file
        create_parents: Whether to create parent directories
        **kwargs: Additional arguments passed to df.to_csv

    Returns:
        True if write succeeded, False otherwise
    """
    path = Path(file_path)

    try:
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, **kwargs)
        logger.debug(f"Successfully wrote CSV to {path}")
        return True
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing CSV file {path}: {e}")
        return False


def read_csv_with_validation(
    file_path: Path | str,
    required_columns: list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Read CSV and validate it has required columns.

    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame if valid, None otherwise
    """
    df = safe_read_csv(file_path, **kwargs)

    if df.empty:
        logger.warning(f"CSV file is empty: {file_path}")
        return None

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"CSV file {file_path} missing required columns: {missing}")
            return None

    return df


def merge_csv_files(
    file_paths: list[Path | str],
    output_path: Path | str,
    **kwargs: Any,
) -> bool:
    """Merge multiple CSV files into one.

    Args:
        file_paths: List of CSV file paths to merge
        output_path: Path for output CSV file
        **kwargs: Additional arguments for merge operation

    Returns:
        True if merge succeeded, False otherwise
    """
    try:
        dataframes = []
        for file_path in file_paths:
            df = safe_read_csv(file_path)
            if not df.empty:
                dataframes.append(df)

        if not dataframes:
            logger.error("No valid dataframes to merge")
            return False

        merged = pd.concat(dataframes, ignore_index=True, **kwargs)
        return safe_write_csv(merged, output_path)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.error(f"Error merging CSV files: {e}")
        return False
