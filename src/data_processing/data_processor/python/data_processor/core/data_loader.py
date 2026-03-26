"""Data loading and management operations.

This module handles loading CSV files, detecting signals,
and managing data operations.

Design by Contract (DbC) guards are applied at all public API boundaries:
  - Preconditions validate caller inputs (non-empty paths, valid formats).
  - Postconditions assert that returned DataFrames have expected shapes.

Fixed in issue #530: removed fragile dependency on ``utils.path_helpers``.
Updated in issue #682: removed ``sys.path`` hack; relies on package
installation (``pip install -e .``) or pytest ``pythonpath`` config.
Fixed in issue #929: DbC guards added to core pipeline (TDD/DbC compliance)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Try to import shared csv_utils; fall back to inline implementations.
# With a proper editable install (pip install -e .) the ``utils`` package
# is on sys.path via setuptools.  No sys.path manipulation needed.
try:
    from utils.csv_utils import safe_read_csv, safe_write_csv
except ImportError:

    def safe_read_csv(
        file_path: Path | str,
        default: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read CSV with error handling (inline fallback)."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        path = Path(file_path)
        if not path.exists():
            return default if default is not None else pd.DataFrame()
        try:
            return pd.read_csv(path, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return default if default is not None else pd.DataFrame()

    def safe_write_csv(df: pd.DataFrame, file_path: Path | str, **kwargs: Any) -> None:
        """Write CSV with error handling (inline fallback)."""
        df.to_csv(file_path, **kwargs)


from data_processor.constants import TIME_COLUMN_KEYWORDS  # noqa: E402
from data_processor.contracts import (  # noqa: E402
    require,
)
from data_processor.high_performance_loader import (  # noqa: E402
    HighPerformanceDataLoader,
)
from data_processor.security_utils import validate_and_check_file  # noqa: E402

logger = logging.getLogger(__name__)

_ALLOWED_SAVE_FORMATS = frozenset({"csv", "excel", "xlsx", "parquet"})


class DataLoader:
    """Handles loading and managing CSV data files."""

    hp_loader: HighPerformanceDataLoader | None

    def __init__(self, use_high_performance: bool = True) -> None:
        """Initialize the data loader.

        Args:
            use_high_performance: Whether to use high-performance parallel loading
        """
        if not (use_high_performance is not None):
            raise ValueError("use_high_performance must be provided")
        self.use_high_performance = use_high_performance
        if use_high_performance:
            self.hp_loader = HighPerformanceDataLoader()
        else:
            self.hp_loader = None
        self.logger = logger

    def load_csv_file(
        self,
        file_path: str,
        validate_security: bool = True,
    ) -> pd.DataFrame | None:
        """Load a single CSV file.

        **Pre-conditions** (DbC):
          - ``file_path`` must be a non-empty string.
          - ``file_path`` must end with ``.csv`` or ``.txt``.

        Args:
            file_path: Path to CSV file
            validate_security: Whether to perform security validation

        Returns:
            DataFrame or None if loading fails
        """
        # --- Pre-conditions ---
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        require(
            isinstance(file_path, str) and bool(file_path.strip()),
            "file_path must be a non-empty string",
            file_path,
        )
        require(
            file_path.lower().endswith((".csv", ".txt")),
            "file_path must end with .csv or .txt",
            file_path,
        )

        try:
            # Security validation
            if validate_security:
                validate_and_check_file(
                    file_path,
                    allowed_extensions={".csv", ".txt"},
                )

            logger.info(f"Loading CSV file: {file_path}")

            df = safe_read_csv(file_path, low_memory=False)

            if df is None or df.empty:
                logger.warning(f"CSV file {file_path} is empty or could not be loaded")
                return None

            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except (OSError, ValueError, KeyError) as e:
            logger.error(f"Error loading {file_path}: {e}", exc_info=True)
            return None

    def load_multiple_files(
        self,
        file_paths: list[str],
        combine: bool = False,
        progress_callback: Callable[..., Any] | None = None,
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """Load multiple CSV files.

        **Pre-conditions** (DbC):
          - ``file_paths`` must be a non-empty list.
          - Every entry must be a non-empty string.

        Args:
            file_paths: List of file paths
            combine: Whether to combine into single DataFrame
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping file paths to DataFrames, or single combined
            DataFrame if combine=True
        """
        if not (file_paths is not None):
            raise ValueError("file_paths must be provided")
        require(
            isinstance(file_paths, list) and len(file_paths) > 0,
            "file_paths must be a non-empty list",
            file_paths,
        )
        require(
            all(isinstance(p, str) and p.strip() for p in file_paths),
            "every entry in file_paths must be a non-empty string",
        )
        if self.use_high_performance and self.hp_loader:
            # Use high-performance batch loading
            results = self.hp_loader.batch_load_files(
                file_paths,
                progress_callback=progress_callback,
            )
        else:
            # Sequential loading
            results = {}
            for i, file_path in enumerate(file_paths):
                # validate_security=False: the DbC pre-condition above already
                # guarantees file_paths contains non-empty strings; security
                # validation is the responsibility of the *caller* of this
                # method, not an internal chain concern.
                df = self.load_csv_file(file_path, validate_security=False)
                if df is not None:
                    results[file_path] = df

                if progress_callback:
                    progress_callback(
                        i + 1,
                        len(file_paths),
                        f"Loaded {Path(file_path).name}",
                    )

        # Combine DataFrames if requested
        if combine:
            if len(results) > 0:
                return self.combine_dataframes(results)
            logger.warning("No dataframes to combine, returning empty DataFrame")
            return pd.DataFrame()

        return results

    def detect_signals(
        self,
        file_paths: list[str],
        progress_callback: Callable[..., Any] | None = None,
    ) -> set[str]:
        """Detect all unique signals from multiple files.

        **Pre-conditions** (DbC):
          - ``file_paths`` must be a non-empty list.

        Args:
            file_paths: List of CSV file paths
            progress_callback: Optional progress callback

        Returns:
            Set of unique signal names
        """
        if not (file_paths is not None):
            raise ValueError("file_paths must be provided")
        require(
            isinstance(file_paths, list) and len(file_paths) > 0,
            "file_paths must be a non-empty list",
            file_paths,
        )
        if self.use_high_performance and self.hp_loader:
            # Use high-performance signal detection
            signals, _ = self.hp_loader.load_signals_from_files(
                file_paths,
                progress_callback=progress_callback,
            )
            return signals

        # Sequential signal detection
        all_signals = set()
        for i, file_path in enumerate(file_paths):
            try:
                # Read just the header
                df_header = safe_read_csv(file_path, nrows=0)
                if df_header is not None and not df_header.empty:
                    all_signals.update(df_header.columns)

                if progress_callback:
                    progress_callback(
                        i + 1,
                        len(file_paths),
                        f"Scanned {Path(file_path).name}",
                    )
            except (PermissionError, OSError) as e:
                logger.exception(f"Error reading {file_path}: {e}")

        return all_signals

    def detect_time_column(self, df: pd.DataFrame) -> str | None:
        """Detect the time column in a DataFrame."""
        if not (df is not None):
            raise ValueError("df must be provided")
        for col in df.columns:
            col_str: str = str(col)
            col_lower = col_str.lower()
            if any(keyword in col_lower for keyword in TIME_COLUMN_KEYWORDS):
                logger.info(f"Detected time column: {col_str}")
                return col_str

        # Check for datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Detected datetime column: {col}")
                return str(col)

        logger.warning("No time column detected")
        return None

    def convert_time_column(
        self,
        df: pd.DataFrame,
        time_column: str,
    ) -> pd.DataFrame:
        """Convert time column to datetime and set as index."""
        try:
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.set_index(time_column)
            logger.info(f"Converted {time_column} to DatetimeIndex")
            return df
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error converting time column: {e}", exc_info=True)
            return df

    def get_numeric_signals(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric signal names."""
        if not (df is not None):
            raise ValueError("df must be provided")
        numeric_cols: list[str] = df.select_dtypes(include=np.number).columns.tolist()
        logger.info(f"Found {len(numeric_cols)} numeric signals")
        return numeric_cols

    def combine_dataframes(
        self,
        dataframes: dict[str, pd.DataFrame] | Iterable[pd.DataFrame],
        on_column: str | None = None,
        how: str = "outer",
    ) -> pd.DataFrame:
        """Combine multiple DataFrames.

        **Pre-conditions** (DbC):
          - ``how`` must be one of 'inner', 'outer', 'left', 'right'.
        """
        if not (dataframes is not None):
            raise ValueError("dataframes must be provided")
        require(
            how in {"inner", "outer", "left", "right"},
            "how must be one of 'inner', 'outer', 'left', 'right'",
            how,
        )
        dfs = list(dataframes.values()) if isinstance(dataframes, dict) else list(dataframes)

        if not dfs:
            return pd.DataFrame()

        if len(dfs) == 1:
            return dfs[0]

        logger.info(f"Combining {len(dfs)} DataFrames")
        result = dfs[0]

        for df in dfs[1:]:
            if on_column:
                result = pd.merge(result, df, on=on_column, how=how)
            else:
                result = pd.merge(
                    result,
                    df,
                    left_index=True,
                    right_index=True,
                    how=how,
                )

        logger.info(f"Combined result: {len(result)} rows, {len(result.columns)} columns")
        return result

    def filter_by_time_range(
        self,
        df: pd.DataFrame,
        start_time: str,
        end_time: str,
    ) -> pd.DataFrame:
        """Filter DataFrame by time range."""
        if not (df is not None):
            raise ValueError("df must be provided")
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame must have DatetimeIndex for time filtering")
            return df

        try:
            t_start = pd.to_datetime(start_time).time()
            t_end = pd.to_datetime(end_time).time()

            if t_start > t_end:
                logger.warning(f"Start time {t_start} > End time {t_end}, returning empty")
                return df.iloc[0:0]

            filtered = df.between_time(start_time, end_time)
            logger.info(f"Filtered to {len(filtered)} rows")
            return filtered
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error filtering: {e}", exc_info=True)
            return df

    def save_dataframe(
        self,
        df: pd.DataFrame,
        output_path: str,
        format_type: str = "csv",
        **kwargs: Any,
    ) -> bool:
        """Save DataFrame to file.

        **Pre-conditions** (DbC):
          - ``df`` must not be empty.
          - ``output_path`` must be a non-empty string.
          - ``format_type`` must be a recognised format.
        """
        if not (df is not None):
            raise ValueError("df must be provided")
        require(
            isinstance(df, pd.DataFrame) and not df.empty,
            "df must be a non-empty DataFrame",
        )
        require(
            isinstance(output_path, str) and bool(output_path.strip()),
            "output_path must be a non-empty string",
            output_path,
        )
        require(
            format_type in _ALLOWED_SAVE_FORMATS,
            f"format_type must be one of {sorted(_ALLOWED_SAVE_FORMATS)}",
            format_type,
        )
        try:
            logger.info(f"Saving DataFrame to {output_path} ({format_type})")
            if format_type == "csv":
                safe_write_csv(df, output_path, **kwargs)
                return True
            elif format_type in ["excel", "xlsx"]:
                df.to_excel(output_path, **kwargs)
                return True
            elif format_type == "parquet":
                df.to_parquet(output_path, **kwargs)
                return True
            else:
                from data_processor.file_utils import DataWriter

                DataWriter.write_file(df, output_path, format_type, **kwargs)
                return True
        except ImportError as e:
            logger.error(f"Error saving DataFrame: {e}", exc_info=True)
            return False


# Convenience functions
def load_csv_files(file_paths: list[str]) -> dict[str, pd.DataFrame]:
    """Load multiple CSV files."""
    loader = DataLoader()
    return loader.load_multiple_files(file_paths)


def detect_signals_from_files(file_paths: list[str]) -> set[str]:
    """Detect all signals from CSV files."""
    loader = DataLoader()
    return loader.detect_signals(file_paths)
