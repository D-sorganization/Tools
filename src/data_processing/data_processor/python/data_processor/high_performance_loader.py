"""
High-Performance Data Loading System

Optimized for chemical plant data processing with:
- Parallel file reading
- Caching and memoization
- Lazy loading strategies
- Memory-efficient operations
- Background processing
- Smart signal detection
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Import logging
try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger  # type: ignore

# Import security utilities
try:
    from .security_utils import FileSizeError, check_file_size
except ImportError:
    from security_utils import FileSizeError, check_file_size  # type: ignore


try:
    from utils.file_utils import safe_read_json
except ImportError:
    import json

try:
    from utils.csv_utils import safe_read_csv, safe_write_csv
except ImportError:
    import pandas as pd
    from pathlib import Path
    def safe_read_csv(path, default=None, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            return default if default is not None else pd.DataFrame()
    def safe_write_csv(df, path, create_parents=True, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        safe_write_csv(df, path, **kwargs)
    def safe_read_json(path, default=None):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default
# Module logger
logger = get_logger(__name__)


@dataclass
class FileMetadata:
    """Metadata for a data file."""

    path: str
    size_bytes: int
    modified_time: float
    hash: str
    signal_count: int
    signals: set[str]
    sample_data: pd.DataFrame | None = None


@dataclass
class LoadingConfig:
    """Configuration for data loading optimization."""

    max_workers: int = -1  # -1 for all cores
    cache_enabled: bool = True
    parallel_loading: bool = True
    lazy_loading: bool = True
    sample_size: int = 1000  # Number of rows to sample for analysis
    cache_ttl: int = 3600  # Cache time-to-live in seconds
    max_files_per_batch: int = 50
    use_process_pool: bool = (
        False  # Use processes instead of threads for CPU-bound tasks
    )


class HighPerformanceDataLoader:
    """
    High-performance data loading system optimized for chemical plant data.

    Features:
    - Parallel file reading using ThreadPoolExecutor/ProcessPoolExecutor
    - Intelligent caching with file modification time checking
    - Lazy loading for large datasets
    - Memory-efficient operations
    - Background processing with progress callbacks
    - Smart signal detection and deduplication
    """

    def __init__(self, config: LoadingConfig | None = None) -> None:
        """Initialize the high-performance data loader."""
        self.config = config or LoadingConfig()
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.file_metadata_cache: dict[str, FileMetadata] = {}
        self._loading_lock = threading.Lock()

        # Set up parallel processing
        if self.config.max_workers == -1:
            self.config.max_workers = min(32, (os.cpu_count() or 1) + 4)

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling so process pools can serialize the loader safely."""
        state = self.__dict__.copy()
        # threading.Lock objects are not picklable; drop and recreate on restore
        state["_loading_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore loader state after pickling."""
        self.__dict__.update(state)
        self._loading_lock = threading.Lock()

    def load_signals_from_files(
        self,
        file_paths: list[str],
        progress_callback: Callable[..., Any] | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> tuple[set[str], dict[str, FileMetadata]]:
        """
        Load signals from multiple files with high performance.

        Args:
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates
            cancel_flag: Optional threading.Event to signal cancellation

        Returns:
            Tuple of (unique_signals_set, file_metadata_dict)
        """
        if not file_paths:
            return set(), {}

        logger.info(f"High-Performance Loading: {len(file_paths)} files")

        # Step 1: Collect file metadata in parallel
        file_metadata = self._collect_file_metadata_parallel(
            file_paths,
            progress_callback,
            cancel_flag,
        )

        if cancel_flag and cancel_flag.is_set():
            return set(), {}

        # Step 2: Extract signals from metadata
        all_signals = set()
        for metadata in file_metadata.values():
            all_signals.update(metadata.signals)

        logger.info(
            f"Found {len(all_signals)} unique signals from {len(file_metadata)} files",
        )

        return all_signals, file_metadata

    def _collect_file_metadata_parallel(
        self,
        file_paths: list[str],
        progress_callback: Callable[..., Any] | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> dict[str, FileMetadata]:
        """Collect file metadata in parallel."""
        file_metadata = {}

        # Use appropriate executor based on configuration
        executor_class = (
            ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all file processing tasks
            future_to_path = {
                executor.submit(self._get_file_metadata, file_path): file_path
                for file_path in file_paths
            }

            completed = 0
            total = len(file_paths)

            # Collect results as they complete
            for future in as_completed(future_to_path):
                if cancel_flag and cancel_flag.is_set():
                    # Cancel remaining tasks
                    for f in future_to_path:
                        f.cancel()
                    break

                file_path = future_to_path[future]
                try:
                    metadata = future.result()
                    if metadata:
                        file_metadata[file_path] = metadata

                    completed += 1
                    if progress_callback:
                        progress_callback(
                            completed,
                            total,
                            f"Processed {Path(file_path).name}",
                        )

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                    completed += 1

        return file_metadata

    def _get_file_metadata(self, file_path: str) -> FileMetadata | None:
        """Get metadata for a single file with caching."""
        if not Path(file_path).exists():
            return None

        # Check file size for security
        try:
            check_file_size(file_path)
        except FileSizeError as e:
            logger.warning(f"Skipping file due to size limit: {file_path} - {e}")
            return None

        # Check cache first
        if self.config.cache_enabled:
            cached_metadata = self._get_cached_metadata(file_path)
            if cached_metadata and self._is_cache_valid(cached_metadata, file_path):
                return cached_metadata

        # Generate metadata
        try:
            stat = os.stat(file_path)
            file_hash = self._calculate_file_hash(file_path)

            # Read file header and sample data
            signals, sample_data = self._read_file_header_and_sample(file_path)

            metadata = FileMetadata(
                path=file_path,
                size_bytes=stat.st_size,
                modified_time=stat.st_mtime,
                hash=file_hash,
                signal_count=len(signals),
                signals=signals,
                sample_data=sample_data,
            )

            # Cache the metadata
            if self.config.cache_enabled:
                self._cache_metadata(metadata)

            return metadata

        except Exception as e:
            logger.error(f"Error reading metadata for {file_path}: {e}", exc_info=True)
            return None

    def _read_file_header_and_sample(
        self,
        file_path: str,
    ) -> tuple[set[str], pd.DataFrame | None]:
        """Read file header and sample data efficiently (avoid duplicate file reads)."""
        try:
            # Read sample data or just header in a single operation
            sample_df = None
            if self.config.lazy_loading:
                try:
                    # Read a small sample for data type analysis (includes header)
                    sample_df = safe_read_csv(file_path, nrows=self.config.sample_size,
                        low_memory=False,
                    )
                    signals = set(sample_df.columns)
                except Exception as e:
                    logger.warning(f"Could not read sample data from {file_path}: {e}")
                    # Fallback to header-only read
                    header_df = safe_read_csv(file_path, nrows=0)
                    signals = set(header_df.columns)
            else:
                # Just read header when not lazy loading
                header_df = safe_read_csv(file_path, nrows=0)
                signals = set(header_df.columns)

            return signals, sample_df

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return set(), None

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate a hash of the file for caching."""
        try:
            # Use file size and modification time for quick hash
            stat = os.stat(file_path)
            content = f"{stat.st_size}_{stat.st_mtime}_{Path(file_path).name}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return "unknown"

    def _get_cached_metadata(self, file_path: str) -> FileMetadata | None:
        """Get cached metadata for a file (using secure JSON storage)."""
        try:
            cache_file = (
                self.cache_dir / f"{hashlib.md5(file_path.encode()).hexdigest()}.json"
            )
            if cache_file.exists():
                data = safe_read_json(cache_file, default=None)
                    # Convert signals list back to set
                    data["signals"] = set(data["signals"])
                    # Sample data is not cached (too large for JSON)
                    data["sample_data"] = None
                    return FileMetadata(**data)
        except Exception as e:
            logger.error(f"Error reading cache for {file_path}: {e}", exc_info=True)
        return None

    def _cache_metadata(self, metadata: FileMetadata) -> None:
        """Cache file metadata (using secure JSON storage)."""
        try:
            cache_file = (
                self.cache_dir
                / f"{hashlib.md5(metadata.path.encode()).hexdigest()}.json"
            )
            with open(cache_file, "w", encoding="utf-8") as f:
                data = asdict(metadata)
                # Convert set to list for JSON serialization
                data["signals"] = list(data["signals"])
                # Don't cache sample_data (too large for JSON, will be regenerated)
                data["sample_data"] = None
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(
                f"Error caching metadata for {metadata.path}: {e}",
                exc_info=True,
            )

    def _is_cache_valid(self, metadata: FileMetadata, file_path: str) -> bool:
        """Check if cached metadata is still valid."""
        try:
            current_stat = os.stat(file_path)
            return (
                metadata.size_bytes == current_stat.st_size
                and metadata.modified_time == current_stat.st_mtime
            )
        except Exception:
            return False

    def load_file_data(
        self,
        file_path: str,
        signals: list[str] | None = None,
        max_rows: int | None = None,
    ) -> pd.DataFrame | None:
        """
        Load actual data from a file with optimizations.

        Args:
            file_path: Path to the file
            signals: Optional list of signals to load
            max_rows: Optional maximum number of rows to load

        Returns:
            DataFrame with the data or None if error
        """
        # Check file size for security
        try:
            check_file_size(file_path)
        except FileSizeError as e:
            logger.exception(f"Cannot load file due to size limit: {file_path} - {e}")
            return None

        try:
            # Use optimized pandas reading
            read_kwargs = {
                "low_memory": False,
                "engine": "c",  # Use C engine for speed
            }

            if max_rows:
                read_kwargs["nrows"] = max_rows

            if signals:
                read_kwargs["usecols"] = signals

            df = safe_read_csv(file_path, **read_kwargs)

            # Optimize data types
            return self._optimize_dtypes(df)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
            return None

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency.

        Uses single-pass optimization for better performance.
        """
        try:
            # Single-pass optimization for all columns
            for col in df.columns:
                col_dtype = df[col].dtype

                # Handle object and string columns (convert_dtypes makes them string)
                if pd.api.types.is_string_dtype(col_dtype) or col_dtype == "object":
                    # Optimization: Check sample to avoid expensive full-column attempts
                    sample = df[col].dropna().iloc[:100]
                    if len(sample) == 0:
                        continue

                    is_numeric = False

                    # 1. Try numeric conversion if sample looks numeric
                    try:
                        # Fast check with sample
                        pd.to_numeric(sample, errors="raise")

                        # Sample passed, try full column
                        numeric_col = pd.to_numeric(df[col], errors="coerce")

                        # Validate enough values were converted
                        if numeric_col.notna().sum() / len(df[col]) > 0.5:
                            df[col] = numeric_col
                            col_dtype = df[col].dtype
                            is_numeric = True
                    except (ValueError, TypeError):
                        pass

                    # 2. Try datetime conversion if not numeric
                    if not is_numeric:
                        try:
                            # Try converting sample first
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pd.to_datetime(sample, errors="raise")

                            # Sample passed, try full column
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            col_dtype = df[col].dtype
                        except (ValueError, TypeError):
                            # 3. Try categorical if string and low cardinality
                            # Only if dataset is large enough to matter
                            if len(df) > 1000:
                                # Heuristic: check for duplicates in sample
                                if sample.nunique() < len(sample):
                                    # Check true cardinality
                                    n_unique = df[col].nunique()
                                    if n_unique / len(df) < 0.3:  # < 30% unique
                                        df[col] = df[col].astype("category")

                # Downcast numeric types in same pass
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast="float")

            return df

        except Exception as e:
            logger.warning(f"Could not optimize dtypes: {e}")
            return df

    def batch_load_files(
        self,
        file_paths: list[str],
        signals: list[str] | None = None,
        progress_callback: Callable[..., Any] | None = None,
        cancel_flag: threading.Event | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load multiple files in parallel batches.

        Args:
            file_paths: List of file paths to load
            signals: Optional list of signals to load from each file
            progress_callback: Optional callback for progress updates
            cancel_flag: Optional threading.Event to signal cancellation

        Returns:
            Dictionary mapping file paths to DataFrames
        """
        results = {}

        # Process files in batches
        batch_size = min(self.config.max_files_per_batch, len(file_paths))

        for i in range(0, len(file_paths), batch_size):
            if cancel_flag and cancel_flag.is_set():
                break

            batch_paths = file_paths[i : i + batch_size]

            # Load batch in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.load_file_data, path, signals): path
                    for path in batch_paths
                }

                for future in as_completed(future_to_path):
                    if cancel_flag and cancel_flag.is_set():
                        break

                    file_path = future_to_path[future]
                    try:
                        df = future.result()
                        if df is not None:
                            results[file_path] = df
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}", exc_info=True)

            # Update progress
            if progress_callback:
                progress_callback(
                    i + len(batch_paths),
                    len(file_paths),
                    f"Loaded batch {i // batch_size + 1}",
                )

        return results

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        try:
            # Clear both JSON and legacy pickle files
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}", exc_info=True)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            json_files = list(self.cache_dir.glob("*.json"))
            pkl_files = list(self.cache_dir.glob("*.pkl"))
            cache_files = json_files + pkl_files
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_files": len(cache_files),
                "json_files": len(json_files),
                "legacy_pkl_files": len(pkl_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}", exc_info=True)
            return {}


# Convenience functions for backward compatibility
def load_signals_fast(file_paths: list[str], **kwargs: Any) -> set[str]:
    """Fast signal loading function."""
    loader = HighPerformanceDataLoader()
    signals, _ = loader.load_signals_from_files(file_paths, **kwargs)
    return signals


def load_data_fast(
    file_paths: list[str],
    signals: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Fast data loading function."""
    loader = HighPerformanceDataLoader()
    return loader.batch_load_files(file_paths, signals, **kwargs)
