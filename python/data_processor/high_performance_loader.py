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

import os
import time
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import pickle

# Import constants


@dataclass
class FileMetadata:
    """Metadata for a data file."""
    path: str
    size_bytes: int
    modified_time: float
    hash: str
    signal_count: int
    signals: Set[str]
    sample_data: Optional[pd.DataFrame] = None


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
    use_process_pool: bool = False  # Use processes instead of threads for CPU-bound tasks


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
    
    def __init__(self, config: LoadingConfig = None):
        """Initialize the high-performance data loader."""
        self.config = config or LoadingConfig()
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.file_metadata_cache: Dict[str, FileMetadata] = {}
        self._loading_lock = threading.Lock()
        
        # Set up parallel processing
        if self.config.max_workers == -1:
            self.config.max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    def load_signals_from_files(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[callable] = None,
        cancel_flag: Optional[threading.Event] = None
    ) -> Tuple[Set[str], Dict[str, FileMetadata]]:
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
        
        print(f"🚀 High-Performance Loading: {len(file_paths)} files")
        
        # Step 1: Collect file metadata in parallel
        file_metadata = self._collect_file_metadata_parallel(
            file_paths, progress_callback, cancel_flag
        )
        
        if cancel_flag and cancel_flag.is_set():
            return set(), {}
        
        # Step 2: Extract signals from metadata
        all_signals = set()
        for metadata in file_metadata.values():
            all_signals.update(metadata.signals)
        
        print(f"✅ Found {len(all_signals)} unique signals from {len(file_metadata)} files")
        
        return all_signals, file_metadata
    
    def _collect_file_metadata_parallel(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[callable] = None,
        cancel_flag: Optional[threading.Event] = None
    ) -> Dict[str, FileMetadata]:
        """Collect file metadata in parallel."""
        file_metadata = {}
        
        # Use appropriate executor based on configuration
        executor_class = ProcessPoolExecutor if self.config.use_process_pool else ThreadPoolExecutor
        
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
                        progress_callback(completed, total, f"Processed {os.path.basename(file_path)}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    completed += 1
        
        return file_metadata
    
    def _get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for a single file with caching."""
        if not os.path.exists(file_path):
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
                sample_data=sample_data
            )
            
            # Cache the metadata
            if self.config.cache_enabled:
                self._cache_metadata(metadata)
            
            return metadata
            
        except Exception as e:
            print(f"Error reading metadata for {file_path}: {e}")
            return None
    
    def _read_file_header_and_sample(self, file_path: str) -> Tuple[Set[str], Optional[pd.DataFrame]]:
        """Read file header and sample data efficiently."""
        try:
            # Read only header first
            header_df = pd.read_csv(file_path, nrows=0)
            signals = set(header_df.columns)
            
            # Read sample data for analysis
            sample_df = None
            if self.config.lazy_loading:
                try:
                    # Read a small sample for data type analysis
                    sample_df = pd.read_csv(
                        file_path, 
                        nrows=self.config.sample_size,
                        low_memory=False
                    )
                except Exception as e:
                    print(f"Warning: Could not read sample data from {file_path}: {e}")
            
            return signals, sample_df
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return set(), None
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate a hash of the file for caching."""
        try:
            # Use file size and modification time for quick hash
            stat = os.stat(file_path)
            content = f"{stat.st_size}_{stat.st_mtime}_{os.path.basename(file_path)}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def _get_cached_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get cached metadata for a file."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(file_path.encode()).hexdigest()}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error reading cache for {file_path}: {e}")
        return None
    
    def _cache_metadata(self, metadata: FileMetadata) -> None:
        """Cache file metadata."""
        try:
            cache_file = self.cache_dir / f"{hashlib.md5(metadata.path.encode()).hexdigest()}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(metadata, f)
        except Exception as e:
            print(f"Error caching metadata for {metadata.path}: {e}")
    
    def _is_cache_valid(self, metadata: FileMetadata, file_path: str) -> bool:
        """Check if cached metadata is still valid."""
        try:
            current_stat = os.stat(file_path)
            return (
                metadata.size_bytes == current_stat.st_size and
                metadata.modified_time == current_stat.st_mtime
            )
        except Exception:
            return False
    
    def load_file_data(
        self, 
        file_path: str, 
        signals: Optional[List[str]] = None,
        max_rows: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load actual data from a file with optimizations.
        
        Args:
            file_path: Path to the file
            signals: Optional list of signals to load
            max_rows: Optional maximum number of rows to load
            
        Returns:
            DataFrame with the data or None if error
        """
        try:
            # Use optimized pandas reading
            read_kwargs = {
                'low_memory': False,
                'engine': 'c',  # Use C engine for speed
            }
            
            if max_rows:
                read_kwargs['nrows'] = max_rows
            
            if signals:
                read_kwargs['usecols'] = signals
            
            df = pd.read_csv(file_path, **read_kwargs)
            
            # Optimize data types
            df = self._optimize_dtypes(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        try:
            # Convert object columns to appropriate types
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        # Column could not be converted to numeric; leave as-is and try datetime next
                        pass
                    
                    # Try to convert to datetime
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except (ValueError, TypeError):
                            # Not all object columns are datetimes; safe to leave as object if conversion fails
                            pass
            
            # Downcast integer columns
            for col in df.select_dtypes(include=['integer']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Downcast float columns
            for col in df.select_dtypes(include=['floating']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            return df
            
        except Exception as e:
            print(f"Warning: Could not optimize dtypes: {e}")
            return df
    
    def batch_load_files(
        self, 
        file_paths: List[str], 
        signals: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
        cancel_flag: Optional[threading.Event] = None
    ) -> Dict[str, pd.DataFrame]:
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
            
            batch_paths = file_paths[i:i + batch_size]
            
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
                        print(f"Error loading {file_path}: {e}")
            
            # Update progress
            if progress_callback:
                progress_callback(i + len(batch_paths), len(file_paths), f"Loaded batch {i//batch_size + 1}")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {}


# Convenience functions for backward compatibility
def load_signals_fast(file_paths: List[str], **kwargs) -> Set[str]:
    """Fast signal loading function."""
    loader = HighPerformanceDataLoader()
    signals, _ = loader.load_signals_from_files(file_paths, **kwargs)
    return signals


def load_data_fast(file_paths: List[str], signals: Optional[List[str]] = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """Fast data loading function."""
    loader = HighPerformanceDataLoader()
    return loader.batch_load_files(file_paths, signals, **kwargs)
