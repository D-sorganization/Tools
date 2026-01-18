"""
Performance utilities for optimizing common operations across the Tools repository.

This module provides high-performance implementations of common operations
to replace slower patterns found throughout the codebase.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class OptimizedFileScanner:
    """
    High-performance file scanner with parallel processing and caching.

    Replaces slow sequential os.walk() operations with parallel scanning.
    """

    def __init__(self, max_workers: int = -1) -> None:
        """Initialize the scanner with specified number of workers."""
        self.max_workers = (
            max_workers if max_workers != -1 else min(32, (os.cpu_count() or 1) + 4)
        )
        self._cache: dict[str, tuple[float, list[Path]]] = {}
        self._cache_lock = threading.Lock()

    def scan_directory_parallel(
        self, root_path: Path, pattern: str = "*", max_depth: int = 10
    ) -> Generator[Path, None, None]:
        """
        Scan directory in parallel for better performance on large directories.

        Args:
            root_path: Root directory to scan
            pattern: File pattern to match (e.g., "*.py", "*")
            max_depth: Maximum recursion depth

        Yields:
            Path objects for matching files
        """
        if not root_path.exists():
            return

        # Check cache first
        cache_key = f"{root_path}:{pattern}:{max_depth}"
        with self._cache_lock:
            if cache_key in self._cache:
                cached_time, cached_files = self._cache[cache_key]
                # Cache valid for 60 seconds
                if (os.path.getmtime(root_path) - cached_time) < 60:
                    yield from cached_files
                    return

        # Parallel directory scanning
        found_files = []

        def scan_subdirectory(directory: Path, depth: int) -> list[Path]:
            """Scan a single subdirectory."""
            if depth > max_depth:
                return []

            files = []
            try:
                for item in directory.iterdir():
                    if item.is_file() and item.match(pattern):
                        files.append(item)
                    elif item.is_dir() and depth < max_depth:
                        # Don't recurse too deep in parallel to avoid thread explosion
                        if depth < 2:
                            files.extend(scan_subdirectory(item, depth + 1))
                        else:
                            # Use sequential scanning for deeper levels
                            files.extend(item.rglob(pattern))
            except (PermissionError, OSError):
                pass  # Skip inaccessible directories

            return files

        # Start with immediate subdirectories in parallel
        subdirs = [item for item in root_path.iterdir() if item.is_dir()]

        if len(subdirs) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(subdirs))
            ) as executor:
                future_to_dir = {
                    executor.submit(scan_subdirectory, subdir, 1): subdir
                    for subdir in subdirs
                }

                for future in as_completed(future_to_dir):
                    try:
                        found_files.extend(future.result())
                    except Exception:
                        pass  # Skip failed directories
        else:
            # Sequential fallback for small directories
            for subdir in subdirs:
                found_files.extend(scan_subdirectory(subdir, 1))

        # Add files in root directory
        try:
            for item in root_path.iterdir():
                if item.is_file() and item.match(pattern):
                    found_files.append(item)
        except (PermissionError, OSError):
            pass

        # Cache results
        with self._cache_lock:
            self._cache[cache_key] = (os.path.getmtime(root_path), found_files)

        yield from found_files


class FastHasher:
    """
    Optimized file hashing with two-pass strategy for deduplication.

    First pass: Fast hash using file size + first/last chunks
    Second pass: Full hash only for potential duplicates
    """

    CHUNK_SIZE = 64 * 1024  # 64KB chunks
    FAST_CHUNK_SIZE = 8 * 1024  # 8KB for fast hashing

    @classmethod
    def fast_hash(cls, file_path: Path) -> str:
        """
        Generate fast hash using file size and first/last chunks.

        This is much faster than full file hashing and catches most duplicates.
        """
        try:
            stat = file_path.stat()
            size = stat.st_size

            if size == 0:
                return f"empty_{stat.st_mtime}"

            # For small files, just hash the whole thing
            if size <= cls.FAST_CHUNK_SIZE * 2:
                return cls.full_hash(file_path)

            # For larger files, hash size + first chunk + last chunk
            hasher = hashlib.md5()
            hasher.update(str(size).encode())

            with file_path.open("rb") as f:
                # First chunk
                first_chunk = f.read(cls.FAST_CHUNK_SIZE)
                hasher.update(first_chunk)

                # Last chunk
                f.seek(-cls.FAST_CHUNK_SIZE, 2)
                last_chunk = f.read(cls.FAST_CHUNK_SIZE)
                hasher.update(last_chunk)

            return f"fast_{hasher.hexdigest()}"

        except OSError:
            return f"error_{file_path.name}"

    @classmethod
    def full_hash(cls, file_path: Path) -> str:
        """Generate full file hash for exact duplicate detection."""
        try:
            hasher = hashlib.md5()
            with file_path.open("rb") as f:
                while chunk := f.read(cls.CHUNK_SIZE):
                    hasher.update(chunk)
            return f"full_{hasher.hexdigest()}"
        except OSError:
            return f"error_{file_path.name}"


class MemoryOptimizedProcessor:
    """
    Memory-efficient processing utilities for large datasets.
    """

    @staticmethod
    def chunked_processing(
        items: list[Any],
        chunk_size: int = 1000,
        processor_func: Callable[..., Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Process large lists in chunks to avoid memory issues.

        Args:
            items: List of items to process
            chunk_size: Number of items per chunk
            processor_func: Function to apply to each chunk

        Yields:
            Processed results
        """
        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            if processor_func is not None:
                yield processor_func(chunk)
            else:
                yield chunk

    @staticmethod
    def lazy_file_reader(
        file_path: Path, chunk_size: int = 8192
    ) -> Generator[str, None, None]:
        """
        Read large files lazily to avoid loading everything into memory.

        Args:
            file_path: Path to file
            chunk_size: Size of each chunk to read

        Yields:
            File content chunks
        """
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
        except OSError:
            pass


# Global instances for easy access
file_scanner = OptimizedFileScanner()
fast_hasher = FastHasher()
memory_processor = MemoryOptimizedProcessor()
