"""Enhanced duplicate detection with SHA256 hashing."""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class DuplicateFinder:
    """Finds duplicate PDF files based on size and SHA256 hash."""

    def __init__(self, directory: Path, recursive: bool = True):
        """
        Initialize duplicate finder.

        Args:
            directory: Root directory to scan
            recursive: Whether to scan subdirectories
        """
        assert directory is not None, "directory must be provided"
        self.directory = directory
        self.recursive = recursive

    def find_duplicates(self) -> dict[str, list[Path]]:
        """
        Finds duplicates based on file size and SHA256 hash.

        Returns:
            Dictionary where key is the hash and value is list of paths.
            Only includes entries where len(paths) > 1.
        """
        size_map: dict[int, list[Path]] = defaultdict(list)

        # 1. Group by size (fast pre-filter)
        pattern = "**/*.pdf" if self.recursive else "*.pdf"
        for file_path in self.directory.glob(pattern):
            if file_path.is_file() and not file_path.is_symlink():
                try:
                    size = file_path.stat().st_size
                    size_map[size].append(file_path)
                except OSError as e:
                    logger.error(f"Error accessing file {file_path}: {e}")

        # 2. Check hashes for files with same size
        duplicates: dict[str, list[Path]] = {}

        for files in size_map.values():
            if len(files) < 2:
                continue

            # Group by hash
            hash_map: dict[str, list[Path]] = defaultdict(list)
            for file_path in files:
                try:
                    file_hash = self._calculate_sha256(file_path)
                    hash_map[file_hash].append(file_path)
                except OSError as e:
                    logger.error(f"Error hashing file {file_path}: {e}")

            # Add to duplicates if collision found
            for file_hash, paths in hash_map.items():
                if len(paths) > 1:
                    # Sort by path length and name for deterministic ordering
                    duplicates[file_hash] = sorted(
                        paths, key=lambda p: (len(str(p)), p.name)
                    )

        return duplicates

    def _calculate_sha256(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 hash of a file.

        Args:
            file_path: Path to file
            chunk_size: Size of chunks to read (default 8KB)

        Returns:
            Hexadecimal SHA256 hash string
        """
        assert file_path is not None, "file_path must be provided"
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
