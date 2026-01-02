import hashlib
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class DuplicateFinder:
    def __init__(self, directory: Path):
        self.directory = directory

    def find_duplicates(self) -> dict[str, list[Path]]:
        """
        Finds duplicates based on file size and MD5 hash.
        Returns a dictionary where key is the hash and value is list of paths.
        """
        size_map: dict[int, list[Path]] = defaultdict(list)

        # 1. Group by size
        for file_path in self.directory.glob("**/*.pdf"):
            if file_path.is_file():
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
                    file_hash = self._calculate_md5(file_path)
                    hash_map[file_hash].append(file_path)
                except OSError as e:
                    logger.error(f"Error hashing file {file_path}: {e}")

            # Add to duplicates if collision found
            for file_hash, paths in hash_map.items():
                if len(paths) > 1:
                    duplicates[file_hash] = paths

        return duplicates

    def _calculate_md5(self, file_path: Path, chunk_size: int = 8192) -> str:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
