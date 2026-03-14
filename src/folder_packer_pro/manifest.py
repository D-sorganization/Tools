"""Package manifest management for Folder Packer Pro.

Provides metadata tracking for packed files including checksums,
file listings, and serialization.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any


class PackageManifest:
    """Manage package manifest with metadata."""

    def __init__(self) -> None:
        """Initialize the manifest."""
        self.created_at = datetime.now()
        self.files: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.stats: defaultdict[str, int] = defaultdict(int)

    def add_file(self, file_path: str, size: int, checksum: str) -> None:
        """Add file to manifest."""
        assert file_path is not None, "file_path must be provided"
        self.files.append(
            {
                "path": file_path,
                "size": size,
                "checksum": checksum,
                "added_at": datetime.now().isoformat(),
            },
        )
        self.stats["total_files"] += 1
        self.stats["total_size"] += size

    def set_metadata(
        self,
        key: str,
        value: str | float | bool | None | list[Any] | dict[str, Any],
    ) -> None:
        """Set metadata value."""
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "created_at": self.created_at.isoformat(),
            "files": self.files,
            "metadata": self.metadata,
            "statistics": dict(self.stats),
        }

    def to_json(self) -> str:
        """Convert manifest to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PackageManifest:
        """Create manifest from JSON string."""
        assert json_str is not None, "json_str must be provided"
        data = json.loads(json_str)
        manifest = cls()
        manifest.created_at = datetime.fromisoformat(data["created_at"])
        manifest.files = data["files"]
        manifest.metadata = data["metadata"]
        manifest.stats = defaultdict(int, data["statistics"])
        return manifest
