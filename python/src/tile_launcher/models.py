"""Data models for the PyQt6 tile launcher."""

from __future__ import annotations

import json

# Use shared file utility
try:
    from utils.file_utils import safe_read_json
except ImportError:
    # Fallback
    def safe_read_json(path, default=None):
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from utils.compatibility import StrEnum


class LaunchType(StrEnum):
    """Supported launch mechanisms for apps."""

    PYTHON = "python"
    BAT = "bat"
    HTML = "html"
    FILE = "file"


@dataclass(frozen=True)
class AppDefinition:
    """Information describing an app that can be surfaced as a tile."""

    id: str
    name: str
    relative_path: str
    launch_type: LaunchType
    logo: str | None = None
    description: str | None = None

    def resolved_path(self, repository_root: Path) -> Path:
        """Return the absolute path for the app based on the repository root."""

        return repository_root.joinpath(self.relative_path).resolve()


class LayoutStore(Protocol):
    """Interface describing how layout selections are persisted."""

    def load(self) -> list[str]:
        """Return the ordered list of app identifiers currently stored."""

    def save(self, layout_ids: Sequence[str]) -> None:
        """Persist the ordered list of app identifiers."""


@dataclass
class FileLayoutStore:
    """Filesystem-backed implementation of the layout store."""

    path: Path

    def load(self) -> list[str]:
        """Load the layout from the JSON file."""
        if not self.path.exists():
            return []

        content = self.path.read_text(encoding="utf-8")
        if not content.strip():
            return []

        return list(safe_read_json(content, default=None))

    def save(self, layout_ids: Sequence[str]) -> None:
        """Save the layout to the JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(list(layout_ids), indent=2)
        self.path.write_text(serialized, encoding="utf-8")


class InMemoryLayoutStore:
    """Testing helper that keeps layout state in memory only."""

    def __init__(self, layout: Iterable[str] | None = None) -> None:
        """Initialize with an optional initial layout."""
        self._layout = list(layout) if layout else []

    def load(self) -> list[str]:
        """Return the current in-memory layout."""
        return list(self._layout)

    def save(self, layout_ids: Sequence[str]) -> None:
        """Update the in-memory layout."""
        self._layout = list(layout_ids)
