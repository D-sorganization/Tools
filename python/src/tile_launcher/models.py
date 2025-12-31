"""Data models for the PyQt6 tile launcher."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import json


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
        if not self.path.exists():
            return []

        content = self.path.read_text(encoding="utf-8")
        if not content.strip():
            return []

        return list(json.loads(content))

    def save(self, layout_ids: Sequence[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(list(layout_ids), indent=2)
        self.path.write_text(serialized, encoding="utf-8")


class InMemoryLayoutStore:
    """Testing helper that keeps layout state in memory only."""

    def __init__(self, layout: Iterable[str] | None = None) -> None:
        self._layout = list(layout) if layout else []

    def load(self) -> list[str]:
        return list(self._layout)

    def save(self, layout_ids: Sequence[str]) -> None:
        self._layout = list(layout_ids)
