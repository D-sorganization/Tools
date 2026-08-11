"""Atomic filesystem adapter for validated workspace documents."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .workspace_document import (
    WorkspaceDocument,
    workspace_from_json,
    workspace_to_json,
)


def _destination_path(destination: str | Path) -> Path:
    path = Path(destination)
    if not path.name:
        raise ValueError("workspace destination must name a file")
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"workspace parent directory does not exist: {parent}")
    if path.exists() and not path.is_file():
        raise IsADirectoryError(f"workspace destination is not a file: {path}")
    return path


def read_workspace(source: str | Path) -> WorkspaceDocument:
    """Read and fully validate one workspace before returning it.

    Args:
        source: Existing UTF-8 workspace JSON file.

    Returns:
        A completely parsed immutable workspace document.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"workspace file does not exist: {path}")
    return workspace_from_json(path.read_text(encoding="utf-8"))


def write_workspace_atomic(
    document: WorkspaceDocument, destination: str | Path | None
) -> bool:
    """Atomically replace a workspace file after complete serialization.

    Args:
        document: Valid whole-workspace document.
        destination: Destination path, or ``None`` for a cancelled operation.

    Returns:
        ``True`` after replacement; ``False`` when the operation was cancelled.

    Raises:
        OSError: If staging, flushing, or atomic replacement fails. The prior
            destination remains untouched whenever replacement did not occur.
    """
    if destination is None:
        return False
    serialized = workspace_to_json(document)
    path = _destination_path(destination)
    _atomic_replace_text(serialized, path)
    return True


def _atomic_replace_text(text: str, path: Path) -> None:
    """Stage, flush, and replace one validated destination path."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(text: str, destination: str | Path | None) -> bool:
    """Atomically replace a UTF-8 text file, or return false on cancellation."""
    if destination is None:
        return False
    if not isinstance(text, str):
        raise TypeError("workspace export must be text")
    path = _destination_path(destination)
    _atomic_replace_text(text, path)
    return True


__all__ = ["read_workspace", "write_text_atomic", "write_workspace_atomic"]
