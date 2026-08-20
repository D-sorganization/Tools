"""Atomic filesystem adapter for validated workspace documents."""

from __future__ import annotations

from pathlib import Path

from .atomic_text_files import write_utf8_text_atomic
from .workspace_document import (
    WorkspaceDocument,
    workspace_from_json,
    workspace_to_json,
)


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
    # `bool(...)` is for the type checker: the changed-file MyPy gate runs
    # `--follow-imports=skip`, so `write_utf8_text_atomic` resolves to `Any`.
    return bool(
        write_utf8_text_atomic(serialized, destination, document_name="workspace")
    )


def write_text_atomic(text: str, destination: str | Path | None) -> bool:
    """Atomically replace a UTF-8 text file, or return false on cancellation.

    Args:
        text: Already-serialized document text.
        destination: Destination path, or ``None`` for a cancelled operation.

    Returns:
        ``True`` after replacement; ``False`` when the operation was cancelled.

    Raises:
        TypeError: If ``text`` is not a string.
        OSError: If staging, flushing, or atomic replacement fails.
    """
    # `bool(...)` for the same reason as `write_workspace_atomic` above.
    return bool(write_utf8_text_atomic(text, destination, document_name="workspace"))


__all__ = ["read_workspace", "write_text_atomic", "write_workspace_atomic"]
