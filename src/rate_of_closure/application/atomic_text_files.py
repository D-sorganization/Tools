"""Small atomic UTF-8 text-file boundary shared by application documents."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _destination_path(destination: str | Path, document_name: str) -> Path:
    path = Path(destination)
    if not path.name:
        raise ValueError(f"{document_name} destination must name a file")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"{document_name} parent directory does not exist: {path.parent}"
        )
    if path.exists() and not path.is_file():
        raise IsADirectoryError(f"{document_name} destination is not a file: {path}")
    return path


def write_utf8_text_atomic(
    text: str,
    destination: str | Path | None,
    *,
    document_name: str,
) -> bool:
    """Flush and atomically replace one UTF-8 text file.

    ``None`` represents a cancelled chooser. Serialization must happen before
    this function is called so a validation failure cannot touch the target.
    """
    if destination is None:
        return False
    if not isinstance(text, str):
        raise TypeError("atomic file content must be text")
    path = _destination_path(destination, document_name)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


__all__ = ["write_utf8_text_atomic"]
