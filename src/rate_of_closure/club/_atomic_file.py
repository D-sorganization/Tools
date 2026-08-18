"""Internal same-directory atomic byte replacement for club artifacts."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path

from rate_of_closure._contracts import ensure, require


def write_bytes_atomic(payload: bytes, path: str | Path) -> Path:
    """Atomically replace ``path`` with non-empty ``payload``.

    Preconditions:
        ``payload`` is non-empty bytes and ``path`` identifies a file.
    Postconditions:
        The destination exists and its size matches ``payload``.
    """
    require(isinstance(payload, bytes) and bool(payload), "payload must be bytes")
    require(isinstance(path, (str, Path)), "path must be a string or Path")
    target = Path(path)
    require(bool(target.name), "path must identify a file")
    temporary_path: Path | None = None
    replaced = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(target)
        replaced = True
    finally:
        if not replaced and temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    ensure(target.is_file(), "artifact destination must exist after replacement")
    ensure(
        target.stat().st_size == len(payload), "artifact destination size must match"
    )
    return target


__all__ = ["write_bytes_atomic"]
