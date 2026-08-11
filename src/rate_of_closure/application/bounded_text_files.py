"""Shared bounded file-snapshot readers for strict native JSON adapters."""

from __future__ import annotations

from pathlib import Path


def read_bounded_utf8(
    path: Path,
    max_bytes: int,
    document_name: str,
) -> str:
    """Read one handle snapshot with a sentinel byte and strict UTF-8."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(document_name) is not str or not document_name.strip():
        raise ValueError("document_name must be nonblank text")
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"{document_name} exceeds maximum wire size")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{document_name} must be valid UTF-8") from exc


__all__ = ["read_bounded_utf8"]
