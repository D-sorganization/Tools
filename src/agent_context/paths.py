"""Bounded filesystem reads for repository context, without executing sources."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 2_000_000


class CatalogError(ValueError):
    """A catalog, source or verification record cannot support its claims."""


def safe_path(root: Path, value: str, *, directory: bool = False) -> Path:
    """Resolve a repository-relative path; refuse escapes and absent targets."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise CatalogError(f"Invalid relative path: {value!r}")
    parts = PurePosixPath(value)
    if parts.is_absolute() or ".." in parts.parts or value != parts.as_posix():
        raise CatalogError(f"Invalid relative path: {value!r}")
    path = root / value
    if not path.resolve().is_relative_to(root.resolve()):
        raise CatalogError(f"Escaping repository path: {value}")
    if not path.is_file() and not (directory and path.is_dir()):
        raise CatalogError(f"Missing source path: {value}")
    return path


def read_bytes(path: Path) -> bytes:
    """Read a bounded source snapshot, normalizing only text line endings."""
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise CatalogError(f"Source exceeds {MAX_FILE_BYTES} byte limit: {path}")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n")


def read_text(path: Path) -> str:
    """Read UTF-8 text; binary and oversized sources cannot be cited as text."""
    try:
        return read_bytes(path).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"Source is not UTF-8 text: {path}") from exc


def file_hash(path: Path) -> str:
    """Return a reproducible content identity for a bounded source file."""
    return hashlib.sha256(read_bytes(path)).hexdigest()
