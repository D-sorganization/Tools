"""Content-based trust checks for the disposable lexical CodeMap index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import db


class StaleIndexError(RuntimeError):
    """The index cannot answer authoritatively for this source checkout."""


def implementation_id() -> str:
    """Fingerprint index implementation and optional parser distribution versions."""
    folder = Path(__file__).parent
    payload = {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for p in sorted(folder.glob("*.py"))
    }
    for name in (
        "tree-sitter",
        "tree-sitter-python",
        "tree-sitter-javascript",
        "tree-sitter-typescript",
        "tree-sitter-rust",
        "tree-sitter-markdown",
        "defusedxml",
        "pathspec",
        "blake3",
    ):
        try:
            payload[name] = version(name)
        except PackageNotFoundError:
            payload[name] = "unavailable"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def current_files(repo: Path) -> dict[str, str]:
    """Hash the supported lexical corpus, refusing unreadable or escaping files."""
    from .indexer import _hash_bytes, _walk

    result = {}
    for path, relative in _walk(repo):
        if not path.resolve().is_relative_to(repo.resolve()):
            raise OSError(f"Source path escapes checkout: {relative}")
        result[relative] = _hash_bytes(path.read_bytes())
    return result


def inspect(repo: Path, conn: sqlite3.Connection) -> dict[str, Any]:
    """Expose errors and coverage without treating a timestamp as freshness."""
    errors: list[str] = []
    try:
        manifest = json.loads(db.manifest_path(repo).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid manifest")
        if manifest.get("repo_root") != str(repo.resolve()):
            errors.append("different checkout")
        if manifest.get("schema_version") != db.SCHEMA_VERSION:
            errors.append("different schema")
        if manifest.get("implementation") != implementation_id():
            errors.append("different indexer or parser dependencies")
        if manifest.get("errors") != []:
            errors.append("partial or unknown indexing coverage")
        actual = current_files(repo)
        indexed = {
            r["path"]: r["hash"] for r in conn.execute("SELECT path, hash FROM files")
        }
        if actual != indexed:
            errors.append("source content or membership changed")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return {
        "verified": not errors,
        "errors": errors,
        "scope": (
            "supported lexical source files; calls are heuristic, "
            "not a complete semantic graph"
        ),
    }


def require_current(repo: Path, conn: sqlite3.Connection) -> None:
    """Fail with recovery instructions instead of returning plausible stale symbols."""
    state = inspect(repo, conn)
    if not state["verified"]:
        raise StaleIndexError(
            "CodeMap is unverified: "
            + "; ".join(state["errors"])
            + ". Run codemap --repo <checkout> rebuild, or inspect source directly."
        )


def reconcile(
    repo: Path, conn: sqlite3.Connection, stats: Any
) -> list[tuple[Path, str]]:
    """Remove deleted rows and include dirty/untracked files in incremental runs."""
    reset_obsolete_parser(repo, conn, stats)
    actual = current_files(repo)
    indexed = {
        r["path"]: r["hash"] for r in conn.execute("SELECT path, hash FROM files")
    }
    for missing in sorted(indexed.keys() - actual.keys()):
        stats.symbols_deleted += conn.execute(
            "DELETE FROM symbols WHERE path = ?", (missing,)
        ).rowcount
        conn.execute("DELETE FROM files WHERE path = ?", (missing,))
    return [(repo / p, p) for p in sorted(actual) if actual[p] != indexed.get(p)]


def reset_obsolete_parser(repo: Path, conn: sqlite3.Connection, stats: Any) -> None:
    """Reparse unchanged source after a parser/indexer change or unknown provenance."""
    try:
        manifest = json.loads(db.manifest_path(repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    if (
        not isinstance(manifest, dict)
        or manifest.get("implementation") != implementation_id()
    ):
        stats.symbols_deleted += conn.execute("DELETE FROM symbols").rowcount
        conn.execute("DELETE FROM files")
