"""Tests for codemap.db schema init."""

from __future__ import annotations

from pathlib import Path

from codemap import db as db_mod


def test_open_db_creates_schema_and_gitignore(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    conn = db_mod.open_db(repo)
    try:
        # Tables exist.
        names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table','virtual table')",
            ).fetchall()
        }
        assert "files" in names
        assert "symbols" in names
        assert "meta" in names
        # FTS table exists (it appears as 'table' rows due to shadow tables).
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'symbols_fts'",
        ).fetchall()
        assert rows, "symbols_fts virtual table missing"
        # Schema version recorded.
        assert db_mod.get_schema_version(conn) == db_mod.SCHEMA_VERSION
    finally:
        conn.close()

    # .codemap/.gitignore created.
    assert (repo / ".codemap" / ".gitignore").read_text(encoding="utf-8").strip() == "*"


def test_open_db_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for _ in range(3):
        conn = db_mod.open_db(repo)
        try:
            assert db_mod.get_schema_version(conn) == db_mod.SCHEMA_VERSION
        finally:
            conn.close()


def test_db_path_helpers(tmp_path: Path) -> None:
    assert db_mod.db_path(tmp_path) == tmp_path / ".codemap" / "index.db"
    assert db_mod.manifest_path(tmp_path) == tmp_path / ".codemap" / "manifest.json"
