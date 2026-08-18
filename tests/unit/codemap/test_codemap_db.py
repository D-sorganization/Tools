from __future__ import annotations

import sqlite3
from pathlib import Path

from codemap import db as codemap_db
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def test_path_helpers_use_canonical_codemap_locations(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"

    assert codemap_db.db_path(repo_root) == repo_root / ".codemap" / "index.db"
    assert codemap_db.manifest_path(repo_root) == (
        repo_root / ".codemap" / "manifest.json"
    )


def test_open_db_initializes_schema_pragmas_and_local_gitignore(
    tmp_path: Path,
) -> None:
    conn = codemap_db.open_db(tmp_path)

    try:
        codemap_dir = tmp_path / ".codemap"
        assert codemap_dir.is_dir()
        assert (codemap_dir / "index.db").is_file()
        assert (codemap_dir / ".gitignore").read_text(encoding="utf-8") == "*\n"
        assert conn.row_factory is sqlite3.Row
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert codemap_db.get_schema_version(conn) == codemap_db.SCHEMA_VERSION

        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            )
        }
        assert {
            "meta",
            "files",
            "symbols",
            "symbols_fts",
            "symbols_ai",
            "symbols_ad",
            "symbols_au",
        } <= tables
    finally:
        conn.close()


def test_init_schema_keeps_fts_index_in_sync_for_symbol_writes() -> None:
    conn = sqlite3.connect(":memory:")

    try:
        codemap_db.init_schema(conn)
        conn.execute(
            """
            INSERT INTO files(path, language, hash, mtime, size, imports, indexed_at)
            VALUES('src/tool.py', 'python', 'file-hash', 1.0, 100, '[]', 2.0)
            """,
        )
        cursor = conn.execute(
            """
            INSERT INTO symbols(
                path, kind, name, qualified, sig, docstring, start_line,
                end_line, calls_out, hash
            )
            VALUES(
                'src/tool.py', 'function', 'parse_tokens',
                'tool.parse_tokens', '(source: str) -> list[str]',
                'Tokenize source text.', 10, 20, '["tool.lex"]', 'symbol-hash'
            )
            """,
        )
        symbol_id = cursor.lastrowid

        assert _fts_names(conn, "tokenize") == ["parse_tokens"]

        conn.execute(
            "UPDATE symbols SET name = ?, qualified = ? WHERE id = ?",
            ("parse_stream", "tool.parse_stream", symbol_id),
        )
        assert _fts_names(conn, "parse_stream") == ["parse_stream"]
        assert _fts_names(conn, "parse_tokens") == []

        conn.execute("DELETE FROM symbols WHERE id = ?", (symbol_id,))
        assert _fts_names(conn, "parse_stream") == []
    finally:
        conn.close()


def test_open_db_is_idempotent_and_preserves_existing_local_gitignore(
    tmp_path: Path,
) -> None:
    codemap_dir = tmp_path / ".codemap"
    codemap_dir.mkdir()
    gitignore = codemap_dir / ".gitignore"
    gitignore.write_text("custom-ignore\n", encoding="utf-8")

    conn = codemap_db.open_db(tmp_path)

    try:
        codemap_db.init_schema(conn)
        assert gitignore.read_text(encoding="utf-8") == "custom-ignore\n"
        assert codemap_db.get_schema_version(conn) == codemap_db.SCHEMA_VERSION
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM meta WHERE key = 'schema_version'",
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_init_schema_migrates_legacy_fts_alias_schema() -> None:
    conn = sqlite3.connect(":memory:")

    try:
        conn.executescript("""
            CREATE TABLE meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO meta(key, value) VALUES('schema_version', '1');

            CREATE TABLE files (
                path        TEXT PRIMARY KEY,
                language    TEXT NOT NULL,
                hash        TEXT NOT NULL,
                mtime       REAL NOT NULL,
                size        INTEGER NOT NULL,
                imports     TEXT NOT NULL DEFAULT '[]',
                indexed_at  REAL NOT NULL
            );

            CREATE TABLE symbols (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                path        TEXT NOT NULL,
                kind        TEXT NOT NULL,
                name        TEXT NOT NULL,
                qualified   TEXT NOT NULL,
                sig         TEXT NOT NULL DEFAULT '',
                docstring   TEXT NOT NULL DEFAULT '',
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                calls_out   TEXT NOT NULL DEFAULT '[]',
                hash        TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE symbols_fts USING fts5(
                name, qualified, sig, docstring, co,
                content='symbols', content_rowid='id',
                tokenize='unicode61'
            );
            CREATE TRIGGER symbols_ai AFTER INSERT ON symbols BEGIN
                SELECT 1;
            END;
            """)

        codemap_db.init_schema(conn)

        fts_columns = {row[1] for row in conn.execute("PRAGMA table_info(symbols_fts)")}
        assert "calls_out" in fts_columns
        assert "co" not in fts_columns
        assert codemap_db.get_schema_version(conn) == codemap_db.SCHEMA_VERSION
    finally:
        conn.close()


def test_get_schema_version_returns_zero_for_missing_or_malformed_values() -> None:
    conn = sqlite3.connect(":memory:")

    try:
        codemap_db.init_schema(conn)

        conn.execute("DELETE FROM meta WHERE key = 'schema_version'")
        assert codemap_db.get_schema_version(conn) == 0

        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', 'not-an-int')"
        )
        assert codemap_db.get_schema_version(conn) == 0
    finally:
        conn.close()


def _fts_names(conn: sqlite3.Connection, query: str) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM symbols_fts WHERE symbols_fts MATCH ? ORDER BY rowid",
            (query,),
        )
    ]
