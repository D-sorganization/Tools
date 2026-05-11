"""Tests for codemap.indexer — cold and incremental rebuilds."""

from __future__ import annotations

from pathlib import Path

from codemap import db as db_mod
from codemap import indexer as indexer_mod


def _seed_repo(root: Path) -> None:
    (root / "a.py").write_text(
        '"""Module a."""\n'
        "def alpha():\n"
        "    return beta()\n"
        "\n"
        "def beta():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (root / "b.py").write_text(
        "class WGSReactor:\n    def shift(self):\n        return 2\n",
        encoding="utf-8",
    )
    (root / "c.md").write_text("# Title\n## Section\nbody\n", encoding="utf-8")


def test_cold_rebuild_indexes_all_files(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    stats = indexer_mod.rebuild(tmp_path)
    assert stats.files_parsed == 3
    assert stats.symbols_inserted >= 5  # alpha, beta, WGSReactor, shift, Title, Section
    assert stats.elapsed_s >= 0

    conn = db_mod.open_db(tmp_path)
    try:
        files = conn.execute("SELECT path FROM files ORDER BY path").fetchall()
        assert {r["path"] for r in files} == {"a.py", "b.py", "c.md"}
        # FTS works.
        rows = conn.execute(
            "SELECT s.qualified FROM symbols_fts "
            "JOIN symbols s ON s.id = symbols_fts.rowid "
            "WHERE symbols_fts MATCH 'beta*'",
        ).fetchall()
        assert any(r["qualified"] == "beta" for r in rows)
    finally:
        conn.close()


def test_incremental_rebuild_skips_unchanged_files(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    indexer_mod.rebuild(tmp_path)
    # Second rebuild with no changes — every file should be skipped via hash check.
    stats = indexer_mod.rebuild(tmp_path)
    assert stats.files_skipped_unchanged == 3
    assert stats.files_parsed == 0


def test_change_to_one_file_only_reparses_that_file(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    indexer_mod.rebuild(tmp_path)
    (tmp_path / "a.py").write_text(
        "def alpha():\n    return 99\ndef gamma():\n    return alpha()\n",
        encoding="utf-8",
    )
    stats = indexer_mod.rebuild(tmp_path)
    assert stats.files_parsed == 1
    assert stats.files_skipped_unchanged == 2

    conn = db_mod.open_db(tmp_path)
    try:
        quals = {
            r["qualified"]
            for r in conn.execute(
                "SELECT qualified FROM symbols WHERE path = 'a.py'"
            ).fetchall()
        }
        # 'beta' is gone, 'gamma' is new, 'alpha' still present.
        assert "alpha" in quals
        assert "gamma" in quals
        assert "beta" not in quals
    finally:
        conn.close()


def test_manifest_written(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    indexer_mod.rebuild(tmp_path)
    manifest = db_mod.manifest_path(tmp_path)
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "schema_version" in text
    assert "last_indexed" in text
