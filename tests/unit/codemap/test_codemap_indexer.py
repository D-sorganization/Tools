from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from codemap._ts_common import ParsedSymbol, ParseResult

from codemap import db as codemap_db
from codemap import indexer

# test_codemap_indexer uses mock parsers and does not require the tree-sitter stack.


def _python_result(path: str | Path, _source: bytes | str) -> ParseResult:
    stem = Path(path).stem
    return ParseResult(
        "python",
        imports=["os"],
        symbols=[
            ParsedSymbol(
                kind="function",
                name=f"{stem}_entry",
                qualified=f"{stem}.{stem}_entry",
                sig="() -> None",
                docstring=f"Entry point for {stem}.",
                start_line=1,
                end_line=1,
                calls_out=["print"],
            )
        ],
    )


def _symbols(repo_root: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(codemap_db.db_path(repo_root))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT path, kind, name, qualified, calls_out FROM symbols ORDER BY path"
        ).fetchall()
    finally:
        conn.close()


def _files(repo_root: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(codemap_db.db_path(repo_root))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT path, language, imports FROM files ORDER BY path"
        ).fetchall()
    finally:
        conn.close()


def test_rebuild_indexes_supported_files_respects_gitignore_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (repo / "tool.py").write_text("def run(): pass\n", encoding="utf-8")
    (repo / "ignored.py").write_text("def hidden(): pass\n", encoding="utf-8")
    (repo / "notes.txt").write_text("not source\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "vendored.py").write_text(
        "def vendored(): pass\n", encoding="utf-8"
    )
    monkeypatch.setattr(indexer.parsers_mod, "dispatch", _python_result)
    monkeypatch.setattr(indexer, "_current_commit", lambda _repo: "abc123")

    stats = indexer.rebuild(repo)

    assert stats.files_seen == 1
    assert stats.files_parsed == 1
    assert stats.symbols_inserted == 1
    assert stats.errors == []
    assert [
        (row["path"], row["language"], json.loads(row["imports"]))
        for row in _files(repo)
    ] == [("tool.py", "python", ["os"])]
    assert [
        (row["path"], row["name"], json.loads(row["calls_out"]))
        for row in _symbols(repo)
    ] == [("tool.py", "tool_entry", ["print"])]
    manifest = json.loads(codemap_db.manifest_path(repo).read_text(encoding="utf-8"))
    assert manifest["repo_root"] == str(repo.resolve())
    assert manifest["schema_version"] == codemap_db.SCHEMA_VERSION
    assert manifest["last_commit"] == "abc123"
    assert manifest["files"] == 1
    assert manifest["symbols"] == 1


def test_rebuild_skips_unchanged_files_after_hash_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "tool.py"
    source.write_text("def run(): pass\n", encoding="utf-8")
    dispatch_calls: list[str] = []

    def dispatch(path: str | Path, source_bytes: bytes | str) -> ParseResult:
        dispatch_calls.append(str(path))
        return _python_result(path, source_bytes)

    monkeypatch.setattr(indexer.parsers_mod, "dispatch", dispatch)

    first = indexer.rebuild(repo)
    second = indexer.rebuild(repo)

    assert first.files_parsed == 1
    assert second.files_seen == 1
    assert second.files_parsed == 0
    assert second.files_skipped_unchanged == 1
    assert dispatch_calls == ["tool.py"]


def test_incremental_rebuild_removes_deleted_files_and_ignores_unsupported_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    keep = repo / "keep.py"
    gone = repo / "gone.py"
    keep.write_text("def keep(): pass\n", encoding="utf-8")
    gone.write_text("def gone(): pass\n", encoding="utf-8")
    monkeypatch.setattr(indexer.parsers_mod, "dispatch", _python_result)
    indexer.rebuild(repo)
    gone.unlink()
    (repo / "README.txt").write_text("changed unsupported file\n", encoding="utf-8")
    monkeypatch.setattr(
        indexer,
        "_git_changed_files",
        lambda _repo, _since: ["gone.py", "README.txt"],
    )

    stats = indexer.rebuild(repo, since="HEAD~1")

    assert stats.files_seen == 0
    assert stats.files_parsed == 0
    assert stats.symbols_deleted == 1
    assert [row["path"] for row in _files(repo)] == ["keep.py"]
    assert [row["path"] for row in _symbols(repo)] == ["keep.py"]


def test_incremental_rebuild_reprocesses_supported_changed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "keep.py"
    source.write_text("def keep(): pass\n", encoding="utf-8")
    monkeypatch.setattr(indexer.parsers_mod, "dispatch", _python_result)
    indexer.rebuild(repo)
    source.write_text("def keep_changed(): pass\n", encoding="utf-8")
    monkeypatch.setattr(
        indexer, "_git_changed_files", lambda _repo, _since: ["keep.py"]
    )

    stats = indexer.rebuild(repo, since="HEAD~1")

    assert stats.files_seen == 1
    assert stats.files_parsed == 1
    assert stats.symbols_deleted == 1
    assert [row["path"] for row in _symbols(repo)] == ["keep.py"]


def test_incremental_rebuild_falls_back_to_full_walk_when_git_diff_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tool.py").write_text("def run(): pass\n", encoding="utf-8")
    monkeypatch.setattr(indexer.parsers_mod, "dispatch", _python_result)
    monkeypatch.setattr(indexer, "_git_changed_files", lambda _repo, _since: [])

    stats = indexer.rebuild(repo, since="missing-base")

    assert stats.files_seen == 1
    assert stats.files_parsed == 1
    assert [row["path"] for row in _files(repo)] == ["tool.py"]


def test_process_file_records_prior_symbol_deletions_and_stat_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "tool.py"
    source.write_text("def original(): pass\n", encoding="utf-8")
    monkeypatch.setattr(indexer.parsers_mod, "dispatch", _python_result)
    indexer.rebuild(repo)
    source.write_text("def changed(): pass\n", encoding="utf-8")
    monkeypatch.setattr(indexer, "_read_bytes", lambda _path: b"def changed(): pass\n")
    conn = codemap_db.open_db(repo)
    stats = indexer.RebuildStats()

    class StatlessPath:
        def stat(self) -> object:
            raise OSError("metadata unavailable")

    try:
        indexer._process_file(StatlessPath(), "tool.py", repo, conn, stats)  # type: ignore[arg-type]
        conn.commit()
    finally:
        conn.close()

    assert stats.files_seen == 1
    assert stats.files_parsed == 1
    assert stats.symbols_deleted == 1
    assert len(_symbols(repo)) == 1


def test_process_file_handles_unreadable_files_and_parser_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "tool.py"
    source.write_text("def run(): pass\n", encoding="utf-8")
    conn = codemap_db.open_db(repo)
    stats = indexer.RebuildStats()
    try:
        monkeypatch.setattr(indexer, "_read_bytes", lambda _path: None)
        indexer._process_file(source, "tool.py", repo, conn, stats)
        assert stats.files_seen == 0

        monkeypatch.setattr(indexer, "_read_bytes", lambda _path: b"def run(): pass\n")
        monkeypatch.setattr(indexer.parsers_mod, "dispatch", lambda _rel, _data: None)
        indexer._process_file(source, "tool.py", repo, conn, stats)
        assert stats.files_seen == 1
        assert stats.files_parsed == 0
        assert _files(repo) == []
    finally:
        conn.close()


def test_read_bytes_returns_none_for_os_errors() -> None:
    class UnreadablePath:
        def read_bytes(self) -> bytes:
            raise OSError("permission denied")

    assert indexer._read_bytes(UnreadablePath()) is None  # type: ignore[arg-type]


def test_rebuild_collects_per_file_errors_and_still_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "bad.py").write_text("def bad(): pass\n", encoding="utf-8")

    def fail_process(
        _abs_path: Path,
        rel: str,
        _repo_root: Path,
        _conn: sqlite3.Connection,
        _stats: indexer.RebuildStats,
    ) -> None:
        raise RuntimeError(f"cannot parse {rel}")

    monkeypatch.setattr(indexer, "_process_file", fail_process)

    stats = indexer.rebuild(repo)

    assert stats.errors == ["bad.py: cannot parse bad.py"]
    manifest = json.loads(codemap_db.manifest_path(repo).read_text(encoding="utf-8"))
    assert manifest["files"] == 0
    assert manifest["symbols"] == 0


def test_git_helpers_return_safe_defaults_when_git_commands_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing_git(*_args: object, **_kwargs: object) -> str:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", raise_missing_git)

    assert indexer._git_changed_files(tmp_path, "HEAD~1") == []
    assert indexer._current_commit(tmp_path) is None


def test_git_helpers_parse_successful_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check_output(args: list[str], **_kwargs: object) -> str:
        if len(args) >= 3 and args[1:3] == ["diff", "--name-only"]:
            return "\n src/tool.py \n\nREADME.md\n"
        if len(args) >= 3 and args[1:3] == ["rev-parse", "HEAD"]:
            return " abc123 \n"
        raise AssertionError(args)

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    assert indexer._git_changed_files(tmp_path, "HEAD~1") == [
        "src/tool.py",
        "README.md",
    ]
    assert indexer._current_commit(tmp_path) == "abc123"


def test_hash_bytes_uses_blake3_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHasher:
        def hexdigest(self) -> str:
            return "fake-blake3"

    fake_blake3 = SimpleNamespace(blake3=lambda data: FakeHasher())
    monkeypatch.setitem(sys.modules, "blake3", fake_blake3)

    assert indexer._hash_bytes(b"payload") == "fake-blake3"


def test_hash_bytes_fallback_when_blake3_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "blake3", None)
    import hashlib

    expected = hashlib.blake2b(b"payload", digest_size=16).hexdigest()
    assert indexer._hash_bytes(b"payload") == expected


def test_hash_bytes_raises_when_blake3_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_blake3(data: bytes) -> None:
        raise RuntimeError("Corrupt memory")

    fake_blake3 = SimpleNamespace(blake3=bad_blake3)
    monkeypatch.setitem(sys.modules, "blake3", fake_blake3)
    with pytest.raises(RuntimeError, match="Corrupt memory"):
        indexer._hash_bytes(b"payload")


def test_gitignore_loader_uses_simple_fallback_when_pathspec_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("# comment\nignored/\n*.tmp\n", encoding="utf-8")
    original_import = __import__

    def fake_import(
        name: str,
        globals_: object = None,
        locals_: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "pathspec":
            raise ImportError("pathspec missing")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    ignored = indexer._load_gitignore(repo)

    assert ignored("src/ignored/module.py") is True
    assert ignored(".codemap/index.db") is True
    assert ignored("src/tool.py") is False
