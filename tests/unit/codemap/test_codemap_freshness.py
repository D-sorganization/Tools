"""Public queries must never present dirty or partial indexes as current."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from codemap._ts_common import ParsedSymbol, ParseResult

from codemap import api, db, indexer


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "module.py"
    source.write_text("def target(): pass\n", encoding="utf-8")

    def parse(path: str, data: bytes) -> ParseResult:
        return ParseResult(
            "python",
            [],
            [
                ParsedSymbol(
                    "function", "target", "module.target", "def target()", "", 1, 1
                )
            ],
        )

    monkeypatch.setattr(indexer.parsers_mod, "dispatch", parse)
    indexer.rebuild(tmp_path)
    return tmp_path


def test_dirty_and_new_files_require_rebuild(indexed: Path) -> None:
    assert api.get_symbol("target", repo_root=indexed) is not None
    (indexed / "module.py").write_text("def changed(): pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="rebuild"):
        api.get_symbol("target", repo_root=indexed)
    indexer.rebuild(indexed)
    (indexed / "added.py").write_text("# new\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="rebuild"):
        api.search_code("target", repo_root=indexed)


def test_full_rebuild_removes_deleted_rows(indexed: Path) -> None:
    (indexed / "module.py").unlink()
    indexer.rebuild(indexed)
    assert api.get_symbol("target", repo_root=indexed) is None


def test_copied_or_partial_manifest_cannot_authorize_query(indexed: Path) -> None:
    path = db.manifest_path(indexed)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["errors"] = ["unreadable source"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rebuild"):
        api.imports_of("module.py", repo_root=indexed)
    data["errors"] = []
    data["repo_root"] = str(indexed / "other")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="rebuild"):
        api.who_calls("target", repo_root=indexed)


def test_default_root_is_not_reused_after_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / ".git").mkdir(parents=True)
    monkeypatch.setattr(api, "_DEFAULT_ROOT", None)
    monkeypatch.setattr(api, "discover_repo_root", Path.cwd)
    monkeypatch.chdir(first)
    assert api._resolve(None) == first
    monkeypatch.chdir(second)
    assert api._resolve(None) == second


def test_missing_parser_is_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codemap import _lang_python

    monkeypatch.setattr(_lang_python, "get_parser", lambda _: None)
    source = tmp_path / "module.py"
    source.write_text("def target(): pass\n", encoding="utf-8")
    result = indexer.rebuild(tmp_path)
    assert result.errors
    with pytest.raises(RuntimeError, match="rebuild"):
        api.get_symbol("target", repo_root=tmp_path)


def test_changed_parser_identity_forces_reparse(indexed: Path) -> None:
    path = db.manifest_path(indexed)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["implementation"] = "obsolete-parser-version"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = indexer.rebuild(indexed)
    assert result.files_parsed == 1
    assert result.files_skipped_unchanged == 0


def test_stale_export_is_rejected(indexed: Path) -> None:
    from codemap import cli

    (indexed / "module.py").write_text("# changed\n", encoding="utf-8")
    assert cli.main(["--repo", str(indexed), "export"]) == 2


def test_failed_file_transaction_is_reparsed_on_retry(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = indexer._process_file
    (indexed / "module.py").write_text("def changed(): pass\n", encoding="utf-8")

    def interrupted(*args: object) -> None:
        original(*args)
        raise RuntimeError("interrupted after writing file hash")

    monkeypatch.setattr(indexer, "_process_file", interrupted)
    assert indexer.rebuild(indexed).errors
    monkeypatch.setattr(indexer, "_process_file", original)
    assert indexer.rebuild(indexed).files_parsed == 1
