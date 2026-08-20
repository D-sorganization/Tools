from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from codemap import cli, db
from tests.helpers.codemap_optional_deps import CODEMAP_DEPS_SKIP

# Scoped to this module only; a session-wide skip hook silenced the whole
# suite here once already (issue #4497).
pytestmark = CODEMAP_DEPS_SKIP


def _insert_symbol(repo: Path) -> None:
    conn = db.open_db(repo)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO files("
            "path, language, hash, mtime, size, imports, indexed_at"
            ") VALUES(?, ?, ?, ?, ?, ?, ?)",
            ("pkg/mod.py", "python", "file-hash", 1.0, 10, "[]", 2.0),
        )
        conn.execute(
            "INSERT INTO symbols("
            "path, kind, name, qualified, sig, docstring, start_line, end_line, "
            "calls_out, hash"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "pkg/mod.py",
                "function",
                "target",
                "pkg.mod.target",
                "def target() -> None",
                "target docs",
                3,
                7,
                "[]",
                "symbol-hash",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_main_rebuild_reports_stats_and_first_error(monkeypatch, capsys) -> None:
    stats = SimpleNamespace(
        files_parsed=2,
        files_skipped_unchanged=1,
        symbols_inserted=4,
        elapsed_s=0.25,
        errors=["bad.py: parse failed"],
    )
    calls: list[tuple[str | None, str | None]] = []

    def rebuild(repo: str | None, *, since: str | None = None):
        calls.append((repo, since))
        return stats

    monkeypatch.setattr(cli.indexer_mod, "rebuild", rebuild)

    assert cli.main(["--repo", "repo-root", "rebuild", "--since", "HEAD~1"]) == 0

    captured = capsys.readouterr()
    assert calls == [("repo-root", "HEAD~1")]
    assert "indexed 2 files (1 unchanged), 4 symbols in 0.25s" in captured.out
    assert "1 errors (first: bad.py: parse failed)" in captured.err


def test_main_search_formats_hits_and_joins_query_terms(monkeypatch, capsys) -> None:
    symbol = SimpleNamespace(
        kind="function",
        qualified="pkg.mod.target",
        path="pkg/mod.py",
        start_line=3,
        end_line=7,
        sig="def target() -> None",
    )
    calls: list[tuple[str, int, str | None, str | None]] = []

    def search_code(query: str, *, k: int, kind: str | None, repo_root: str | None):
        calls.append((query, k, kind, repo_root))
        return [SimpleNamespace(score=12.5, symbol=symbol)]

    monkeypatch.setattr(cli.api_mod, "search_code", search_code)

    assert (
        cli.main(
            [
                "--repo",
                "repo",
                "search",
                "target",
                "symbol",
                "-k",
                "3",
                "--kind",
                "function",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert calls == [("target symbol", 3, "function", "repo")]
    assert "[  12.50] function pkg.mod.target" in captured.out
    assert "pkg/mod.py:3-7  def target() -> None" in captured.out


def test_search_and_who_calls_print_empty_messages(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.api_mod, "search_code", lambda *args, **kwargs: [])
    monkeypatch.setattr(cli.api_mod, "who_calls", lambda *args, **kwargs: [])

    assert cli.main(["search", "missing"]) == 0
    assert cli.main(["who-calls", "pkg.mod.missing"]) == 0

    captured = capsys.readouterr()
    assert "(no matches)" in captured.out
    assert "(no callers found)" in captured.out


def test_who_calls_formats_callers(monkeypatch, capsys) -> None:
    caller = SimpleNamespace(
        kind="method",
        qualified="pkg.mod.Caller.run",
        path="pkg/mod.py",
        start_line=11,
    )
    monkeypatch.setattr(cli.api_mod, "who_calls", lambda *args, **kwargs: [caller])

    assert cli.main(["--repo", "repo", "who-calls", "pkg.mod.target"]) == 0

    captured = capsys.readouterr()
    assert "method   pkg.mod.Caller.run  pkg/mod.py:11" in captured.out


def test_export_writes_plain_jsonl_and_default_gzip(
    tmp_path, monkeypatch, capsys
) -> None:
    _insert_symbol(tmp_path)
    plain_out = tmp_path / "symbols.jsonl"

    assert cli.main(["--repo", str(tmp_path), "export", "--jsonl", str(plain_out)]) == 0

    plain_records = [
        json.loads(line) for line in plain_out.read_text(encoding="utf-8").splitlines()
    ]
    assert plain_records[0]["qualified"] == "pkg.mod.target"
    assert "exported 1 symbols" in capsys.readouterr().out

    monkeypatch.setattr(cli.api_mod, "discover_repo_root", lambda: tmp_path)

    assert cli.main(["export"]) == 0

    gzip_out = tmp_path / ".codemap" / "exports" / "code_map.jsonl.gz"
    with gzip.open(gzip_out, "rt", encoding="utf-8") as fh:
        gzip_records = [json.loads(line) for line in fh]
    assert gzip_records[0]["name"] == "target"


def test_info_prints_sorted_language_summary(monkeypatch, capsys) -> None:
    summary = SimpleNamespace(
        repo_root=Path("repo"),
        files=5,
        symbols=8,
        db_size_bytes=2048,
        last_commit=None,
        languages={"markdown": 1, "python": 4},
    )
    monkeypatch.setattr(cli.api_mod, "repo_summary", lambda repo_root=None: summary)

    assert cli.main(["info"]) == 0

    captured = capsys.readouterr()
    assert "repo:       repo" in captured.out
    assert "db size:    2.0 KiB" in captured.out
    assert "last cmt:   (unknown)" in captured.out
    assert captured.out.index("python") < captured.out.index("markdown")
