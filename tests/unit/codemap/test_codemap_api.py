"""Focused coverage for the public codemap query API."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from codemap import api, db


def _insert_file(
    repo: Path,
    path: str,
    *,
    language: str = "python",
    imports: object = (),
) -> None:
    conn = db.open_db(repo)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO files("
            "path, language, hash, mtime, size, imports, indexed_at"
            ") VALUES(?, ?, ?, ?, ?, ?, ?)",
            (path, language, f"hash-{path}", 1.0, 10, json.dumps(imports), 2.0),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_symbol(
    repo: Path,
    *,
    path: str = "pkg/mod.py",
    kind: str = "function",
    name: str = "target",
    qualified: str = "pkg.mod.target",
    sig: str = "def target() -> None",
    docstring: str = "target docs",
    calls_out: object = (),
) -> None:
    _insert_file(repo, path)
    conn = db.open_db(repo)
    try:
        conn.execute(
            "INSERT INTO symbols("
            "path, kind, name, qualified, sig, docstring, start_line, end_line, "
            "calls_out, hash"
            ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                kind,
                name,
                qualified,
                sig,
                docstring,
                3,
                7,
                json.dumps(calls_out),
                f"symbol-{qualified}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_discover_repo_root_uses_git_and_falls_back_to_parent_markers(
    tmp_path, monkeypatch
) -> None:
    nested = tmp_path / "repo" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.setattr(
        api.subprocess,
        "check_output",
        lambda *args, **kwargs: str(tmp_path / "git-root") + "\n",
    )

    assert api.discover_repo_root(nested) == tmp_path / "git-root"

    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(api.subprocess, "check_output", _raise)
    (tmp_path / "repo" / ".codemap").mkdir()

    assert api.discover_repo_root(nested) == tmp_path / "repo"


def test_resolve_caches_discovered_default_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_DEFAULT_ROOT", None)
    calls = {"count": 0}

    def _discover() -> Path:
        calls["count"] += 1
        return tmp_path

    monkeypatch.setattr(api, "discover_repo_root", _discover)

    assert api._resolve(None) == tmp_path
    assert api._resolve(None) == tmp_path
    assert calls["count"] == 1


def test_search_code_sanitizes_queries_filters_kind_and_maps_hits(tmp_path) -> None:
    _insert_symbol(
        tmp_path,
        name="build_index",
        qualified="pkg.mod.build_index",
        sig="def build_index(repo: Path) -> None",
        docstring="Builds the index.",
        calls_out=["pkg.mod.target"],
    )
    _insert_symbol(
        tmp_path,
        kind="class",
        name="BuildIndex",
        qualified="pkg.mod.BuildIndex",
        sig="class BuildIndex",
        docstring="Build index class.",
    )

    assert api.search_code("!!!", repo_root=tmp_path) == []

    hits = api.search_code("build", repo_root=tmp_path)
    assert {hit.symbol.qualified for hit in hits} == {
        "pkg.mod.build_index",
        "pkg.mod.BuildIndex",
    }
    assert all("build" in hit.snippet.lower() for hit in hits)

    function_hits = api.search_code("build_index", kind="function", repo_root=tmp_path)
    assert [hit.symbol.qualified for hit in function_hits] == ["pkg.mod.build_index"]
    assert function_hits[0].symbol.calls_out == ["pkg.mod.target"]


def test_get_symbol_supports_exact_suffix_missing_and_bad_calls_json(tmp_path) -> None:
    _insert_symbol(
        tmp_path,
        qualified="pkg.deep.Widget.run",
        name="run",
        calls_out=["pkg.deep.helper"],
    )
    conn = db.open_db(tmp_path)
    try:
        conn.execute(
            "UPDATE symbols SET calls_out = ? WHERE qualified = ?",
            ("not-json", "pkg.deep.Widget.run"),
        )
        conn.commit()
    finally:
        conn.close()

    exact = api.get_symbol("pkg.deep.Widget.run", repo_root=tmp_path)
    suffix = api.get_symbol("Widget.run", repo_root=tmp_path)

    assert exact is not None
    assert exact.calls_out == []
    assert suffix is not None
    assert suffix.qualified == "pkg.deep.Widget.run"
    assert api.get_symbol("missing", repo_root=tmp_path) is None


def test_who_calls_matches_exact_and_suffix_calls_and_skips_bad_json(tmp_path) -> None:
    _insert_symbol(
        tmp_path,
        name="caller_exact",
        qualified="pkg.mod.caller_exact",
        calls_out=["pkg.mod.target"],
    )
    _insert_symbol(
        tmp_path,
        name="caller_suffix",
        qualified="pkg.mod.caller_suffix",
        calls_out=["external.target"],
    )
    _insert_symbol(
        tmp_path,
        name="bad",
        qualified="pkg.mod.bad",
        calls_out=["target"],
    )
    conn = db.open_db(tmp_path)
    try:
        conn.execute(
            "UPDATE symbols SET calls_out = ? WHERE qualified = ?",
            ("[", "pkg.mod.bad"),
        )
        conn.commit()
    finally:
        conn.close()

    callers = api.who_calls("pkg.mod.target", repo_root=tmp_path)

    assert {symbol.qualified for symbol in callers} == {
        "pkg.mod.caller_exact",
        "pkg.mod.caller_suffix",
    }
    assert api.who_calls("!!!", repo_root=tmp_path) == []


def test_imports_of_handles_normal_missing_malformed_and_nonlist_values(
    tmp_path,
) -> None:
    _insert_file(tmp_path, "pkg/mod.py", imports=["os", "sys"])
    _insert_file(tmp_path, "pkg/nonlist.py", imports={"not": "a-list"})
    _insert_file(tmp_path, "pkg/bad.py", imports=[])
    conn = db.open_db(tmp_path)
    try:
        conn.execute("UPDATE files SET imports = ? WHERE path = ?", ("[", "pkg/bad.py"))
        conn.commit()
    finally:
        conn.close()

    assert api.imports_of("pkg\\mod.py", repo_root=tmp_path) == ["os", "sys"]
    assert api.imports_of("pkg/missing.py", repo_root=tmp_path) == []
    assert api.imports_of("pkg/bad.py", repo_root=tmp_path) == []
    assert api.imports_of("pkg/nonlist.py", repo_root=tmp_path) == []


def test_neighbors_walks_outbound_and_inbound_edges(tmp_path) -> None:
    _insert_symbol(
        tmp_path,
        name="root",
        qualified="pkg.root",
        calls_out=["pkg.child"],
    )
    _insert_symbol(tmp_path, name="child", qualified="pkg.child")
    _insert_symbol(
        tmp_path,
        name="caller",
        qualified="pkg.caller",
        calls_out=["pkg.root"],
    )

    neighbors = api.neighbors("pkg.root", repo_root=tmp_path)

    assert {symbol.qualified for symbol in neighbors} == {"pkg.child", "pkg.caller"}
    assert api.neighbors("pkg.missing", repo_root=tmp_path) == []


def test_repo_summary_reports_counts_manifest_and_malformed_manifest(tmp_path) -> None:
    _insert_file(tmp_path, "pkg/mod.py", language="python")
    _insert_file(tmp_path, "web/app.ts", language="typescript")
    _insert_symbol(tmp_path, path="pkg/mod.py", qualified="pkg.mod.target")
    db.manifest_path(tmp_path).write_text(
        json.dumps({"last_indexed": 123.5, "last_commit": "abc123"}),
        encoding="utf-8",
    )

    summary = api.repo_summary(repo_root=tmp_path)

    assert summary.repo_root == str(tmp_path.resolve())
    assert summary.files == 2
    assert summary.symbols == 1
    assert summary.languages == {"python": 1, "typescript": 1}
    assert summary.db_size_bytes > 0
    assert summary.last_indexed == 123.5
    assert summary.last_commit == "abc123"

    db.manifest_path(tmp_path).write_text("{bad-json", encoding="utf-8")
    malformed = api.repo_summary(repo_root=tmp_path)
    assert malformed.last_indexed is None
    assert malformed.last_commit is None
