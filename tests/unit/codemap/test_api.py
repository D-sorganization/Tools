"""Tests for codemap.api — search, who_calls, get_symbol, neighbors."""

from __future__ import annotations

from pathlib import Path

import pytest
from codemap import api as api_mod
from codemap import indexer as indexer_mod


@pytest.fixture
def indexed_repo(tmp_path: Path) -> Path:
    (tmp_path / "reactor.py").write_text(
        '"""WGS reactor logic."""\n'
        "def shift_reaction(temperature):\n"
        "    return temperature * 2\n"
        "\n"
        "def run_wgs_reactor():\n"
        "    return shift_reaction(450)\n",
        encoding="utf-8",
    )
    (tmp_path / "ui.py").write_text(
        "from reactor import run_wgs_reactor\n"
        "\n"
        "def on_button_click():\n"
        "    run_wgs_reactor()\n",
        encoding="utf-8",
    )
    indexer_mod.rebuild(tmp_path)
    return tmp_path


def test_search_code_returns_relevant_hits(indexed_repo: Path) -> None:
    hits = api_mod.search_code("wgs reactor", repo_root=indexed_repo)
    assert hits, "search_code returned no hits"
    quals = [h.symbol.qualified for h in hits]
    assert any("run_wgs_reactor" in q for q in quals)


def test_search_code_kind_filter(indexed_repo: Path) -> None:
    hits = api_mod.search_code("shift", kind="function", repo_root=indexed_repo)
    assert hits
    assert all(h.symbol.kind == "function" for h in hits)


def test_get_symbol_exact_and_suffix(indexed_repo: Path) -> None:
    sym = api_mod.get_symbol("run_wgs_reactor", repo_root=indexed_repo)
    assert sym is not None
    assert sym.qualified == "run_wgs_reactor"
    # Suffix match.
    sym2 = api_mod.get_symbol("shift_reaction", repo_root=indexed_repo)
    assert sym2 is not None


def test_get_symbol_missing_returns_none(indexed_repo: Path) -> None:
    assert api_mod.get_symbol("does_not_exist", repo_root=indexed_repo) is None


def test_who_calls_finds_at_least_one_caller(indexed_repo: Path) -> None:
    callers = api_mod.who_calls("run_wgs_reactor", repo_root=indexed_repo)
    assert callers, "expected at least one caller of run_wgs_reactor"
    assert any(c.qualified == "on_button_click" for c in callers)


def test_imports_of(indexed_repo: Path) -> None:
    imports = api_mod.imports_of("ui.py", repo_root=indexed_repo)
    assert "reactor" in imports


def test_repo_summary(indexed_repo: Path) -> None:
    stats = api_mod.repo_summary(repo_root=indexed_repo)
    assert stats.files == 2
    assert stats.symbols >= 3
    assert stats.languages.get("python", 0) == 2


def test_neighbors_returns_callers_and_callees(indexed_repo: Path) -> None:
    nbrs = api_mod.neighbors("run_wgs_reactor", hops=1, repo_root=indexed_repo)
    quals = {n.qualified for n in nbrs}
    # on_button_click calls run_wgs_reactor (inbound); run_wgs_reactor calls
    # shift_reaction (outbound). Both should be 1-hop neighbours.
    assert "on_button_click" in quals
