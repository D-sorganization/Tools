"""Freshness, bounded retrieval, provenance and boundary-review acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_context.catalog import CatalogError
from agent_context.service import ContextService


def test_context_returns_current_evidence_and_consumers(repository: Path) -> None:
    service = ContextService(repository)
    result = service.context("provider")
    assert result["component"]["id"] == "provider"
    assert result["consumers"] == ["consumer"]
    assert result["sources"][0]["path"] == "src/provider.py"
    assert result["sources"][0]["start_line"] == 1
    assert result["authority"] == "source-verified; contract-review-required"
    assert not result["contract_reviews"][0]["verified"]


def test_search_is_bounded_and_unknown_query_does_not_invent(repository: Path) -> None:
    service = ContextService(repository)
    assert (
        service.search("physical measurements", limit=1)["matches"][0]["id"]
        == "provider"
    )
    assert service.search("nonexistent neutron reactor")["matches"] == []
    with pytest.raises(CatalogError):
        service.search("", limit=0)
    with pytest.raises(CatalogError):
        service.context("provider", max_chars=100)


def test_context_rereads_dirty_source_and_never_executes_it(repository: Path) -> None:
    service = ContextService(repository)
    first = service.context("provider")
    path = repository / "src/provider.py"
    path.write_text(
        "raise RuntimeError('must never execute')\n"
        "def convert(value):\n    return value * 3\n",
        encoding="utf-8",
    )
    second = service.context("provider")
    assert first["provenance"]["digest"] != second["provenance"]["digest"]
    assert "value * 3" in second["sources"][0]["excerpt"]


def test_review_is_bound_to_exact_inputs_and_contract(repository: Path) -> None:
    service = ContextService(repository)
    with pytest.raises(CatalogError, match="review"):
        service.check_reviews()
    service.review(
        "conversion-flow", "Reviewed conversion interface and its integration test"
    )
    assert service.check_reviews() == []
    assert (
        service.context("provider")["authority"]
        == "source-verified; contracts-reviewed"
    )
    path = repository / "src/provider.py"
    path.write_text("def convert(value):\n    return value * 4\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="conversion-flow"):
        service.check_reviews()
    record = repository / "docs/agent_context/reviews.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    data["reviews"]["conversion-flow"]["reviewed_at"] = "2099-01-01"
    record.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CatalogError, match="conversion-flow"):
        service.check_reviews()


def test_review_requires_specific_rationale(repository: Path) -> None:
    with pytest.raises(CatalogError, match="rationale"):
        ContextService(repository).review("conversion-flow", "ok")


def test_generated_output_is_deterministic_and_stale_output_fails(
    repository: Path,
) -> None:
    service = ContextService(repository)
    service.review(
        "conversion-flow", "Reviewed fixture conversion contract and consumer behavior"
    )
    service.render()
    first = (repository / "docs/agent_context/README.md").read_bytes()
    assert service.check() == []
    service.render()
    assert (repository / "docs/agent_context/README.md").read_bytes() == first
    (repository / "src/consumer.py").write_text(
        "# changed consumer\n", encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="stale"):
        service.check()


def test_html_escapes_metadata_and_works_without_network(repository: Path) -> None:
    path = repository / "docs/agent_context/catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["components"][0]["summary"] = "<script>alert(1)</script>"
    path.write_text(json.dumps(data), encoding="utf-8")
    ContextService(repository).render()
    html = (repository / "docs/agent_context/index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert '<label for="search"' in html
    assert "https://cdn" not in html
    assert "consumer" in html


def test_no_runtime_cache_is_mistaken_for_another_checkout(
    repository: Path, tmp_path: Path
) -> None:
    service = ContextService(repository)
    service.render()
    cache = repository / ".codemap/context.json"
    assert cache.exists()
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["root"] = str(tmp_path / "different-worktree")
    cache.write_text(json.dumps(data), encoding="utf-8")
    result = service.context("provider")
    assert result["provenance"]["root"] == str(repository.resolve())
    assert result["cache_state"] == "replaced"
