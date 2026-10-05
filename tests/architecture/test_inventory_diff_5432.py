"""Bounded stale-inventory diff helper (Tools #5432)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.architecture._inventory_diff import (
    DEFAULT_LIMIT,
    summarize_inventory_difference,
)


def _entry(path: str, digest: str = "a") -> dict[str, Any]:
    return {"path": path, "content_sha256_lf": digest * 64}


def _payload(entries: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"schema_version": "x/1", "entries": entries, **extra}


def test_identical_payloads_have_no_summary() -> None:
    a = _payload([_entry("src/alpha/a.py"), _entry("src/beta/b.py")])
    b = _payload([_entry("src/alpha/a.py"), _entry("src/beta/b.py")])
    assert summarize_inventory_difference(a, b) is None


def test_changed_shard_is_named_with_its_file() -> None:
    a = _payload([_entry("src/alpha/a.py"), _entry("src/beta/b.py")])
    b = _payload([_entry("src/alpha/a.py"), _entry("src/beta/b.py", "b")])
    summary = summarize_inventory_difference(a, b)
    assert summary is not None
    assert "changed" in summary
    assert "src/beta" in summary
    assert "entries-src-beta.json" in summary
    assert "src/alpha" not in summary


def test_size_split_package_names_the_production_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A package over the shard budget is split one level deeper by
    # partition_entries; the summary must name that deeper shard file.
    from scripts import tools_module_inventory_storage as storage

    monkeypatch.setattr(storage, "SHARD_BUDGET_BYTES", 200)
    base = [
        _entry("src/big/application/__init__.py"),
        _entry("src/big/application/app.py"),
        _entry("src/big/web/view.py"),
        _entry("src/big/web/route.py"),
    ]
    changed = [*base[:1], _entry("src/big/application/app.py", "b"), *base[2:]]
    summary = summarize_inventory_difference(_payload(base), _payload(changed))
    assert summary is not None
    assert "entries-src-big-application.json" in summary
    assert "entries-src-big-web.json" not in summary
    assert "entries-src-big.json" not in summary


def test_added_and_removed_shards_and_top_level_keys_are_reported() -> None:
    a = _payload([_entry("src/alpha/a.py"), _entry("src/beta/b.py")], extra_key=1)
    b = _payload([_entry("src/alpha/a.py"), _entry("src/gamma/c.py")], other=2)
    summary = summarize_inventory_difference(a, b)
    assert summary is not None
    assert "removed" in summary and "src/beta" in summary
    assert "added" in summary and "src/gamma" in summary
    assert "extra_key" in summary and "other" in summary


def test_top_level_value_change_is_reported() -> None:
    a = _payload([_entry("src/alpha/a.py")], release_status="x")
    b = _payload([_entry("src/alpha/a.py")], release_status="y")
    summary = summarize_inventory_difference(a, b)
    assert summary is not None and "release_status" in summary


def test_output_is_capped() -> None:
    a = _payload([_entry(f"src/pkg{i}/a.py") for i in range(200)])
    b = _payload([_entry(f"src/pkg{i}/a.py", "b") for i in range(200)])
    summary = summarize_inventory_difference(a, b)
    assert summary is not None
    assert summary.count("entries-src-pkg") <= DEFAULT_LIMIT
    assert "more" in summary
    assert len(summary.splitlines()) <= DEFAULT_LIMIT + 5


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        summarize_inventory_difference(_payload([]), _payload([]), limit=0)


def test_large_stale_payload_summary_is_fast() -> None:
    """The summary of a 5000-entry payload with one change stays near-instant."""
    entries = [_entry(f"src/pkg{i % 50}/m{i}.py") for i in range(5000)]
    stale = [dict(e) for e in entries]
    stale[2500] = _entry(stale[2500]["path"], "f")
    a, b = _payload(entries), _payload(stale)
    start = time.perf_counter()
    summary = summarize_inventory_difference(a, b)
    elapsed = time.perf_counter() - start
    assert summary is not None and "src/pkg0" in summary
    assert elapsed < 2.0
