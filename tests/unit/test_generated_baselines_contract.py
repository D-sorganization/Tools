"""Tests for generated baselines inventory and bounded query views (#5995)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.generated_baselines import (
    load_inventory,
    query_baseline,
    summarize_baseline,
    verify_baseline_integrity,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_inventory_and_all_baselines_match_hashes() -> None:
    inv = load_inventory(ROOT)
    assert inv["schema_version"] == "1.0.0"
    b_map = {b["id"]: b for b in inv["baselines"]}
    exp = {
        "benchmarks_baseline",
        "sidekick_api_baseline",
        "swing_sim_api_baseline",
        "spec_changelog",
    }
    assert exp.issubset(b_map.keys())
    for entry in b_map.values():
        assert entry["generator"] and entry["test_consumers"] and entry["review_owner"]
        assert entry["determinism"] in {
            "deterministic",
            "environment_dependent_benchmark",
        }
        assert len(entry["sha256"]) == 64
    assert verify_baseline_integrity(ROOT, check_hashes=True) == []


@pytest.mark.unit
def test_integrity_check_detects_missing_file(tmp_path: Path) -> None:
    inv = {
        "schema_version": "1.0.0",
        "baselines": [{"path": "missing.json", "sha256": "0" * 64}],
    }
    errs = verify_baseline_integrity(tmp_path, check_hashes=True, inventory=inv)
    assert len(errs) == 1 and "File not found: missing.json" in errs[0]


@pytest.mark.unit
def test_bounded_summary_and_query_views() -> None:
    for b_id in [
        "benchmarks_baseline",
        "sidekick_api_baseline",
        "swing_sim_api_baseline",
    ]:
        s = summarize_baseline(b_id, ROOT)
        assert (
            len(json.dumps(s)) < 10000
            and s["id"] == b_id
            and s["size_bytes"] > 1000000
            and "entry_count" in s
        )
    sk = query_baseline("sidekick_api_baseline", "standalone", ROOT)
    assert sk["match_count"] > 0 and all("standalone" in m for m in sk["matches"])
    bench = query_baseline("benchmarks_baseline", "euler", ROOT)
    assert bench["match_count"] >= 1 and any(
        "euler" in b["name"] for b in bench["matches"]
    )
    assert all("data" not in b.get("stats", {}) for b in bench["matches"])
