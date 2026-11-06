"""Verified rows from deferred tracker #5807 (board 2026-10-26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROWS_PATH = _REPO_ROOT / "config" / "deferred_board_5807_verified_rows.json"

EXPECTED_5807_FINDING_IDS: list[str] = [
    "TOOLS-01",
    "TOOLS-04",
    "TOOLS-07",
    "TOOLS-09",
    "TOOLS-12",
    "TOOLS-13",
    "TOOLS-14",
    "TOOLS-17",
    "TOOLS-19",
    "TOOLS-21",
    "TOOLS-22",
    "TOOLS-23",
    "TOOLS-24",
    "TOOLS-25",
    "TOOLS-26",
    "TOOLS-27",
    "TOOLS-28",
    "TOOLS-29",
]


@pytest.mark.unit
def test_deferred_board_verified_rows_file_loads() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows, "expected at least one verified row"


@pytest.mark.unit
def test_deferred_board_verified_rows_contains_all_18_rows() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    finding_ids = [row["finding_id"] for row in rows]
    assert len(finding_ids) == 18, f"expected 18 rows, found {len(finding_ids)}"
    assert sorted(finding_ids) == sorted(EXPECTED_5807_FINDING_IDS)
    for row in rows:
        assert row.get("board_epic") == 5796
        assert row.get("tracker_issue") == 5807


@pytest.mark.unit
def test_verified_rows_reference_existing_tests() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    for row in rows:
        rel_path = row.get("test_module")
        if rel_path is not None:
            test_path = _REPO_ROOT / rel_path
            msg_missing = f"missing test file: {rel_path} for {row['finding_id']}"
            assert test_path.is_file(), msg_missing
            source = test_path.read_text(encoding="utf-8")
            for name in row.get("test_names", []):
                msg_fn = f"{name} not in {rel_path} for {row['finding_id']}"
                assert f"def {name}" in source, msg_fn
        else:
            # Only refuted/proof rows may omit test_module
            assert row["finding_id"] == "TOOLS-19"
            notes = row.get("notes", "").lower()
            assert "refuted" in notes or "proof" in notes


@pytest.mark.unit
def test_regression_rows_have_merge_evidence() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    # 15 rows resolved via fix PRs
    pr_rows = [r for r in rows if r.get("merge_pr") is not None]
    assert len(pr_rows) >= 15
    for r in pr_rows:
        assert isinstance(r["merge_pr"], int) and r["merge_pr"] > 0
        assert isinstance(r.get("merge_commit"), str) and len(r["merge_commit"]) >= 7
        assert r.get("test_module") is not None
        assert len(r.get("test_names", [])) >= 1
