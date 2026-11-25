"""Executable acceptance checklist for epic #6019 (Fleet Agent API / MCP)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.acceptance, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST_PATH = ROOT / "config" / "epic_6019_acceptance_checklist.json"
ALLOWED_STATUSES = frozenset({"proven", "open"})
EPIC_ITEM_IDS = frozenset(
    {
        "6019-E1",
        "6019-E2",
        "6019-E3",
        "6019-E4",
        "6019-E5",
        "6019-E6",
        "6019-E7",
        "6019-E8",
    }
)


def _load_checklist() -> dict[str, Any]:
    payload = json.loads(CHECKLIST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_epic_6019_acceptance_checklist_is_valid_and_fail_closed() -> None:
    """Checklist lists every epic item; proven rows need on-disk evidence."""
    checklist = _load_checklist()
    assert checklist["schema_id"] == "tools/epic-6019-acceptance-checklist"
    assert checklist["schema_version"] == 1
    assert (
        checklist["epic_url"] == "https://github.com/D-sorganization/Tools/issues/6019"
    )
    assert checklist["overall_status"] == "open"

    items = checklist["items"]
    assert isinstance(items, list)
    by_id = {row["item_id"]: row for row in items}
    assert set(by_id) == EPIC_ITEM_IDS
    assert len(items) == len(by_id)

    counts = Counter(row["status"] for row in items)
    assert dict(counts) == checklist["status_counts"]

    for row in items:
        status = row["status"]
        assert status in ALLOWED_STATUSES
        assert row["requirement"].strip()
        evidence = row["evidence_files"]
        assert isinstance(evidence, list) and evidence
        gaps = row["gaps"]
        assert isinstance(gaps, list)
        for relative in evidence:
            assert "*" not in relative
            path = ROOT / relative
            if status == "proven":
                missing = f"missing evidence for {row['item_id']}: {relative}"
                assert path.is_file(), missing
            elif not path.is_file():
                assert gaps, row["item_id"]
        if status == "proven":
            assert gaps == [], row["item_id"]
        else:
            assert gaps, row["item_id"]

    e7 = by_id["6019-E7"]
    assert e7["status"] == "open"
    assert "TOOLS-CALC" in e7["requirement"]
