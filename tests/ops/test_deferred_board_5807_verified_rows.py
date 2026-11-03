"""Verified rows from deferred tracker #5807 (board 2026-10-26)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROWS_PATH = _REPO_ROOT / "config" / "deferred_board_5807_verified_rows.json"


@pytest.mark.unit
def test_deferred_board_verified_rows_file_loads() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert rows, "expected at least one verified row"


@pytest.mark.unit
def test_verified_rows_reference_existing_tests() -> None:
    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    for row in rows:
        rel_path = row["test_module"]
        test_path = _REPO_ROOT / rel_path
        assert test_path.is_file(), rel_path
        source = test_path.read_text(encoding="utf-8")
        for name in row["test_names"]:
            assert f"def {name}" in source
