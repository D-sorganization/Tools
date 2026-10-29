"""scripts/check_conflict_markers.py shared scan helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_conflict_markers


def test_scan_conflict_markers_returns_empty_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        check_conflict_markers,
        "scan_conflict_markers",
        lambda _root: [],
    )
    assert check_conflict_markers.main(["--repository-root", str(tmp_path)]) == 0


def test_scan_conflict_markers_reports_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        check_conflict_markers,
        "scan_conflict_markers",
        lambda _root: [f"src/example.py:10:{check_conflict_markers._CONFLICT_MARKER}"],
    )
    assert check_conflict_markers.main(["--repository-root", str(tmp_path)]) == 1
