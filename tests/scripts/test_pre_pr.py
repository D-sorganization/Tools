"""scripts/pre_pr.py orchestrates Tools' local quality-gate mirror (RM-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import pre_pr


def test_main_passes_when_all_gates_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pre_pr, "run_conflict_marker_check", lambda _root: 0)
    monkeypatch.setattr(pre_pr, "run_delta_gates", lambda _root, _base: 0)
    assert pre_pr.main([]) == 0


def test_main_fails_when_conflict_markers_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pre_pr, "run_conflict_marker_check", lambda _root: 1)
    monkeypatch.setattr(pre_pr, "run_delta_gates", lambda _root, _base: 0)
    assert pre_pr.main([]) == 1


def test_main_fails_when_delta_gates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pre_pr, "run_conflict_marker_check", lambda _root: 0)
    monkeypatch.setattr(pre_pr, "run_delta_gates", lambda _root, _base: 1)
    assert pre_pr.main([]) == 1


def test_main_forwards_repository_root_and_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_markers: dict[str, Path | None] = {}
    captured_delta: dict[str, object] = {}

    def fake_markers(root: Path) -> int:
        captured_markers["root"] = root
        return 0

    def fake_delta(root: Path, base_ref: str) -> int:
        captured_delta["root"] = root
        captured_delta["base"] = base_ref
        return 0

    monkeypatch.setattr(pre_pr, "run_conflict_marker_check", fake_markers)
    monkeypatch.setattr(pre_pr, "run_delta_gates", fake_delta)

    pre_pr.main(["--repository-root", "/tmp/x", "--base", "origin/develop"])

    assert captured_markers["root"] == Path("/tmp/x")
    assert captured_delta["root"] == Path("/tmp/x")
    assert captured_delta["base"] == "origin/develop"


def test_scan_conflict_markers_returns_empty_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pre_pr, "scan_conflict_markers", lambda _root: [])
    assert pre_pr.run_conflict_marker_check(tmp_path) == 0


def test_scan_conflict_markers_reports_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pre_pr,
        "scan_conflict_markers",
        lambda _root: [f"src/example.py:10:{pre_pr._CONFLICT_MARKER}"],
    )
    assert pre_pr.run_conflict_marker_check(tmp_path) == 1
