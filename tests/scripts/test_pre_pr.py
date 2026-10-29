"""scripts/pre_pr.py orchestrates Tools' local quality-gate mirror (RM-6)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from scripts import pre_pr


def test_main_passes_when_all_gates_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pre_pr.check_conflict_markers, "main", lambda argv=None: 0)
    monkeypatch.setattr(pre_pr.pre_pr_delta_gates, "main", lambda argv=None: 0)
    assert pre_pr.main([]) == 0


def test_main_fails_when_conflict_markers_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pre_pr.check_conflict_markers, "main", lambda argv=None: 1)
    monkeypatch.setattr(pre_pr.pre_pr_delta_gates, "main", lambda argv=None: 0)
    assert pre_pr.main([]) == 1


def test_main_fails_when_delta_gates_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pre_pr.check_conflict_markers, "main", lambda argv=None: 0)
    monkeypatch.setattr(pre_pr.pre_pr_delta_gates, "main", lambda argv=None: 1)
    assert pre_pr.main([]) == 1


def test_main_forwards_repository_root_and_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_markers: dict[str, Sequence[str] | None] = {}
    captured_delta: dict[str, Sequence[str] | None] = {}

    def fake_markers(argv: Sequence[str] | None = None) -> int:
        captured_markers["argv"] = argv
        return 0

    def fake_delta(argv: Sequence[str] | None = None) -> int:
        captured_delta["argv"] = argv
        return 0

    monkeypatch.setattr(pre_pr.check_conflict_markers, "main", fake_markers)
    monkeypatch.setattr(pre_pr.pre_pr_delta_gates, "main", fake_delta)

    pre_pr.main(["--repository-root", "/tmp/x", "--base", "origin/develop"])

    assert captured_markers["argv"] == ["--repository-root", "/tmp/x"]
    assert captured_delta["argv"] == [
        "--repository-root",
        "/tmp/x",
        "--base",
        "origin/develop",
    ]
