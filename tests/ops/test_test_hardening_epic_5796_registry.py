"""Registry contract for epic #5796 test-hardening closure evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "config" / "test_hardening_epic_5796_registry.json"


def _load_registry() -> dict[str, object]:
    payload = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("registry root must be a mapping")
    return payload


@pytest.mark.unit
def test_epic_5796_registry_file_exists_and_parses() -> None:
    assert _REGISTRY_PATH.is_file()
    registry = _load_registry()
    assert registry["epic_issue"] == 5796
    assert registry["deferred_tracker_issue"] == 5807


@pytest.mark.unit
def test_landed_tests_exist_with_named_functions() -> None:
    registry = _load_registry()
    landed = registry["landed_tests"]
    assert isinstance(landed, list)
    for entry in landed:
        assert isinstance(entry, dict)
        rel_path = entry["path"]
        assert isinstance(rel_path, str)
        test_path = _REPO_ROOT / rel_path
        assert test_path.is_file(), f"missing regression file: {rel_path}"
        source = test_path.read_text(encoding="utf-8")
        names = entry["names"]
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str)
            assert f"def {name}" in source, f"{name} not found in {rel_path}"


@pytest.mark.unit
def test_landing_issue_numbers_are_complete_and_unique() -> None:
    registry = _load_registry()
    landed_children = registry["landed_child_issues"]
    assert isinstance(landed_children, dict)
    expected = {str(n) for n in range(5797, 5807)}
    assert set(landed_children.keys()) == expected


@pytest.mark.unit
def test_deferred_findings_do_not_duplicate_landed_ids() -> None:
    registry = _load_registry()
    landed_tests = registry["landed_tests"]
    deferred = registry["deferred_to_5807_finding_ids"]
    assert isinstance(landed_tests, list)
    assert isinstance(deferred, list)
    landed_ids: set[str] = set()
    for entry in landed_tests:
        assert isinstance(entry, dict)
        finding_ids = entry["finding_ids"]
        assert isinstance(finding_ids, list)
        landed_ids.update(str(item) for item in finding_ids)
    overlap = landed_ids & {str(item) for item in deferred}
    assert not overlap, f"landed/deferred overlap: {sorted(overlap)}"
