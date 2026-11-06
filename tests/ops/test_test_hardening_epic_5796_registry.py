"""Registry contract for epic #5796 test-hardening closure evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "config" / "test_hardening_epic_5796_registry.json"
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


@pytest.mark.unit
def test_deferred_5807_finding_ids_are_complete() -> None:
    registry = _load_registry()
    deferred = registry["deferred_to_5807_finding_ids"]
    assert isinstance(deferred, list)
    assert sorted(str(x) for x in deferred) == sorted(EXPECTED_5807_FINDING_IDS)


@pytest.mark.unit
def test_deferred_5807_landed_tests_exist_with_named_functions() -> None:
    registry = _load_registry()
    deferred_landed = registry.get("deferred_5807_landed_tests")
    assert isinstance(deferred_landed, list), "expected deferred_5807_landed_tests list"
    assert len(deferred_landed) >= 15, (
        f"expected at least 15 routed test entries, got {len(deferred_landed)}"
    )

    covered_findings: set[str] = set()
    for entry in deferred_landed:
        assert isinstance(entry, dict)
        finding_ids = entry["finding_ids"]
        assert isinstance(finding_ids, list) and finding_ids
        for fid in finding_ids:
            assert fid in EXPECTED_5807_FINDING_IDS, f"unexpected finding_id: {fid}"
            covered_findings.add(fid)

        rel_path = entry["path"]
        assert isinstance(rel_path, str)
        test_path = _REPO_ROOT / rel_path
        assert test_path.is_file(), f"missing regression file: {rel_path}"
        source = test_path.read_text(encoding="utf-8")
        names = entry["names"]
        assert isinstance(names, list) and names
        for name in names:
            assert isinstance(name, str)
            assert f"def {name}" in source, f"{name} not found in {rel_path}"

    # All findings except TOOLS-19 (refuted) must have landed regression tests
    expected_with_tests = set(EXPECTED_5807_FINDING_IDS) - {"TOOLS-19"}
    missing = expected_with_tests - covered_findings
    msg = f"missing regression test routing: {sorted(missing)}"
    assert not missing, msg


@pytest.mark.unit
def test_deferred_5807_registry_cross_checks_verified_rows() -> None:
    registry = _load_registry()
    deferred_landed = registry.get("deferred_5807_landed_tests")
    assert isinstance(deferred_landed, list)

    rows = json.loads(_ROWS_PATH.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    rows_by_id = {r["finding_id"]: r for r in rows}

    for entry in deferred_landed:
        for fid in entry["finding_ids"]:
            row = rows_by_id.get(fid)
            assert row is not None, f"finding {fid} not in verified rows file"
            if row.get("test_module") is not None:
                assert row["test_module"] == entry["path"]
                assert set(entry["names"]).issubset(set(row["test_names"]))
