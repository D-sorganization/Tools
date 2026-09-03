"""Contract tests for the closed-stack gap-audit decision ledger (Tools #4921).

The ledger records, per file the audit flagged as missing from ``main`` for
PRs #4212/#4233/#4246, whether it is re-landed, obsolete (superseded by a file
already on ``main``) or needs an owner decision. These tests keep the ledger
honest: every superseding path it cites must exist, every audited file must
carry a decision, and the totals must agree with the per-file rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

ROOT = Path(__file__).parents[2]
DECISIONS_PATH = ROOT / "docs" / "release" / "closed_stack_gap_audit_decisions.v1.json"
AUDIT_PATH = ROOT / "docs" / "release" / "closed_stack_gap_audit.v1.json"
DECISION_CLASSES = {"re-land", "obsolete", "needs-owner"}
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PR_REF_PATTERN = re.compile(r"^#\d{3,5}$")


def _load(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _decisions() -> dict[str, object]:
    return _load(DECISIONS_PATH)


def _pr_rows() -> list[dict[str, object]]:
    return cast(list[dict[str, object]], _decisions()["prs"])


def test_ledger_identity_and_base() -> None:
    ledger = _decisions()
    assert ledger["schema_version"] == "closed-stack-gap-audit-decisions/v1"
    assert ledger["repo"] == "D-sorganization/Tools"
    base = cast(dict[str, str], ledger["base"])
    assert base["ref"] == "origin/main"
    assert SHA_PATTERN.match(base["sha"])
    source = cast(dict[str, object], ledger["source_audit"])
    assert source["path"] == "docs/release/closed_stack_gap_audit.v1.json"
    classes = cast(dict[str, object], ledger["decision_classes"])
    assert classes.keys() == DECISION_CLASSES


def test_every_audited_pr_and_file_has_a_decision() -> None:
    audit_prs = {
        cast(int, pr["number"]): pr
        for pr in cast(list[dict[str, object]], _load(AUDIT_PATH)["prs"])
    }
    rows = _pr_rows()
    assert [row["number"] for row in rows] == [4212, 4233, 4246]
    for row in rows:
        number = cast(int, row["number"])
        audit_pr = audit_prs[number]
        assert row["head_oid"] == audit_pr["head_oid"]
        assert SHA_PATTERN.match(cast(str, row["head_oid"]))
        files = cast(list[dict[str, object]], row["files"])
        decided = {cast(str, f["path"]) for f in files}
        audited = {
            cast(str, f["path"]) if isinstance(f, dict) else cast(str, f)
            for f in cast(list[object], audit_pr["missing_files"])
        }
        undecided = sorted(audited - decided)
        assert (
            not undecided
        ), f"#{number}: audited files without a decision: {undecided}"


def test_file_rows_are_well_formed_and_cite_real_superseders() -> None:
    for row in _pr_rows():
        for entry in cast(list[dict[str, object]], row["files"]):
            path = cast(str, entry["path"])
            decision = cast(str, entry["decision"])
            assert decision in DECISION_CLASSES, (path, decision)
            assert cast(str, entry["rationale"]).strip(), path
            assert isinstance(entry["in_audit_missing_list"], bool)
            assert isinstance(entry["present_at_head_oid"], bool)
            superseded_by = cast(list[str], entry["superseded_by"])
            landed_via = cast(list[str], entry["landed_via"])
            for superseder in superseded_by:
                assert (ROOT / superseder).is_file(), (path, superseder)
            for ref in landed_via:
                assert PR_REF_PATTERN.match(ref), (path, ref)
            present = bool(entry["present_at_head_oid"])
            if decision == "obsolete" and present:
                assert superseded_by and landed_via, f"{path}: cite superseder"
            if decision == "obsolete" and not present:
                assert entry.get("audit_correction") == "misattributed_to_this_pr", path
            if decision == "re-land":
                assert not (ROOT / path).is_file(), f"{path}: already on main"


def test_counts_and_totals_are_consistent() -> None:
    ledger = _decisions()
    totals = cast(dict[str, int], ledger["totals"])
    expected = {cls: 0 for cls in DECISION_CLASSES}
    for row in _pr_rows():
        counts = cast(dict[str, int], row["counts"])
        files = cast(list[dict[str, object]], row["files"])
        for cls in DECISION_CLASSES:
            observed = sum(1 for f in files if f["decision"] == cls)
            assert counts[cls] == observed, (row["number"], cls)
            expected[cls] += observed
        misattributed = sum(1 for f in files if f.get("audit_correction"))
        assert counts["misattributed"] == misattributed, row["number"]
        assert row["true_missing_count_at_head_oid"] == sum(
            1 for f in files if f["present_at_head_oid"]
        )
    for cls, count in expected.items():
        assert totals[cls] == count, cls
    assert totals["files_decided"] == sum(expected.values())
    assert totals["files_to_re_land"] == expected["re-land"]
