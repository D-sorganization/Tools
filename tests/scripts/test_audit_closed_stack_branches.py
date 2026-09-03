"""Unit tests for scripts/audit_closed_stack_branches.py pure helpers.

No gh/git calls: only symbol extraction, classification, grouping, totals and
markdown rendering are exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

import audit_closed_stack_branches as acsb  # noqa: E402

PY_SOURCE = '''
"""module docstring"""
import os

def main():
    pass

def run():
    pass

class ImpactInspector:
    def method_inside_class(self):
        pass

async def compute_face_center(x):
    return x

def _private_helper():
    pass

def ab():
    pass
'''

TS_SOURCE = """
import React from 'react';

export default function App() { return null; }
export function VariationPanel() {}
export const useRegistryStore = () => {};
export class LaunchMonitorClient {}
export interface ShotRecord {}
const notExported = 1;
function alsoNotExported() {}
"""


class TestExtractSymbols:
    def test_python_top_level_only_and_generic_filtered(self) -> None:
        symbols = acsb.extract_symbols(PY_SOURCE, "src/x/y.py")
        assert symbols == ["ImpactInspector", "compute_face_center"]
        assert "_private_helper" not in symbols
        assert "method_inside_class" not in symbols
        assert "main" not in symbols and "run" not in symbols
        assert "ab" not in symbols  # shorter than MIN_SYMBOL_LEN

    def test_typescript_exports_only(self) -> None:
        symbols = acsb.extract_symbols(TS_SOURCE, "src/x/web/src/App.tsx")
        assert symbols == [
            "VariationPanel",
            "useRegistryStore",
            "LaunchMonitorClient",
            "ShotRecord",
        ]

    def test_cap_and_dedup(self) -> None:
        src = "\n".join(f"def function_{i}(): pass" for i in range(20))
        src += "\ndef function_0(): pass"
        symbols = acsb.extract_symbols(src, "a.py")
        assert len(symbols) == acsb.MAX_SYMBOLS_PER_FILE
        assert len(set(symbols)) == len(symbols)

    def test_unknown_extension_yields_nothing(self) -> None:
        assert acsb.extract_symbols("def foo_bar(): pass", "notes.md") == []


class TestClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "drafts/idea.md",
            ".gaai/memory/x.json",
            "docs/assessments/2026-08-01.md",
            "assessments/report.md",
            "src/rate_of_closure/codex_notes.md",
            "src/x/JULES_summary.md",
            "docs/plan_for_launch.md",
            "docs/pr_details.md",
            "docs/release/PLAN.md",
            ".codex-worktrees/pr-3602-fix",
            "_codex_pr_worktrees/x/y.py",
            "docs/codex-summary.md",
        ],
    )
    def test_obsolete_paths(self, path: str) -> None:
        assert acsb.is_obsolete_path(path)
        assert acsb.classify_file(path, ["SomeSymbol"], [{"symbol": "SomeSymbol"}]) == (
            acsb.CLASS_OBSOLETE
        )

    @pytest.mark.parametrize(
        "path",
        [
            "src/rate_of_closure/simulation/planner.py",
            "src/rate_of_closure/web/src/explain.ts",
            "tests/rate_of_closure/test_airplane_model.py",
        ],
    )
    def test_product_path_not_obsolete(self, path: str) -> None:
        assert not acsb.is_obsolete_path(path)

    def test_landed_requires_majority(self) -> None:
        path = "src/rate_of_closure/analysis/screw_axis.py"
        checked = ["ScrewAxis", "compute_axis", "fit_helix", "AxisReport"]
        found_two = [{"symbol": "ScrewAxis"}, {"symbol": "fit_helix"}]
        assert acsb.classify_file(path, checked, found_two) == acsb.CLASS_LANDED
        assert acsb.classify_file(path, checked, found_two[:1]) == acsb.CLASS_MISSING

    def test_no_symbols_is_missing(self) -> None:
        assert acsb.classify_file("docs/guide.md", [], []) == acsb.CLASS_MISSING

    def test_single_symbol_found_is_landed(self) -> None:
        assert (
            acsb.classify_file("a/b.py", ["OnlyOne"], [{"symbol": "OnlyOne"}])
            == acsb.CLASS_LANDED
        )


class TestGrouping:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (
                "src/rate_of_closure/simulation/engine.py",
                "src/rate_of_closure/simulation",
            ),
            (
                "src/rate_of_closure/web/src/components/Panel.tsx",
                "src/rate_of_closure/web",
            ),
            ("src/rate_of_closure/__init__.py", "src/rate_of_closure"),
            ("tests/rate_of_closure/test_engine.py", "tests/rate_of_closure"),
            ("docs/adr/ADR-0050.md", "docs/adr"),
            ("README.md", "README.md"),
            ("docs/x.md", "docs"),
        ],
    )
    def test_group_prefix(self, path: str, expected: str) -> None:
        assert acsb.group_prefix(path) == expected

    def test_majority_class_with_tiebreak(self) -> None:
        assert acsb.majority_class([]) == acsb.CLASS_MISSING
        assert (
            acsb.majority_class([acsb.CLASS_MISSING, acsb.CLASS_OBSOLETE])
            == acsb.CLASS_MISSING
        )
        assert (
            acsb.majority_class(
                [acsb.CLASS_OBSOLETE, acsb.CLASS_OBSOLETE, acsb.CLASS_MISSING]
            )
            == acsb.CLASS_OBSOLETE
        )

    def test_group_files_sorted_and_counted(self) -> None:
        files = [
            {"path": "tests/roc/test_b.py", "classification": acsb.CLASS_MISSING},
            {"path": "tests/roc/test_a.py", "classification": acsb.CLASS_LANDED},
            {"path": "tests/roc/test_c.py", "classification": acsb.CLASS_MISSING},
            {"path": "drafts/x.md", "classification": acsb.CLASS_OBSOLETE},
        ]
        groups = acsb.group_files(files)
        assert [g["prefix"] for g in groups] == ["drafts", "tests/roc"]
        roc = groups[1]
        assert roc["classification"] == acsb.CLASS_MISSING
        assert roc["files"] == [
            "tests/roc/test_a.py",
            "tests/roc/test_b.py",
            "tests/roc/test_c.py",
        ]
        assert roc["counts"] == {
            acsb.CLASS_LANDED: 1,
            acsb.CLASS_MISSING: 2,
            acsb.CLASS_OBSOLETE: 0,
        }


def _fake_pr(number: int, **overrides: object) -> dict:
    pr: dict = {
        "number": number,
        "title": f"PR {number}",
        "state": "CLOSED",
        "url": f"https://github.com/D-sorganization/Tools/pull/{number}",
        "head_ref": f"feat/{number}",
        "head_oid": "abcdef1234567890",
        "base_ref": "main",
        "base_ref_on_origin": True,
        "merged_at": None,
        "closed_at": "2026-08-20T09:01:30Z",
        "diff_ref_used": f"origin/feat/{number}",
        "diff_ref_sha": "abcdef1234567890",
        "head_oid_matches_diff_ref": True,
        "reachable": True,
        "reason": "",
        "error": "",
        "diffstat_summary": "3 files changed, 10 insertions(+)",
        "counts": {"added": 3, "modified": 0, "deleted": 0},
        "missing_files": [],
        "groups": [],
    }
    pr.update(overrides)
    return pr


class TestRecommendationAndTotals:
    def test_keep_when_product_group_missing(self) -> None:
        pr = _fake_pr(
            1,
            groups=[
                {
                    "prefix": "src/rate_of_closure/simulation",
                    "classification": acsb.CLASS_MISSING,
                    "files": ["a"],
                    "counts": {},
                }
            ],
        )
        assert acsb.keep_recommendation(pr).startswith("keep for review")

    def test_drop_when_only_docs_missing(self) -> None:
        pr = _fake_pr(
            1,
            groups=[
                {
                    "prefix": "docs/adr",
                    "classification": acsb.CLASS_MISSING,
                    "files": ["a"],
                    "counts": {},
                }
            ],
        )
        assert acsb.keep_recommendation(pr) == (
            "drop (only non-product groups are missing)"
        )

    def test_unreachable(self) -> None:
        assert "unreachable" in acsb.keep_recommendation(_fake_pr(1, reachable=False))

    def test_totals(self) -> None:
        prs = [
            _fake_pr(
                1,
                missing_files=[
                    {"path": "a.py", "classification": acsb.CLASS_MISSING},
                    {"path": "b.py", "classification": acsb.CLASS_LANDED},
                ],
                groups=[{"classification": acsb.CLASS_MISSING}],
            ),
            _fake_pr(2, reachable=False),
        ]
        totals = acsb.compute_totals(prs)
        assert totals["prs"] == 2
        assert totals["reachable"] == 1
        assert totals["unreachable"] == [2]
        assert totals["files_by_class"][acsb.CLASS_MISSING] == 1
        assert totals["files_by_class"][acsb.CLASS_LANDED] == 1
        assert totals["groups_by_class"][acsb.CLASS_MISSING] == 1


class TestRenderMarkdown:
    def test_render_small_result(self) -> None:
        missing = [
            {
                "path": "src/rate_of_closure/simulation/engine.py",
                "classification": acsb.CLASS_MISSING,
                "symbols_checked": ["Engine"],
                "symbols_found_on_main": [],
            },
            {
                "path": "drafts/plan.md",
                "classification": acsb.CLASS_OBSOLETE,
                "symbols_checked": [],
                "symbols_found_on_main": [],
            },
        ]
        pr_ok = _fake_pr(4466, missing_files=missing, groups=acsb.group_files(missing))
        pr_bad = _fake_pr(
            4449, reachable=False, reason="origin/feat/4449 absent", diff_ref_used=None
        )
        result = {
            "schema_version": 1,
            "generated_at": "2026-09-02T00:00:00Z",
            "repo": "D-sorganization/Tools",
            "base": {"ref": "origin/main", "sha": "0" * 40},
            "prs": [pr_ok, pr_bad],
            "totals": acsb.compute_totals([pr_ok, pr_bad]),
        }
        md = acsb.render_markdown(result)
        assert md.startswith("# Closed-stack gap audit")
        assert "**Unreachable refs:** #4449" in md
        assert "## #4466 - PR 4466" in md
        assert "| `src/rate_of_closure/simulation` | missing | 1 | 0 / 1 / 0 |" in md
        assert "| `drafts` | obsolete | 1 | 0 / 0 / 1 |" in md
        assert "- `src/rate_of_closure/simulation/engine.py`" in md
        assert "`drafts/plan.md`" not in md.split("Missing files")[1].split("##")[0]
        assert "keep for review" in md
        assert "**Ref unreachable:** origin/feat/4449 absent" in md
        assert "differs from PR head SHA" not in md
        pr_ok["head_oid_matches_diff_ref"] = False
        pr_ok["diff_ref_sha"] = "fedcba9876543210"
        assert "branch tip `fedcba987654` differs from PR head SHA" in (
            acsb.render_markdown(result)
        )
        assert "## Totals" in md
        assert "Files absent from main: 2" in md

    def test_missing_list_is_capped(self) -> None:
        missing = [
            {
                "path": f"tests/roc/test_{i:03d}.py",
                "classification": acsb.CLASS_MISSING,
                "symbols_checked": [],
                "symbols_found_on_main": [],
            }
            for i in range(acsb.MD_LIST_CAP + 5)
        ]
        pr = _fake_pr(1, missing_files=missing, groups=acsb.group_files(missing))
        result = {
            "schema_version": 1,
            "generated_at": "x",
            "repo": "r",
            "base": {"ref": "origin/main", "sha": "0"},
            "prs": [pr],
            "totals": acsb.compute_totals([pr]),
        }
        md = acsb.render_markdown(result)
        assert "- ... and 5 more" in md
        assert md.count("- `tests/roc/test_") == acsb.MD_LIST_CAP
