"""Agent docs must route auto-merge arming through automerge_guard (#5744)."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CLAUDE.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / ".agent",
    REPO_ROOT / ".claude",
    REPO_ROOT / ".gaai",
)

ALLOWLIST_SUFFIXES = (
    "scripts/automerge_guard.py",
    "tests/ops/test_automerge_guard.py",
    "tests/ops/test_agent_automerge_docs.py",
    "wave_solver.py",
    "SPEC.md",
    "docs/putting_launch_monitor/TURNOVER.md",
    "docs/ci-cd/AUTO_ISSUE_RESOLVER.md",
)

RAW_ARM = re.compile(r"gh\s+pr\s+merge\b[^`\n]*--auto\b", re.IGNORECASE)


def _paths_to_scan() -> list[Path]:
    paths: list[Path] = []
    for root in SCAN_ROOTS:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(sorted(root.rglob("*.md")))
            paths.extend(sorted(root.rglob("*.mdc")))
            paths.extend(sorted(root.rglob("SKILL.md")))
    return paths


def test_agent_facing_docs_do_not_instruct_raw_gh_pr_merge_auto() -> None:
    violations: list[str] = []
    for path in _paths_to_scan():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if any(rel.endswith(suffix) or suffix in rel for suffix in ALLOWLIST_SUFFIXES):
            continue
        text = path.read_text(encoding="utf-8")
        for match in RAW_ARM.finditer(text):
            if "automerge_guard" in match.group(0):
                continue
            violations.append(f"{rel}: {match.group(0).strip()}")
    assert violations == []
