"""The legacy .agent/ tree stays removed and GAAI defers to the fleet PR lifecycle.

agy and Codex read .agents/; Tools' .agent/ copies were deleted (#5893,
Repository_Management#2567).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_tracked_files_under_legacy_agent_dir() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".agent/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked == []


def test_root_allowlist_drops_legacy_agent_entry() -> None:
    from scripts.check_root_allowlist import ROOT_ALLOWLIST

    assert ".agent" not in ROOT_ALLOWLIST


def test_gaai_project_rules_fence_ci_watch_and_fix() -> None:
    # The core skill keeps a session alive for CI; the fleet PR lifecycle
    # ends it at PR open, so the project layer fences it (RM #2567).
    rules = REPO_ROOT / ".gaai" / "project" / "contexts" / "rules"
    text = (rules / "project.rules.md").read_text(encoding="utf-8")
    assert "invoke the `ci-watch-and-fix` skill" not in text
    assert "Do not run core skill `delivery/ci-watch-and-fix`" in text
    assert "fleet-pr-lifecycle" in text
