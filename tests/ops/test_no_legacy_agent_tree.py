"""The legacy .agent/ tree stays removed.

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
