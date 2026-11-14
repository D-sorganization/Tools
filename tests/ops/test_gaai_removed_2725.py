"""The retired vendored GAAI framework stays removed (Repository_Management#2725)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# Live files that must not mention GAAI. Historical docs, assessments and the
# SPEC.md change-log are deliberately out of scope.
LIVE_FILES = (
    "CLAUDE.md",
    "AGENTS.md",
    "scripts/check_root_allowlist.py",
    "scripts/check_public_repo_sanitation.py",
    "scripts/tools_module_inventory_extractors.py",
    "tests/ops/test_agent_docs_automerge_arm.py",
    "tests/ops/test_no_legacy_agent_tree.py",
)

_FLEET_BLOCK = re.compile(
    r"<!--\s*BEGIN FLEET-MANAGED.*?<!--\s*END FLEET-MANAGED[^>]*-->", re.S
)
# The fleet agent roster ("... > local > gaai") is fleet-owned text.
_ROSTER_LINE = re.compile(r"^.*\blocal\b.*\bgaai\b.*$", re.M | re.I)


def _tracked(*pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *pathspec],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


def test_no_tracked_files_under_gaai_dir() -> None:
    assert _tracked(".gaai/") == []


def test_no_tracked_gaai_launcher_or_slash_commands() -> None:
    offenders = [
        p
        for p in _tracked()
        if "gaai" in Path(p).name.lower()
        and (p.startswith(".claude/commands/") or Path(p).suffix == ".sh")
    ]
    assert offenders == []


@pytest.mark.parametrize("rel", LIVE_FILES)
def test_live_files_do_not_mention_gaai(rel: str) -> None:
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    text = _ROSTER_LINE.sub("", _FLEET_BLOCK.sub("", text))
    assert not re.search("gaai", text, re.I), f"{rel} still mentions GAAI"
