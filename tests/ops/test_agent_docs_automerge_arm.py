"""Agent-facing docs must arm auto-merge only through automerge_guard."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths agents and fleet skills actually read (issue #5744).
_AGENT_DOC_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "AGENTS.md",
    _REPO_ROOT / "CLAUDE.md",
    _REPO_ROOT / "CONTRIBUTING.md",
    _REPO_ROOT / ".gaai" / "core" / "contexts" / "rules" / "orchestration.rules.md",
    _REPO_ROOT / "docs" / "ci-cd" / "AUTO_ISSUE_RESOLVER.md",
)

_AGENT_DOC_DIRS: tuple[Path, ...] = (
    _REPO_ROOT / ".agent" / "skills",
    _REPO_ROOT / ".agent" / "workflows",
    _REPO_ROOT / ".claude" / "skills",
)

_RAW_AUTO_MERGE = re.compile(r"gh\s+pr\s+merge\b.*--auto", re.IGNORECASE)


def _iter_agent_doc_files() -> list[Path]:
    files: list[Path] = [p for p in _AGENT_DOC_ROOTS if p.is_file()]
    for directory in _AGENT_DOC_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.md")))
    return files


def _agent_doc_id(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


@pytest.mark.parametrize("path", _iter_agent_doc_files(), ids=_agent_doc_id)
def test_agent_docs_do_not_instruct_raw_gh_pr_merge_auto(path: Path) -> None:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    offenders: list[str] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        if _RAW_AUTO_MERGE.search(line):
            if re.search(r"never\s+raw|do not use raw|bypasses\s+hold", line, re.I):
                continue
            offenders.append(f"{rel}:{lineno}: {stripped}")
    assert not offenders, (
        "use `python scripts/automerge_guard.py <owner>/<repo> <pr> --arm "
        "--strategy squash` instead of raw gh pr merge --auto:\n"
        + "\n".join(offenders)
    )
