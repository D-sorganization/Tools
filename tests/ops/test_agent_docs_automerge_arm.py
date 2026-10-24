"""Agent-facing docs: arm auto-merge without squash (Dieter / #5744)."""

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

_SQUASH_ARM = re.compile(
    r"--strategy\s+squash|--squash\b|arm\s+squash\s+auto-?merge",
    re.IGNORECASE,
)
_GH_PR_MERGE_AUTO = re.compile(r"gh\s+pr\s+merge\b.*--auto", re.IGNORECASE)


def _iter_agent_doc_files() -> list[Path]:
    files: list[Path] = [p for p in _AGENT_DOC_ROOTS if p.is_file()]
    for directory in _AGENT_DOC_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.md")))
    return files


def _agent_doc_id(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _iter_doc_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append((lineno, line))
    return lines


@pytest.mark.parametrize("path", _iter_agent_doc_files(), ids=_agent_doc_id)
def test_agent_docs_do_not_instruct_squash_auto_merge_arm(path: Path) -> None:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    offenders: list[str] = []
    for lineno, line in _iter_doc_lines(path):
        if _SQUASH_ARM.search(line) and not re.search(
            r"never\s+[`']?--squash|not\s+[`']?--squash", line, re.I
        ):
            offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "arm with `gh pr merge <pr> --repo <owner>/<repo> --auto` or "
        "`python scripts/automerge_guard.py <owner>/<repo> <pr> --arm` "
        "(never --squash or --strategy squash):\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", _iter_agent_doc_files(), ids=_agent_doc_id)
def test_agent_docs_gh_pr_merge_auto_uses_repo_not_squash(path: Path) -> None:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    offenders: list[str] = []
    for lineno, line in _iter_doc_lines(path):
        if not _GH_PR_MERGE_AUTO.search(line):
            continue
        if re.search(r"never\s+raw|bypasses\s+hold", line, re.I):
            continue
        lower = line.lower()
        if "--squash" in lower and not re.search(r"never\s+.*--squash", lower):
            offenders.append(f"{rel}:{lineno}: includes --squash with --auto")
        if "--repo" not in lower:
            offenders.append(f"{rel}:{lineno}: missing --repo on gh pr merge --auto")
    assert not offenders, "\n".join(offenders)
