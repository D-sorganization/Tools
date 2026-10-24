"""Agent-facing docs must arm auto-merge only through automerge_guard."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths agents are instructed to read. Non-agent docs (SPEC, turnover guides) are
# out of scope — see Tools #5744 acceptance criteria.
_AGENT_DOC_GLOBS = (
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".agent/skills/issues-*/SKILL.md",
    ".agent/workflows/issues-*.md",
    ".claude/skills/issues-*/SKILL.md",
    ".gaai/core/contexts/rules/orchestration.rules.md",
    ".gaai/core/skills/delivery/coordinate-handoffs/SKILL.md",
    ".gaai/core/agents/delivery.agent.md",
)

_ALLOWLIST_SUBSTRINGS = (
    "scripts/automerge_guard.py",
    "tests/ops/test_automerge_guard.py",
    "Merge-Hold-Guard.yml",
    "gh pr merge --admin",
    "gh pr merge --disable-auto",
)


def _agent_doc_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in _AGENT_DOC_GLOBS:
        paths.extend(_REPO_ROOT.glob(pattern))
    return sorted({p.resolve() for p in paths if p.is_file()})


def _line_forbids_raw_auto_arm(line: str) -> bool:
    lowered = line.lower()
    if "gh pr merge" not in lowered or "--auto" not in lowered:
        return False
    if any(token in line for token in _ALLOWLIST_SUBSTRINGS):
        return False
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith("<!--"):
        return False
    return True


@pytest.mark.unit
def test_agent_docs_never_instruct_raw_gh_pr_merge_auto() -> None:
    offenders: list[str] = []
    for path in _agent_doc_paths():
        rel = path.relative_to(_REPO_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if _line_forbids_raw_auto_arm(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "agent docs must arm via scripts/automerge_guard.py, not raw "
        "`gh pr merge --auto`:\n" + "\n".join(offenders)
    )
