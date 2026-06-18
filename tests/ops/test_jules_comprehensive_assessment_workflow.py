from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_WORKFLOW = Path(".github/workflows/Jules-Comprehensive-Assessment.yml")


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_comprehensive_assessment_timeout_covers_jules_poll_budget() -> None:
    workflow = _workflow_text()
    timeout_match = re.search(
        r"comprehensive-assessment:\n\s+timeout-minutes: (?P<minutes>\d+)",
        workflow,
    )
    max_wait_match = re.search(r"\n\s+MAX_WAIT=(?P<seconds>\d+)", workflow)

    assert timeout_match is not None
    assert max_wait_match is not None
    timeout_seconds = int(timeout_match.group("minutes")) * 60
    max_wait_seconds = int(max_wait_match.group("seconds"))
    assert timeout_seconds >= max_wait_seconds + 1800


def test_pr_title_fallback_uses_portable_gh_pr_list_flags() -> None:
    workflow = _workflow_text()
    enforce_step = workflow[workflow.index("- name: Enforce PR Title") :]

    assert "--sort" not in enforce_step
    assert "--json number,createdAt" in enforce_step
    assert "sort_by(.createdAt)" in enforce_step
