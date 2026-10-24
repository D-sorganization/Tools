"""Contracts for conflict-disarm scheduled load (#5699)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "conflict-disarm.yml"


def test_conflict_disarm_has_no_scheduled_sweep_on_self_hosted_pool() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = workflow.get("on", workflow.get(True, {}))
    assert isinstance(on, dict)
    assert "schedule" not in on
    assert "push" in on
    assert "workflow_dispatch" in on
