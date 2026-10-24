"""Contracts for Bot CI Trigger load reduction (#5705)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "Bot-CI-Trigger.yml"


def test_bot_ci_trigger_has_no_scheduled_self_hosted_sweep() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = workflow.get("on", workflow.get(True, {}))
    assert isinstance(on, dict)
    assert "schedule" not in on
    assert "pull_request" in on
    assert "workflow_dispatch" in on
