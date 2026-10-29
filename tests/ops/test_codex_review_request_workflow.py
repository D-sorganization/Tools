"""Contracts for the Codex review request caller workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "codex-review-request.yml"


def test_codex_review_request_uses_hosted_runner_with_gh() -> None:
    """Self-hosted fleet runners may not expose gh; hosted CI does."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    request = workflow["jobs"]["request"]
    uses = request["uses"]
    assert "reusable-codex-review-request.yml@" in uses
    assert request["with"]["runner"] == "ubuntu-latest"
