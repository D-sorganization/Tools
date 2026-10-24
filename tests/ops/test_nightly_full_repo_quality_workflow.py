"""Contracts for the weekly full-repo quality lane (#5703)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly-full-repo-quality.yml"


def test_nightly_full_repo_quality_runs_on_github_hosted_runner() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"]["full-repo-quality"]["runs-on"] == "ubuntu-latest"
