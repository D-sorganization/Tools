"""Core PR gates run on GitHub-hosted runners for public Tools (#5782)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

HOSTED_CORE_JOBS = {
    "anti-phantom-merge.yml": ("guard", "ubuntu-latest"),
    "detect-secrets.yml": ("detect-secrets", "ubuntu-latest"),
    "Merge-Hold-Guard.yml": ("enforce-holds", "ubuntu-latest"),
}


def _job_runs_on(path: Path, job_id: str) -> str:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    job = workflow["jobs"][job_id]
    return str(job["runs-on"])


def test_static_core_workflows_use_github_hosted() -> None:
    for filename, (job_id, expected) in HOSTED_CORE_JOBS.items():
        assert _job_runs_on(WORKFLOWS / filename, job_id) == expected


def test_pick_runner_consumer_workflows_honor_dispatcher_output() -> None:
    for filename, job_id in (
        ("docs-governance.yml", "docs-governance"),
        ("spec-check.yml", "spec-freshness"),
        ("workflow-lint.yml", "lint-workflows"),
    ):
        workflow = yaml.safe_load((WORKFLOWS / filename).read_text(encoding="utf-8"))
        runs_on = workflow["jobs"][job_id]["runs-on"]
        assert runs_on == "${{ needs.pick-runner.outputs.runner }}"
