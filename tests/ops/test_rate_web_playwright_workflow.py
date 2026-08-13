"""Security and reproducibility contracts for the Rate Playwright workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "rate-web-playwright.yml"
RUNNER_GUARD_PATH = REPO_ROOT / ".github" / "workflows" / "local-only-runner-guard.yml"
FULL_ACTION_SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _workflow() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _run_steps(job: dict[str, Any]) -> dict[str, str]:
    return {
        str(step["name"]): str(step["run"])
        for step in job["steps"]
        if isinstance(step, dict) and "name" in step and "run" in step
    }


def test_fork_pull_requests_cannot_reach_the_persistent_fleet() -> None:
    jobs = _workflow()["jobs"]
    trusted = jobs["trusted-production-worker-e2e"]
    fork = jobs["fork-production-worker-e2e"]

    assert trusted["runs-on"] == "d-sorg-fleet"
    assert "head.repo.full_name == github.repository" in trusted["if"]
    assert fork["runs-on"] == "ubuntu-latest"
    assert "head.repo.full_name != github.repository" in fork["if"]
    assert "github.event_name == 'pull_request'" in fork["if"]


def test_fork_and_trusted_jobs_run_the_same_locked_production_gate() -> None:
    jobs = _workflow()["jobs"]
    trusted = jobs["trusted-production-worker-e2e"]
    fork = jobs["fork-production-worker-e2e"]

    assert trusted["steps"] == fork["steps"]
    assert _run_steps(trusted) == _run_steps(fork)
    commands = _run_steps(fork)
    assert commands["Install locked web dependencies"] == "npm ci"
    assert commands["Install Playwright-pinned Chromium runtime"] == (
        "npx --no-install playwright install --with-deps chromium"
    )
    assert commands["Exercise production Worker lifecycle and layouts"] == (
        "npm run test:e2e"
    )


def test_external_actions_are_immutable_and_artifacts_identify_attempts() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}

    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                assert FULL_ACTION_SHA.fullmatch(str(step["uses"]))
        artifact = next(
            step
            for step in job["steps"]
            if step.get("name") == "Retain Playwright evidence"
        )
        name = artifact["with"]["name"]
        assert "${{ github.run_id }}" in name
        assert "${{ github.run_attempt }}" in name
        assert artifact["with"]["path"] == (
            "src/rate_of_closure/web/playwright-report/\n"
            "src/rate_of_closure/web/test-results/\n"
        )


def test_touched_hosted_runner_guard_uses_only_immutable_actions() -> None:
    loaded = yaml.safe_load(RUNNER_GUARD_PATH.read_text(encoding="utf-8"))
    steps = loaded["jobs"]["reject-hosted-runner-routing"]["steps"]

    action_uses = [str(step["uses"]) for step in steps if "uses" in step]
    assert action_uses
    assert all(FULL_ACTION_SHA.fullmatch(value) for value in action_uses)
