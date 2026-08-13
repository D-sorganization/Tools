"""Security and reproducibility contracts for the Rate Playwright workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
PR_WORKFLOW_PATH = WORKFLOW_DIR / "rate-web-playwright.yml"
TRUSTED_WORKFLOW_PATH = WORKFLOW_DIR / "rate-web-playwright-trusted.yml"
RUNNER_GUARD_PATH = WORKFLOW_DIR / "local-only-runner-guard.yml"
FULL_ACTION_SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")
EVIDENCE_PATHS = (
    "src/rate_of_closure/web/playwright-report/\n"
    "src/rate_of_closure/web/test-results/\n"
)


def _workflow(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _run_steps(job: dict[str, Any]) -> dict[str, str]:
    return {
        str(step["name"]): str(step["run"])
        for step in job["steps"]
        if isinstance(step, dict) and "name" in step and "run" in step
    }


def _checkout(job: dict[str, Any]) -> dict[str, Any]:
    return next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )


def test_pull_request_workflow_is_hosted_only_without_fleet_vocabulary() -> None:
    text = PR_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _workflow(PR_WORKFLOW_PATH)

    assert "d-sorg-fleet" not in text.lower()
    assert "self-hosted" not in text.lower()
    assert "\n  pull_request:" in text
    assert "\n  push:" not in text
    assert "workflow_dispatch" not in text
    assert set(workflow["jobs"]) == {"production-worker-e2e"}
    assert workflow["jobs"]["production-worker-e2e"]["runs-on"] == "ubuntu-latest"


def test_trusted_workflow_has_no_pull_request_or_untrusted_ref_seam() -> None:
    text = TRUSTED_WORKFLOW_PATH.read_text(encoding="utf-8")
    jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]

    assert "pull_request" not in text
    assert "github.event.pull_request" not in text
    assert "inputs." not in text
    assert "${{ github.ref" not in text
    assert "${{ github.head_ref" not in text
    assert "${{ github.sha" not in text
    assert "\n  push:" in text
    assert "workflow_dispatch:" in text
    assert all(job["runs-on"] == "d-sorg-fleet" for job in jobs.values())

    push_checkout = _checkout(jobs["push-production-worker-e2e"])
    manual_checkout = _checkout(jobs["manual-production-worker-e2e"])
    assert "with" not in push_checkout or "ref" not in push_checkout["with"]
    assert manual_checkout["with"]["ref"] == "main"


def test_pr_and_trusted_jobs_run_the_same_locked_production_gate() -> None:
    pr_job = _workflow(PR_WORKFLOW_PATH)["jobs"]["production-worker-e2e"]
    trusted_jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]
    expected_commands = _run_steps(pr_job)

    assert all(_run_steps(job) == expected_commands for job in trusted_jobs.values())
    assert expected_commands["Install locked web dependencies"] == "npm ci"
    assert expected_commands["Install Playwright-pinned Chromium runtime"] == (
        "npx --no-install playwright install --with-deps chromium"
    )
    assert expected_commands["Exercise production Worker lifecycle and layouts"] == (
        "npm run test:e2e"
    )


def test_external_actions_are_immutable_and_artifacts_identify_attempts() -> None:
    for path in (PR_WORKFLOW_PATH, TRUSTED_WORKFLOW_PATH):
        workflow = _workflow(path)
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
            assert artifact["with"]["path"] == EVIDENCE_PATHS


def test_touched_hosted_runner_guard_uses_only_immutable_actions() -> None:
    loaded = _workflow(RUNNER_GUARD_PATH)
    steps = loaded["jobs"]["reject-hosted-runner-routing"]["steps"]

    action_uses = [str(step["uses"]) for step in steps if "uses" in step]
    assert action_uses
    assert all(FULL_ACTION_SHA.fullmatch(value) for value in action_uses)
