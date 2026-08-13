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
PLAYWRIGHT_EVIDENCE_PATHS = (
    "src/rate_of_closure/web/playwright-report/\n"
    "src/rate_of_closure/web/test-results/\n"
)
PR_EVIDENCE_PATHS = PLAYWRIGHT_EVIDENCE_PATHS + "rate-pyqt-screenshots/\n"
PYQT_AUTHORITY_PATHS = {
    "src/rate_of_closure/club/**",
    "src/rate_of_closure/model.py",
    "src/rate_of_closure/plotting/**",
    "src/rate_of_closure/simulation/**",
    "src/rate_of_closure/variation/**",
    "src/rate_of_closure/ui/pyqt6/**",
    "src/shared/python/swing_sim/variation/**",
    "tests/rate_of_closure/pyqt_variation_render_probe.py",
    "tests/rate_of_closure/test_pyqt_variation_rendered_interactions.py",
    "tests/ops/test_rate_web_playwright_workflow.py",
    "pyproject.toml",
}


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


def test_trusted_workflow_is_main_push_only_without_untrusted_ref_seam() -> None:
    text = TRUSTED_WORKFLOW_PATH.read_text(encoding="utf-8")
    jobs = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]

    assert "pull_request" not in text
    assert "github.event.pull_request" not in text
    assert "inputs." not in text
    assert "${{ github.ref" not in text
    assert "${{ github.head_ref" not in text
    assert "${{ github.sha" not in text
    assert "\n  push:" in text
    assert "workflow_dispatch" not in text
    assert set(jobs) == {"push-production-worker-e2e"}
    assert all(job["runs-on"] == "d-sorg-fleet" for job in jobs.values())

    push_checkout = _checkout(jobs["push-production-worker-e2e"])
    assert "with" not in push_checkout or "ref" not in push_checkout["with"]


def test_pr_runs_locked_cross_browser_gate_and_trusted_keeps_chromium_gate() -> None:
    pr_job = _workflow(PR_WORKFLOW_PATH)["jobs"]["production-worker-e2e"]
    trusted_job = _workflow(TRUSTED_WORKFLOW_PATH)["jobs"]["push-production-worker-e2e"]
    pr_commands = _run_steps(pr_job)
    trusted_commands = _run_steps(trusted_job)

    assert pr_commands["Install locked web dependencies"] == "npm ci"
    assert trusted_commands["Install locked web dependencies"] == "npm ci"
    assert pr_commands["Install Playwright-pinned browser runtimes"] == (
        "npx --no-install playwright install --with-deps chromium firefox webkit"
    )
    assert trusted_commands["Install Playwright-pinned Chromium runtime"] == (
        "npx --no-install playwright install --with-deps chromium"
    )
    assert (
        pr_commands["Exercise production Worker lifecycle, layouts, and browser parity"]
        == "npm run test:e2e"
    )
    assert pr_commands["Install bounded PyQt render dependencies"] == (
        'python -m pip install -e ".[gui]" "scipy>=1.10,<1.18" pytest '
        '"pytest-xdist>=3.6,<4"'
    )
    assert pr_commands[
        "Exercise PyQt rendered interactions at 100 and 150 percent DPI"
    ] == (
        "python -m pytest "
        "tests/rate_of_closure/test_pyqt_variation_rendered_interactions.py -q -n 0"
    )
    assert (
        trusted_commands["Exercise production Worker lifecycle and layouts"]
        == "npm run test:e2e -- --project=chromium-desktop --project=chromium-narrow"
    )


def test_pr_trigger_tracks_every_pyqt_render_authority() -> None:
    workflow = _workflow(PR_WORKFLOW_PATH)
    assert PYQT_AUTHORITY_PATHS <= set(workflow[True]["pull_request"]["paths"])


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
            expected_paths = (
                PR_EVIDENCE_PATHS
                if path == PR_WORKFLOW_PATH
                else PLAYWRIGHT_EVIDENCE_PATHS
            )
            assert artifact["with"]["path"] == expected_paths


def test_touched_hosted_runner_guard_uses_only_immutable_actions() -> None:
    loaded = _workflow(RUNNER_GUARD_PATH)
    steps = loaded["jobs"]["reject-hosted-runner-routing"]["steps"]

    action_uses = [str(step["uses"]) for step in steps if "uses" in step]
    assert action_uses
    assert all(FULL_ACTION_SHA.fullmatch(value) for value in action_uses)
