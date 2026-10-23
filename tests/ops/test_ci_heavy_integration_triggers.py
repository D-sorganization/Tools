"""Workflow trigger contracts for heavy integration lanes (Tools #5694)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAVY_INTEGRATION = REPO_ROOT / ".github" / "workflows" / "heavy-integration-tests.yml"
HEAVY_OPT_IN = REPO_ROOT / ".github" / "workflows" / "heavy-tests-opt-in.yml"
CROSS_REPO_PYTHON = (
    REPO_ROOT / ".github" / "workflows" / "cross-repo-python-integration.yml"
)
CROSS_REPO_RUST = (
    REPO_ROOT / ".github" / "workflows" / "cross-repo-rust-integration.yml"
)

_SHARED_PYTHON_PATHS = {
    "src/shared/python/**",
    "src/python/**",
    "manifests/**",
    ".github/workflows/cross-repo-python-integration.yml",
}
_RUST_PATHS = {
    "rust_core/**",
    "Cargo.toml",
    "rust-toolchain.toml",
}
_HEAVY_OPT_IN_LABEL = "run-heavy-integration"


def _triggers(path: Path) -> dict[str, object]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (True, "on"):
        if key in workflow:
            on = workflow[key]
            break
    else:
        raise KeyError(f"{path.name}: missing on: trigger block")
    assert isinstance(on, dict)
    return on


def test_heavy_integration_runs_on_merge_queue_and_main_push() -> None:
    triggers = _triggers(HEAVY_INTEGRATION)

    assert "pull_request" not in triggers
    assert "merge_group" in triggers
    push = triggers["push"]
    assert isinstance(push, dict)
    assert push.get("branches") == ["main"]


def test_heavy_opt_in_uses_labelled_pull_request_not_every_sync() -> None:
    triggers = _triggers(HEAVY_OPT_IN)
    pull_request = triggers.get("pull_request")
    assert isinstance(pull_request, dict)
    assert pull_request.get("types") == [
        "opened",
        "synchronize",
        "reopened",
        "labeled",
    ]

    workflow = yaml.safe_load(HEAVY_OPT_IN.read_text(encoding="utf-8"))
    job = workflow["jobs"]["run-heavy-tests"]
    assert isinstance(job, dict)
    job_if = job.get("if", "")
    assert _HEAVY_OPT_IN_LABEL in job_if
    assert "workflow_dispatch" in job_if


def test_cross_repo_python_integration_runs_on_merge_queue_not_default_pr() -> None:
    triggers = _triggers(CROSS_REPO_PYTHON)

    assert "pull_request" not in triggers
    assert "merge_group" in triggers
    merge_group = triggers["merge_group"]
    assert isinstance(merge_group, dict)
    assert set(merge_group.get("paths", [])) == _SHARED_PYTHON_PATHS


def test_cross_repo_rust_integration_runs_on_merge_queue_not_default_pr() -> None:
    triggers = _triggers(CROSS_REPO_RUST)

    assert "pull_request" not in triggers
    assert "merge_group" in triggers
    merge_group = triggers["merge_group"]
    assert isinstance(merge_group, dict)
    assert set(merge_group.get("paths", [])) == _RUST_PATHS
