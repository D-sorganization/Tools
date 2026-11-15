"""Hosted-runner allowlists stay aligned with #5695 migrations."""

from __future__ import annotations

from pathlib import Path

from scripts import check_local_only_workflows as hosted_guard

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".github" / "workflows" / "local-only-runner-guard.yml"

HOSTED_WORKFLOW_FILES = frozenset(
    {
        "anti-phantom-merge.yml",
        "full-suite-nightly.yml",
        "nightly-full-repo-quality.yml",
    }
)


def test_local_only_guard_allowlists_github_hosted_workflow_files() -> None:
    text = GUARD.read_text(encoding="utf-8")
    for name in HOSTED_WORKFLOW_FILES:
        assert f'"{name}"' in text


def test_check_local_only_workflows_allowlists_hosted_migration_jobs() -> None:
    legacy = {Path(p).name for p in hosted_guard.LEGACY_HOSTED_RUNNER_ALLOWLIST}
    assert HOSTED_WORKFLOW_FILES <= legacy
    job_ids = {
        "anti-phantom-merge.yml": "guard",
        "full-suite-nightly.yml": "full-suite",
        "nightly-full-repo-quality.yml": "full-repo-quality",
    }
    for name, job_id in job_ids.items():
        key = (f".github/workflows/{name}", job_id)
        assert key in hosted_guard.HOSTED_RUNNER_ALLOWLIST
