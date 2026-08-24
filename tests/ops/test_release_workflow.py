"""Contracts for bounded, file-backed release notes in release automation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_release_notes_cross_job_boundaries_as_an_artifact() -> None:
    jobs = _workflow()["jobs"]

    assert "changelog_entry" not in jobs["analyse-commits"].get("outputs", {})
    assert _step(jobs["analyse-commits"], "Upload release notes")["with"]["path"] == (
        "release-changelog-entry.md"
    )
    for job_name in ("bump-version", "github-release"):
        download = _step(jobs[job_name], "Download release notes")
        assert download["with"]["name"] == "release-changelog-entry"


def test_release_steps_read_notes_from_a_file_not_an_environment_variable() -> None:
    jobs = _workflow()["jobs"]
    relevant_steps = (
        _step(jobs["bump-version"], "Update CHANGELOG.md"),
        _step(jobs["bump-version"], "Open version bump PR"),
        _step(jobs["github-release"], "Create GitHub Release"),
    )

    for step in relevant_steps:
        assert "CHANGELOG_ENTRY" not in step.get("env", {})
        assert "release-changelog-entry.md" in step["run"]
