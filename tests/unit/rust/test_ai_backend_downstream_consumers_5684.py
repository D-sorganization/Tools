"""Downstream ai_backend wheel consumer survey (#5684 slice 2, #6023 wave 0.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SURVEY_PATH = _REPO_ROOT / "config" / "ai_backend_downstream_consumer_survey.json"


@pytest.mark.unit
def test_survey_github_issue_linkage_refs_only_not_fixes() -> None:
    """Slice 2 must use Refs #5684, not Fixes (Win/macOS qualification still open)."""
    payload = json.loads(_SURVEY_PATH.read_text(encoding="utf-8"))
    linkage = payload.get("github_issue_linkage")
    assert linkage == "refs_only", (
        "survey PRs must use Refs #5684 in title/body, not Fixes — "
        "native Win/macOS qualification is unmet"
    )


@pytest.mark.unit
def test_downstream_consumer_survey_artifact_exists_and_is_valid() -> None:
    """Slice 2 acceptance: survey is recorded in-repo, not only in issue comments."""
    assert _SURVEY_PATH.is_file(), f"missing {_SURVEY_PATH.relative_to(_REPO_ROOT)}"
    payload = json.loads(_SURVEY_PATH.read_text(encoding="utf-8"))
    assert payload.get("issue") == 5684
    conclusion = payload.get("conclusion")
    assert conclusion in {"no_downstream_wheel_consumer", "consumer_found"}
    repos = payload.get("downstream_repos")
    assert isinstance(repos, list) and repos, (
        "downstream_repos must list surveyed hosts"
    )
    for entry in repos:
        assert "repo" in entry
        assert isinstance(entry.get("imports_ai_backend_wheel"), bool)


@pytest.mark.unit
def test_survey_documents_tools_in_repo_importers() -> None:
    payload = json.loads(_SURVEY_PATH.read_text(encoding="utf-8"))
    importers = payload.get("tools_in_repo_importers")
    assert isinstance(importers, list) and importers
    for row in importers:
        rel = row["path"]
        path = _REPO_ROOT / rel
        assert path.is_file(), f"missing importer path {rel}"
        assert row.get("wheel_required") in {True, False}


@pytest.mark.unit
def test_no_downstream_repo_requires_native_ai_backend_wheel() -> None:
    """Wave 0.4: UpstreamDrift and Gasification_Model do not import ai_backend today."""
    payload = json.loads(_SURVEY_PATH.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "no_downstream_wheel_consumer"
    for entry in payload["downstream_repos"]:
        assert entry["imports_ai_backend_wheel"] is False, entry["repo"]
