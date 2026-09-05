"""Guard against required status checks that can never report.

Branch protection waits for every required context before allowing a merge. A
`paths`/`paths-ignore` filter on a workflow's ``pull_request`` trigger skips the
whole workflow when a PR touches only filtered paths -- so the context never
reports at all. The PR then sits BLOCKED with zero failures and nothing
pending, indistinguishable from a check that simply has not started yet. It can
never merge, and no amount of waiting or re-running helps.

This is not hypothetical. ``ci-standard.yml`` filtered ``pull_request`` on
``LICENSE`` and ``.gitignore`` while ``quality-gate`` was a required context,
so a ``.gitignore``-only PR was permanently unmergeable -- observed on
D-sorganization/Repository_Management#1529 before the filter was removed. The
same class of bug is recorded in D-sorganization/Runner_Dashboard#1167.

``push`` filters are deliberately unaffected: post-merge CI does not gate a
required context, so ignoring cheap paths there is fine. ``types`` filters are
fine too -- they cannot skip a workflow based on which files a PR touches.

REQUIRED_CONTEXTS below is a local declaration, not an observation. The
authoritative list lives in branch protection. Refresh it with:

    gh api repos/D-sorganization/<REPO>/rules/branches/main \
      --jq '[.[]|select(.type=="required_status_checks")
             |.parameters.required_status_checks[].context]'

Listing a context that is not actually required is harmless -- the guard simply
protects one more workflow than it strictly must. Omitting one that IS required
is the failure mode this file exists to prevent, hence
``test_declared_required_contexts_are_provided_by_a_workflow`` below, which
fails if a declared context has no job to report it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

#: Contexts that branch protection requires on the default branch. See the
#: module docstring for how to refresh this.
REQUIRED_CONTEXTS = frozenset({"quality-gate", "tests"})

#: `on:` parses as the YAML boolean True unless quoted, so both keys are tried.
_ON_KEYS = (True, "on")


def _workflow_files() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(p for p in WORKFLOWS_DIR.iterdir() if p.suffix in (".yml", ".yaml"))


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name}: top-level YAML must be a mapping"
    return data


def _triggers(data: dict[str, Any]) -> Any:
    for key in _ON_KEYS:
        if key in data:
            return data[key]
    return {}


def _providers() -> dict[str, list[str]]:
    """Map each declared required context to the workflows defining that job."""
    found: dict[str, list[str]] = {c: [] for c in REQUIRED_CONTEXTS}
    for path in _workflow_files():
        try:
            data = _load(path)
        except yaml.YAMLError:  # pragma: no cover - malformed YAML is its own bug
            continue
        jobs = data.get("jobs") or {}
        if not isinstance(jobs, dict):
            continue
        for context in REQUIRED_CONTEXTS & set(jobs):
            found[context].append(path.name)
    return found


def test_workflows_directory_is_present() -> None:
    """Sanity check: without this, every assertion below would pass vacuously."""
    assert _workflow_files(), f"no workflow files found under {WORKFLOWS_DIR}"


def test_declared_required_contexts_are_provided_by_a_workflow() -> None:
    """Every declared context must have a job that can report it.

    Without this, renaming a job would silently disable the guard below rather
    than failing loudly: the filter check only inspects workflows that provide
    a required context, so a context nothing provides is never checked.
    """
    missing = sorted(c for c, files in _providers().items() if not files)
    assert not missing, (
        f"REQUIRED_CONTEXTS names {missing}, but no workflow under "
        f"{WORKFLOWS_DIR.name}/ defines a job with that id. Either the job was "
        "renamed -- in which case branch protection is now waiting on a context "
        "nothing reports, and every PR is blocked -- or this list is stale. See "
        "the module docstring for how to refresh it."
    )


@pytest.mark.parametrize("filter_key", ["paths", "paths-ignore"])
def test_required_context_workflows_have_no_pull_request_path_filter(
    filter_key: str,
) -> None:
    """A workflow providing a required context must not be path-filtered on PRs."""
    offenders: list[str] = []

    for path in _workflow_files():
        try:
            data = _load(path)
        except yaml.YAMLError:  # pragma: no cover
            continue
        jobs = data.get("jobs") or {}
        if not isinstance(jobs, dict):
            continue
        provided = REQUIRED_CONTEXTS & set(jobs)
        if not provided:
            continue

        triggers = _triggers(data)
        if not isinstance(triggers, dict):
            continue
        pull_request = triggers.get("pull_request")
        if not isinstance(pull_request, dict):
            continue

        value = pull_request.get(filter_key)
        if value:
            offenders.append(
                f"{path.name} provides {sorted(provided)} but filters "
                f"`pull_request` on `{filter_key}: {value}`"
            )

    assert not offenders, (
        "A required status check cannot report if its workflow is skipped by a "
        "path filter, so any PR touching only those paths is permanently "
        "unmergeable -- BLOCKED with zero failures and nothing pending.\n  "
        + "\n  ".join(offenders)
        + "\nRemove the filter from `pull_request`. Keeping it on `push` is "
        "fine: post-merge CI does not gate a required context."
    )
