"""Runner placement and schedule contracts (#5702–#5704, #5695 follow-ups)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
RUNNER_GUARD = WORKFLOWS / "local-only-runner-guard.yml"

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def _triggers(workflow: dict) -> dict:
    on = workflow.get("on", workflow.get(True, {}))
    return on if isinstance(on, dict) else {}


def _schedule_crons(workflow: dict) -> list[str]:
    on = _triggers(workflow)
    if not on:
        return []
    schedule = on.get("schedule")
    if not isinstance(schedule, list):
        return []
    return [str(entry["cron"]) for entry in schedule if isinstance(entry, dict)]


def _guard_allowlist_jobs() -> set[tuple[str, str]]:
    text = RUNNER_GUARD.read_text(encoding="utf-8")
    match = re.search(r"ALLOWLIST_JOBS\s*=\s*\{([^}]+)\}", text, re.MULTILINE)
    assert match, "local-only-runner-guard ALLOWLIST_JOBS block missing"
    pairs: set[tuple[str, str]] = set()
    for wf, job in re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', match.group(1)):
        pairs.add((wf, job))
    return pairs


def test_benchmark_suite_schedule_is_weekly_advisory() -> None:
    """#5702: advisory benchmark lane must not run daily on the fleet."""
    workflow = _load(WORKFLOWS / "benchmark-suite.yml")
    crons = _schedule_crons(workflow)
    assert crons == ["0 6 * * 0"]
    assert "workflow_dispatch" in _triggers(workflow)


def test_nightly_full_repo_quality_runs_on_github_hosted() -> None:
    """#5703: weekly full-repo audit must not occupy self-hosted runners."""
    workflow = _load(WORKFLOWS / "nightly-full-repo-quality.yml")
    job = workflow["jobs"]["full-repo-quality"]
    assert job["runs-on"] == "ubuntu-latest"
    assert _schedule_crons(workflow) == ["0 3 * * 0"]
    allowlist = _guard_allowlist_jobs()
    assert ("nightly-full-repo-quality.yml", "full-repo-quality") in allowlist


def test_codeql_weekly_scan_routes_schedule_to_github_hosted() -> None:
    """#5704: scheduled CodeQL must not compete with PR lanes on d-sorg-fleet."""
    workflow = _load(WORKFLOWS / "codeql-analysis.yml")
    analyze = workflow["jobs"]["analyze"]
    runs_on = str(analyze["runs-on"])
    assert "ubuntu-latest" in runs_on
    assert "schedule" in runs_on
    assert "d-sorg-fleet" in runs_on
    assert _schedule_crons(workflow) == ["17 6 * * 1"]
    assert ("codeql-analysis.yml", "analyze") in _guard_allowlist_jobs()


def test_bot_ci_trigger_has_no_scheduled_sweep() -> None:
    """#5705: merge-queue covers PR gaps; fleet schedule was removed in #5763."""
    workflow = _load(WORKFLOWS / "Bot-CI-Trigger.yml")
    assert _schedule_crons(workflow) == []
    pick = workflow["jobs"]["pick-runner"]
    assert pick["runs-on"] == "d-sorg-fleet"
