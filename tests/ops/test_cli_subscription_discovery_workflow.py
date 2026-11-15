"""Tier-2 subscription CLI discovery workflow contract (#5727 wave 3.5)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "cli-subscription-discovery.yml"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "run_cli_discovery_probes.py"


def _workflow() -> dict[str, object]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_tier2_discovery_workflow_is_dispatch_only_and_advisory() -> None:
    """No push/PR trigger; job records probes without blocking main CI."""
    assert WORKFLOW.is_file(), "missing cli-subscription-discovery.yml"
    assert RUNNER_SCRIPT.is_file(), "missing scripts/run_cli_discovery_probes.py"
    on = _workflow().get("on") or _workflow().get(True)
    assert isinstance(on, dict)
    assert "workflow_dispatch" in on
    assert "push" not in on
    assert "pull_request" not in on
    assert "merge_group" not in on

    job = _workflow()["jobs"]["tier2-cli-discovery"]
    assert job.get("continue-on-error") is True
    assert job["runs-on"] == "d-sorg-fleet"
    steps = job["steps"]
    run_probe = next(
        step for step in steps if step.get("name") == "Run Tier-2 CLI discovery probes"
    )
    assert "python3 scripts/run_cli_discovery_probes.py" in run_probe["run"]
    assert "cli-discovery-results.json" in run_probe["run"]
