"""Scheduled workflow runner and cadence contracts (#5700, #5701)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

FULL_SUITE_NIGHTLY = REPO_ROOT / ".github" / "workflows" / "full-suite-nightly.yml"
HEAVY_INTEGRATION = REPO_ROOT / ".github" / "workflows" / "heavy-integration-tests.yml"


def _workflow(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _schedule_crons(path: Path) -> list[str]:
    document = _workflow(path)
    on = document.get("on") or document[True]
    assert isinstance(on, dict)
    schedule = on.get("schedule")
    assert isinstance(schedule, list)
    crons: list[str] = []
    for entry in schedule:
        assert isinstance(entry, dict)
        cron = entry.get("cron")
        assert isinstance(cron, str)
        crons.append(cron)
    return crons


def test_full_suite_nightly_runs_on_hosted_runner() -> None:
    document = _workflow(FULL_SUITE_NIGHTLY)
    job = document["jobs"]["full-suite"]
    assert isinstance(job, dict)
    assert job.get("runs-on") == "ubuntu-latest"


def test_heavy_integration_nightly_schedule_avoids_four_and_five_utc() -> None:
    crons = _schedule_crons(HEAVY_INTEGRATION)
    assert len(crons) == 1
    cron = crons[0]
    parts = cron.split()
    assert len(parts) == 5
    minute, hour = parts[0], parts[1]
    assert minute == "0"
    assert hour not in {"4", "5"}
