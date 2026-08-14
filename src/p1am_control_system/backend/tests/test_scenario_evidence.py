"""Declarative synthetic scenario and acceptance-evidence contracts."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evidence_package import EvidencePackageService  # noqa: E402
from scenario_evidence import (  # noqa: E402
    RepresentativeScenarioAdapter,
    ScenarioDefinition,
    ScenarioRunner,
    ScenarioStep,
)

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _scenario() -> ScenarioDefinition:
    return ScenarioDefinition(
        name="Synthetic transport fault and recovery",
        data_classification="synthetic",
        not_for_live_control=True,
        steps=[
            ScenarioStep(
                step_id="disconnect",
                action="transport_disconnect",
                target="SYNTHETIC.TRANSPORT",
                parameters={},
                expected={"connected": False},
                timing_window_ms=100,
            ),
            ScenarioStep(
                step_id="recover",
                action="transport_recover",
                target="SYNTHETIC.TRANSPORT",
                parameters={},
                expected={"connected": True},
                timing_window_ms=100,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_synthetic_fault_and_recovery_emit_self_contained_evidence() -> None:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    adapter = RepresentativeScenarioAdapter(clock=lambda: now)
    runner = ScenarioRunner(
        adapter,
        software_revision="software-test-1",
        configuration_revision="cfg-000001-proof",
        clock=lambda: now,
    )
    evidence = await runner.run(_scenario())
    package = EvidencePackageService().create(_scenario(), evidence)
    verified = EvidencePackageService().verify(package.payload, package.sha256)

    assert evidence.passed is True
    assert [result.passed for result in evidence.results] == [True, True]
    assert all(result.alarms for result in evidence.results)
    assert all(result.audit_events for result in evidence.results)
    assert evidence.results[0].alarms[0].alarm_id.startswith("SYNTHETIC.")
    assert evidence.signoff.prepared_by is None
    assert evidence.signoff.approved_by is None
    assert verified.evidence.evidence_id == evidence.evidence_id
    assert verified.manifest.data_classification == "synthetic"


def test_scenario_contract_rejects_non_synthetic_or_live_targets() -> None:
    with pytest.raises(ValueError, match="SYNTHETIC"):
        ScenarioStep(
            step_id="bad",
            action="set_value",
            target="REAL.TAG",
            parameters={"value": 1},
            expected={"value": 1},
            timing_window_ms=100,
        )

    with pytest.raises(ValueError):
        ScenarioDefinition(
            name="Bad classification",
            data_classification="confidential",
            not_for_live_control=True,
            steps=[],
        )


@pytest.mark.asyncio
async def test_timing_window_failure_is_evidence_not_an_exception() -> None:
    start = datetime(2026, 8, 3, tzinfo=UTC)

    class SlowAdapter:
        async def execute(self, step):
            from scenario_evidence import StepObservation

            return StepObservation(
                step_id=step.step_id,
                started_at=start,
                completed_at=start + timedelta(milliseconds=200),
                observed={"connected": False},
            )

    runner = ScenarioRunner(
        SlowAdapter(),
        software_revision="software-test-1",
        configuration_revision="cfg-000001-proof",
        clock=lambda: start,
    )
    evidence = await runner.run(
        _scenario().model_copy(update={"steps": [_scenario().steps[0]]})
    )

    assert evidence.passed is False
    assert evidence.results[0].within_timing_window is False
    assert "timing" in evidence.results[0].diagnostic.lower()
