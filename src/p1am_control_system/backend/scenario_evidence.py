"""Isolated synthetic FAT/HIL scenarios and hashed acceptance evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

SCENARIO_SCHEMA = "p1am.synthetic-scenario/v1"
EVIDENCE_SCHEMA = "p1am.acceptance-evidence/v1"

ScenarioAction = Literal[
    "set_value",
    "set_quality",
    "transport_disconnect",
    "transport_recover",
]


class ScenarioStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str = Field(min_length=1, max_length=100)
    action: ScenarioAction
    target: str = Field(min_length=1, max_length=200)
    parameters: dict[str, object]
    expected: dict[str, object]
    timing_window_ms: int = Field(gt=0, le=60_000)

    @field_validator("target")
    @classmethod
    def _synthetic_target(cls, value: str) -> str:
        if not value.startswith("SYNTHETIC."):
            raise ValueError("scenario targets must begin with SYNTHETIC.")
        return value


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal["p1am.synthetic-scenario/v1"] = "p1am.synthetic-scenario/v1"
    name: str = Field(min_length=1, max_length=200)
    data_classification: Literal["synthetic"]
    not_for_live_control: Literal[True]
    steps: list[ScenarioStep] = Field(min_length=1, max_length=100)
    limitations: tuple[str, ...] = (
        "Executes only against an isolated representative in-memory adapter.",
        "Does not prove field wiring or independent protection behavior.",
        "Timing results exclude live networks, controllers, and equipment.",
    )


class SyntheticAlarmRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    alarm_id: str
    lifecycle: Literal["unacknowledged", "returned_unacknowledged"]
    priority: Literal["high"] = "high"
    source: Literal["synthetic_scenario"] = "synthetic_scenario"


class SyntheticAuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor: Literal["synthetic.scenario.runner"] = "synthetic.scenario.runner"
    action: ScenarioAction
    target: str
    outcome: Literal["succeeded"] = "succeeded"
    timestamp: datetime


class StepObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    started_at: datetime
    completed_at: datetime
    observed: dict[str, object]
    alarms: tuple[SyntheticAlarmRecord, ...] = ()
    audit_events: tuple[SyntheticAuditRecord, ...] = ()


class StepEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    action: ScenarioAction
    target: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float = Field(ge=0)
    expected: dict[str, object]
    observed: dict[str, object]
    alarms: tuple[SyntheticAlarmRecord, ...]
    audit_events: tuple[SyntheticAuditRecord, ...]
    behavior_matched: bool
    within_timing_window: bool
    passed: bool
    diagnostic: str


class EvidenceSignoff(BaseModel):
    model_config = ConfigDict(frozen=True)

    signoff_required: bool = True
    prepared_by: str | None = None
    witnessed_by: str | None = None
    approved_by: str | None = None
    signed_at: datetime | None = None


class ScenarioEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal["p1am.acceptance-evidence/v1"] = "p1am.acceptance-evidence/v1"
    evidence_id: str
    scenario_name: str
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    software_revision: str
    configuration_revision: str
    started_at: datetime
    completed_at: datetime
    passed: bool
    results: tuple[StepEvidence, ...]
    limitations: tuple[str, ...]
    signoff: EvidenceSignoff = EvidenceSignoff()


class ScenarioAdapter(Protocol):
    async def execute(self, step: ScenarioStep) -> StepObservation: ...


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_model_bytes(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _required_revision(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


class RepresentativeScenarioAdapter:
    """In-memory adapter that has no path to a field driver or runtime control."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._state: dict[str, dict[str, object]] = {
            "SYNTHETIC.TRANSPORT": {"connected": True}
        }

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    async def execute(self, step: ScenarioStep) -> StepObservation:
        if not isinstance(step, ScenarioStep):
            raise TypeError("step must be a ScenarioStep")
        started = self._now()
        state = self._state.setdefault(step.target, {})
        if step.action == "transport_disconnect":
            state["connected"] = False
        elif step.action == "transport_recover":
            state["connected"] = True
        elif step.action == "set_value":
            value = step.parameters.get("value")
            if not isinstance(value, int | float):
                raise ValueError("set_value requires a numeric value")
            state["value"] = value
        elif step.action == "set_quality":
            quality = step.parameters.get("quality")
            if quality not in {"good", "uncertain", "bad", "stale", "simulated"}:
                raise ValueError("set_quality requires a canonical quality")
            state["quality"] = quality
        completed = self._now()
        returned = step.action in {"transport_recover"} or (
            step.action == "set_quality" and state.get("quality") == "good"
        )
        alarm_id = (
            "SYNTHETIC.COMMUNICATIONS"
            if step.action.startswith("transport_")
            else "SYNTHETIC.DATA_QUALITY"
        )
        return StepObservation(
            step_id=step.step_id,
            started_at=started,
            completed_at=completed,
            observed=dict(state),
            alarms=(
                SyntheticAlarmRecord(
                    alarm_id=alarm_id,
                    lifecycle=(
                        "returned_unacknowledged" if returned else "unacknowledged"
                    ),
                ),
            ),
            audit_events=(
                SyntheticAuditRecord(
                    action=step.action,
                    target=step.target,
                    timestamp=completed,
                ),
            ),
        )


class ScenarioRunner:
    """Run validated steps and record failures as evidence rather than hiding them."""

    def __init__(
        self,
        adapter: ScenarioAdapter,
        software_revision: str,
        configuration_revision: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(getattr(adapter, "execute", None)):
            raise TypeError("adapter must implement execute")
        self._adapter = adapter
        self._software_revision = _required_revision(
            software_revision, "software_revision"
        )
        self._configuration_revision = _required_revision(
            configuration_revision, "configuration_revision"
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("clock must return an aware datetime")
        return now

    @staticmethod
    def _step_evidence(
        step: ScenarioStep, observation: StepObservation
    ) -> StepEvidence:
        if observation.step_id != step.step_id:
            raise ValueError("adapter returned the wrong step identity")
        duration = (
            observation.completed_at - observation.started_at
        ).total_seconds() * 1000
        if duration < 0:
            raise ValueError("adapter returned a negative step duration")
        behavior = all(
            observation.observed.get(key) == value
            for key, value in step.expected.items()
        )
        timing = duration <= step.timing_window_ms
        passed = behavior and timing
        reasons = []
        if not behavior:
            reasons.append("expected behavior did not match")
        if not timing:
            reasons.append("timing window exceeded")
        return StepEvidence(
            step_id=step.step_id,
            action=step.action,
            target=step.target,
            started_at=observation.started_at,
            completed_at=observation.completed_at,
            duration_ms=duration,
            expected=step.expected,
            observed=observation.observed,
            alarms=observation.alarms,
            audit_events=observation.audit_events,
            behavior_matched=behavior,
            within_timing_window=timing,
            passed=passed,
            diagnostic="passed" if passed else "; ".join(reasons),
        )

    async def run(self, scenario: ScenarioDefinition) -> ScenarioEvidence:
        if not isinstance(scenario, ScenarioDefinition):
            raise TypeError("scenario must be a ScenarioDefinition")
        started = self._now()
        results = tuple(
            [
                self._step_evidence(step, await self._adapter.execute(step))
                for step in scenario.steps
            ]
        )
        completed = self._now()
        scenario_sha = sha256_bytes(canonical_model_bytes(scenario))
        identity_material = (
            f"{scenario_sha}|{started.isoformat()}|{self._software_revision}|"
            f"{self._configuration_revision}"
        ).encode()
        return ScenarioEvidence(
            evidence_id=f"evidence-{sha256_bytes(identity_material)[:20]}",
            scenario_name=scenario.name,
            scenario_sha256=scenario_sha,
            software_revision=self._software_revision,
            configuration_revision=self._configuration_revision,
            started_at=started,
            completed_at=completed,
            passed=all(result.passed for result in results),
            results=results,
            limitations=scenario.limitations,
        )
