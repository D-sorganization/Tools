"""Isolated synthetic FAT/HIL scenarios and hashed acceptance evidence."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

SCENARIO_SCHEMA = "p1am.synthetic-scenario/v1"
EVIDENCE_SCHEMA = "p1am.acceptance-evidence/v1"
PACKAGE_SCHEMA = "p1am.acceptance-package/v1"
PACKAGE_ENTRIES = frozenset({"manifest.json", "scenario.json", "evidence.json"})
MAX_PACKAGE_BYTES = 5_000_000

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

    schema_id: Literal[SCENARIO_SCHEMA] = SCENARIO_SCHEMA
    name: str = Field(min_length=1, max_length=200)
    data_classification: Literal["synthetic"]
    not_for_live_control: Literal[True]
    steps: list[ScenarioStep] = Field(min_length=1, max_length=100)
    limitations: tuple[str, ...] = (
        "Executes only against an isolated representative in-memory adapter.",
        "Does not prove field wiring or independent protection behavior.",
        "Timing results exclude live networks, controllers, and equipment.",
    )


class StepObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    started_at: datetime
    completed_at: datetime
    observed: dict[str, object]


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

    schema_id: Literal[EVIDENCE_SCHEMA] = EVIDENCE_SCHEMA
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


class EvidencePackageManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_id: Literal[PACKAGE_SCHEMA] = PACKAGE_SCHEMA
    evidence_id: str
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True
    entries: dict[str, str]


@dataclass(frozen=True)
class EvidenceArtifact:
    payload: bytes = field(repr=False)
    sha256: str
    manifest: EvidencePackageManifest


@dataclass(frozen=True)
class VerifiedEvidencePackage:
    manifest: EvidencePackageManifest
    scenario: ScenarioDefinition
    evidence: ScenarioEvidence
    package_sha256: str


class ScenarioAdapter(Protocol):
    async def execute(self, step: ScenarioStep) -> StepObservation: ...


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(model: BaseModel) -> bytes:
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
        return StepObservation(
            step_id=step.step_id,
            started_at=started,
            completed_at=completed,
            observed=dict(state),
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
        scenario_sha = _hash(_canonical(scenario))
        identity_material = (
            f"{scenario_sha}|{started.isoformat()}|{self._software_revision}|"
            f"{self._configuration_revision}"
        ).encode()
        return ScenarioEvidence(
            evidence_id=f"evidence-{_hash(identity_material)[:20]}",
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


class EvidencePackageService:
    def create(
        self, scenario: ScenarioDefinition, evidence: ScenarioEvidence
    ) -> EvidenceArtifact:
        scenario_payload = _canonical(scenario)
        evidence_payload = _canonical(evidence)
        if evidence.scenario_sha256 != _hash(scenario_payload):
            raise ValueError("evidence does not identify the supplied scenario")
        manifest = EvidencePackageManifest(
            evidence_id=evidence.evidence_id,
            entries={
                "scenario.json": _hash(scenario_payload),
                "evidence.json": _hash(evidence_payload),
            },
        )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest.model_dump_json(indent=2))
            archive.writestr("scenario.json", scenario_payload)
            archive.writestr("evidence.json", evidence_payload)
        payload = output.getvalue()
        return EvidenceArtifact(payload, _hash(payload), manifest)

    def verify(
        self, payload: bytes, expected_sha256: str | None = None
    ) -> VerifiedEvidencePackage:
        if (
            not isinstance(payload, bytes)
            or not payload
            or len(payload) > MAX_PACKAGE_BYTES
        ):
            raise ValueError("evidence package size is outside the allowed boundary")
        package_sha = _hash(payload)
        if expected_sha256 is not None and package_sha != expected_sha256.lower():
            raise ValueError("evidence package checksum does not match")
        try:
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                if frozenset(archive.namelist()) != PACKAGE_ENTRIES:
                    raise ValueError("evidence package entries are not allowed")
                manifest_payload = archive.read("manifest.json")
                scenario_payload = archive.read("scenario.json")
                evidence_payload = archive.read("evidence.json")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError("evidence package is not a valid archive") from exc
        manifest = EvidencePackageManifest.model_validate_json(manifest_payload)
        for name, content in (
            ("scenario.json", scenario_payload),
            ("evidence.json", evidence_payload),
        ):
            if manifest.entries.get(name) != _hash(content):
                raise ValueError(f"{name} checksum does not match")
        scenario = ScenarioDefinition.model_validate_json(scenario_payload)
        evidence = ScenarioEvidence.model_validate_json(evidence_payload)
        if evidence.scenario_sha256 != _hash(_canonical(scenario)):
            raise ValueError("evidence scenario identity does not match")
        if evidence.evidence_id != manifest.evidence_id:
            raise ValueError("evidence identity does not match the manifest")
        return VerifiedEvidencePackage(manifest, scenario, evidence, package_sha)
