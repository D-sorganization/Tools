"""Reproducible synthetic advisories that cannot write control commands."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Literal

from identity import Principal
from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_DESCRIPTOR = {
    "algorithm": "representative bounded linear projection",
    "model_id": "SYNTHETIC.MODEL.ADVISORY",
    "version": "1.0.0",
}
DEFAULT_MINIMUM = 40.0
DEFAULT_MAXIMUM = 80.0
CONFIDENCE_HALF_WIDTH = 2.5
CONFIDENCE_LEVEL = 0.90
THROUGHPUT_GAIN = 0.35


def _canonical_sha256(payload: object) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: object) -> object:
    """Convert supported immutable contract values for canonical hashing."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _required_text(value: str, name: str) -> str:
    """Normalize required human or synthetic identifiers."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


class AdvisoryRequest(BaseModel):
    """Synthetic observations and target supplied to the advisory model."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    observed_throughput: float
    observed_energy: float
    requested_throughput: float

    @model_validator(mode="after")
    def validate_request(self) -> AdvisoryRequest:
        """Enforce finite inputs and a synthetic dataset boundary."""
        object.__setattr__(
            self,
            "dataset_id",
            _required_text(self.dataset_id, "dataset_id"),
        )
        if not self.dataset_id.startswith("SYNTHETIC."):
            raise ValueError("dataset_id must identify synthetic data")
        values = (
            self.observed_throughput,
            self.observed_energy,
            self.requested_throughput,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("advisory inputs must be finite")
        return self


class ConstraintEnvelope(BaseModel):
    """Permitted range used to bound a recommendation."""

    model_config = ConfigDict(frozen=True)

    minimum: float
    maximum: float
    unit: str

    @model_validator(mode="after")
    def validate_range(self) -> ConstraintEnvelope:
        """Require an ordered, finite constraint interval."""
        if not all(math.isfinite(value) for value in (self.minimum, self.maximum)):
            raise ValueError("constraint values must be finite")
        if self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        object.__setattr__(self, "unit", _required_text(self.unit, "unit"))
        return self


class ConfidenceInterval(BaseModel):
    """Uncertainty interval around one representative estimate."""

    model_config = ConfigDict(frozen=True)

    level: float = Field(gt=0.0, lt=1.0)
    lower: float
    estimate: float
    upper: float

    @model_validator(mode="after")
    def validate_interval(self) -> ConfidenceInterval:
        """Require a finite ordered interval containing the estimate."""
        values = (self.lower, self.estimate, self.upper)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("confidence values must be finite")
        if not self.lower <= self.estimate <= self.upper:
            raise ValueError("confidence interval must contain estimate")
        return self


class ModelProvenance(BaseModel):
    """Identity of the versioned representative model artifact."""

    model_config = ConfigDict(frozen=True)

    model_id: Literal["SYNTHETIC.MODEL.ADVISORY"]
    version: str
    algorithm: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataProvenance(BaseModel):
    """Identity and digest of the exact synthetic model inputs."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_names: tuple[str, ...]


class ReplayEvidence(BaseModel):
    """Digests needed to reproduce and compare an advisory result."""

    model_config = ConfigDict(frozen=True)

    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified: Literal[True] = True


class AdvisoryResult(BaseModel):
    """Review-only model result with explicit safety and provenance labels."""

    model_config = ConfigDict(frozen=True)

    advisory_id: str
    generated_at: datetime
    model: ModelProvenance
    data: DataProvenance
    constraints: ConstraintEnvelope
    confidence: ConfidenceInterval
    recommended_setpoint: float
    recommendation: str
    limitation: str
    replay: ReplayEvidence
    authoritative_write_available: Literal[False] = False
    data_classification: Literal["synthetic"] = "synthetic"
    not_for_live_control: Literal[True] = True


class DispositionDecision(StrEnum):
    """Review outcomes available to an operator."""

    ACCEPTED_FOR_REVIEW = "accepted_for_review"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class AdvisoryDisposition(BaseModel):
    """Requested operator review disposition."""

    model_config = ConfigDict(frozen=True)

    decision: DispositionDecision
    reason: str

    @model_validator(mode="after")
    def validate_reason(self) -> AdvisoryDisposition:
        """Require an explicit review rationale."""
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        return self


class DispositionRecord(BaseModel):
    """Attributable append-only record that never applies a control value."""

    model_config = ConfigDict(frozen=True)

    advisory_id: str
    decision: DispositionDecision
    reason: str
    actor: str
    recorded_at: datetime
    applied_to_control: Literal[False] = False


class AdvisoryService:
    """Evaluate and retain deterministic, non-authoritative advisories."""

    def __init__(self, now: Callable[[], datetime]) -> None:
        if not callable(now):
            raise TypeError("now must be callable")
        self._now = now
        self._results: dict[str, AdvisoryResult] = {}
        self._dispositions: list[DispositionRecord] = []

    def evaluate(self, request: AdvisoryRequest) -> AdvisoryResult:
        """Evaluate one request; postcondition: result is bounded and replayable."""
        if not isinstance(request, AdvisoryRequest):
            raise TypeError("request must be an AdvisoryRequest")
        input_payload = request.model_dump(mode="json")
        input_sha256 = _canonical_sha256(input_payload)
        model = self._model_provenance()
        constraints = ConstraintEnvelope(
            minimum=DEFAULT_MINIMUM,
            maximum=DEFAULT_MAXIMUM,
            unit="synthetic energy index",
        )
        estimate = self._bounded_estimate(request, constraints)
        confidence = ConfidenceInterval(
            level=CONFIDENCE_LEVEL,
            lower=max(constraints.minimum, estimate - CONFIDENCE_HALF_WIDTH),
            estimate=estimate,
            upper=min(constraints.maximum, estimate + CONFIDENCE_HALF_WIDTH),
        )
        core = self._result_core(request, model, constraints, confidence)
        advisory_id = str(core["advisory_id"])
        retained = self._results.get(advisory_id)
        if retained is not None:
            return retained
        result_sha256 = _canonical_sha256(core)
        # Annotated local: see the typing convention note in SPEC.md — CI runs
        # mypy from the repo root, where flat intra-package imports become Any.
        result: AdvisoryResult = AdvisoryResult.model_validate(
            {
                **core,
                "replay": ReplayEvidence(
                    input_sha256=input_sha256,
                    result_sha256=result_sha256,
                ),
            }
        )
        self._results[result.advisory_id] = result
        return result

    def result(self, advisory_id: str) -> AdvisoryResult:
        """Return one retained immutable advisory result."""
        normalized = _required_text(advisory_id, "advisory_id")
        try:
            return self._results[normalized]
        except KeyError as exc:
            raise KeyError("advisory result not found") from exc

    def record_disposition(
        self,
        advisory_id: str,
        disposition: AdvisoryDisposition,
        principal: Principal,
    ) -> DispositionRecord:
        """Append a review disposition without changing the advisory or controls."""
        result = self.result(advisory_id)
        if not isinstance(disposition, AdvisoryDisposition):
            raise TypeError("disposition must be an AdvisoryDisposition")
        if not isinstance(principal, Principal):
            raise TypeError("principal must be a Principal")
        record = DispositionRecord(
            advisory_id=result.advisory_id,
            decision=disposition.decision,
            reason=disposition.reason,
            actor=principal.subject,
            recorded_at=self._now(),
        )
        self._dispositions.append(record)
        return record

    def dispositions(self, advisory_id: str) -> tuple[DispositionRecord, ...]:
        """Return disposition history for one known result."""
        result = self.result(advisory_id)
        return tuple(
            record
            for record in self._dispositions
            if record.advisory_id == result.advisory_id
        )

    @staticmethod
    def _model_provenance() -> ModelProvenance:
        return ModelProvenance(
            model_id="SYNTHETIC.MODEL.ADVISORY",
            version=MODEL_DESCRIPTOR["version"],
            algorithm=MODEL_DESCRIPTOR["algorithm"],
            artifact_sha256=_canonical_sha256(MODEL_DESCRIPTOR),
        )

    @staticmethod
    def _bounded_estimate(
        request: AdvisoryRequest, constraints: ConstraintEnvelope
    ) -> float:
        delta = request.requested_throughput - request.observed_throughput
        unbounded = request.observed_energy + THROUGHPUT_GAIN * delta
        return round(min(constraints.maximum, max(constraints.minimum, unbounded)), 3)

    def _result_core(
        self,
        request: AdvisoryRequest,
        model: ModelProvenance,
        constraints: ConstraintEnvelope,
        confidence: ConfidenceInterval,
    ) -> dict[str, object]:
        input_payload = request.model_dump(mode="json")
        identity_sha256 = _canonical_sha256(
            {"input": input_payload, "model": model.model_dump(mode="json")}
        )
        return {
            "advisory_id": f"ADV-{identity_sha256[:16]}",
            "generated_at": self._now(),
            "model": model,
            "data": DataProvenance(
                dataset_id=request.dataset_id,
                content_sha256=_canonical_sha256(input_payload),
                feature_names=(
                    "observed_throughput",
                    "observed_energy",
                    "requested_throughput",
                ),
            ),
            "constraints": constraints,
            "confidence": confidence,
            "recommended_setpoint": confidence.estimate,
            "recommendation": "Review bounded synthetic setpoint in scenario",
            "limitation": (
                "Representative linear projection only; not validated against a plant "
                "and unable to issue authoritative commands."
            ),
        }


def representative_advisory_request() -> AdvisoryRequest:
    """Return invented inputs for the product demonstration workspace."""
    return AdvisoryRequest(
        dataset_id="SYNTHETIC.DATASET.REPRESENTATIVE-RUN",
        observed_throughput=62.0,
        observed_energy=47.0,
        requested_throughput=68.0,
    )
