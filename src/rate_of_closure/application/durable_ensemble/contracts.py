"""Strict path-free wire contracts for the durable ensemble authority."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.morris.contracts import (
    MorrisBaseRequest,
    parse_morris_base_request,
)
from rate_of_closure.application.morris.request_document import base_document
from rate_of_closure.variation import (
    DurableEnsembleEvidence,
    build_simulation_ensemble_request,
    durable_ensemble_evidence_from_document,
    durable_ensemble_evidence_to_document,
)
from shared.python.swing_sim.variation import VariationPlan
from shared.python.swing_sim.variation.execution_metadata import plan_sha256

if TYPE_CHECKING:
    from rate_of_closure.simulation.records import SimulationConfig
    from rate_of_closure.variation import SimulationEnsembleSource

DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID = "rate-of-closure/durable-ensemble-request"
DURABLE_ENSEMBLE_JOB_SCHEMA_ID = "rate-of-closure/durable-ensemble-job"
DURABLE_ENSEMBLE_SCOPE = "passive-double-pendulum-global-perturbations/v1"
DURABLE_ENSEMBLE_AUTHORITY_SCHEMA_VERSION = 1
DURABLE_ENSEMBLE_REQUEST_SCHEMA_VERSION = 2

JobStatus = Literal["queued", "running", "completed", "cancelled", "failed"]
_REQUEST_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "scope",
        "request_id",
        "archive_id",
        "base",
        "plan",
        "plan_sha256",
        "chunk_size",
    }
)
_MAX_CHUNK_SIZE = 4096
_MAX_RUNS = 100_000
_JOB_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "job_id",
        "request_id",
        "archive_id",
        "status",
        "completed_trials",
        "total_trials",
        "cancel_requested",
        "evidence",
        "error",
    }
)


def _bounded_integer(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [1, {maximum}]")
    return value


def _plan(value: object) -> VariationPlan:
    if not isinstance(value, dict):
        raise TypeError("plan must be a JSON object")
    result = VariationPlan.from_json_dict(value)
    if result.to_json_dict() != value:
        raise ValueError("plan must use the canonical variation-plan/v2 wire")
    if result.mode != "swing":
        raise ValueError("plan mode must be swing")
    _bounded_integer(result.n_runs, "plan n_runs", _MAX_RUNS)
    if any(spec.time_window_s is not None or spec.point_ids for spec in result.noise):
        raise ValueError("scope supports only global perturbations")
    return result


@dataclass(frozen=True, slots=True)
class _DurableBaseRequest:
    morris: MorrisBaseRequest
    clubhead_speed_mph: float
    contact_mode: str

    def simulation_config(self) -> SimulationConfig:
        from rate_of_closure.simulation.contact import ContactMode

        base = self.morris.simulation_config()
        scenario = replace(base.scenario, clubhead_speed_mph=self.clubhead_speed_mph)
        return replace(
            base, scenario=scenario, contact_mode=ContactMode(self.contact_mode)
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            **dict(self.morris.values),
            "clubhead_speed_mph": self.clubhead_speed_mph,
            "contact_mode": self.contact_mode,
        }


def _finite_positive(value: object, name: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _parse_base(value: object) -> _DurableBaseRequest:
    if not isinstance(value, dict):
        raise TypeError("durable ensemble base must be a JSON object")
    item = dict(value)
    if "clubhead_speed_mph" not in item or "contact_mode" not in item:
        raise ValueError("durable ensemble base fields mismatch")
    contact_mode = item["contact_mode"]
    if contact_mode not in {"delivery_inspection", "fixed_ball_contact"}:
        raise ValueError("durable ensemble base contact_mode is unsupported")
    speed = item.pop("clubhead_speed_mph")
    item.pop("contact_mode")
    return _DurableBaseRequest(
        parse_morris_base_request(item),
        _finite_positive(speed, "base clubhead_speed_mph"),
        str(contact_mode),
    )


@dataclass(frozen=True, slots=True)
class DurableEnsembleAuthorityRequest:
    """Validated request whose source remains lazy and deterministic."""

    request_id: str
    archive_id: str
    base: _DurableBaseRequest
    plan: VariationPlan
    chunk_size: int

    def source(self) -> SimulationEnsembleSource:
        """Build the bounded source without materializing trials or configs."""
        return build_simulation_ensemble_request(
            self.plan, self.base.simulation_config()
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialize the exact path-free authority request."""
        return {
            "schema_id": DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID,
            "schema_version": DURABLE_ENSEMBLE_REQUEST_SCHEMA_VERSION,
            "scope": DURABLE_ENSEMBLE_SCOPE,
            "request_id": self.request_id,
            "archive_id": self.archive_id,
            "base": self.base.to_json_dict(),
            "plan": self.plan.to_json_dict(),
            "plan_sha256": plan_sha256(self.plan),
            "chunk_size": self.chunk_size,
        }


def parse_durable_ensemble_request(
    value: object,
) -> DurableEnsembleAuthorityRequest:
    """Parse one exact request and exercise its bounded source validation."""
    item = exact_mapping(value, _REQUEST_FIELDS, "durable ensemble request")
    if item["schema_id"] != DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID:
        raise ValueError("durable ensemble request schema_id is unsupported")
    if item["schema_version"] != DURABLE_ENSEMBLE_REQUEST_SCHEMA_VERSION:
        raise ValueError("durable ensemble request schema_version is unsupported")
    if item["scope"] != DURABLE_ENSEMBLE_SCOPE:
        raise ValueError("durable ensemble request scope is unsupported")
    request = DurableEnsembleAuthorityRequest(
        stable_id(item["request_id"], "request_id"),
        stable_id(item["archive_id"], "archive_id"),
        _parse_base(item["base"]),
        _plan(item["plan"]),
        _bounded_integer(item["chunk_size"], "chunk_size", _MAX_CHUNK_SIZE),
    )
    if item["plan_sha256"] != plan_sha256(request.plan):
        raise ValueError("durable ensemble request plan digest mismatch")
    request.source()
    return request


def durable_ensemble_request_document(
    request_id: str,
    archive_id: str,
    plan: VariationPlan,
    config: SimulationConfig,
    *,
    chunk_size: int,
) -> dict[str, object]:
    """Author a request only for the exactly represented passive subset."""
    base = _DurableBaseRequest(
        parse_morris_base_request(base_document(config)),
        float(config.scenario.clubhead_speed_mph),
        config.contact_mode.value,
    )
    if base.simulation_config() != config:
        raise ValueError("config differs from pinned passive authority semantics")
    request = DurableEnsembleAuthorityRequest(
        stable_id(request_id, "request_id"),
        stable_id(archive_id, "archive_id"),
        base,
        plan,
        _bounded_integer(chunk_size, "chunk_size", _MAX_CHUNK_SIZE),
    )
    return parse_durable_ensemble_request(request.to_json_dict()).to_json_dict()


@dataclass(frozen=True, slots=True)
class DurableEnsembleJobEnvelope:
    """Path-free incremental lifecycle snapshot returned to either client."""

    job_id: str
    request_id: str
    archive_id: str
    status: JobStatus
    completed_trials: int
    total_trials: int
    cancel_requested: bool
    evidence: DurableEnsembleEvidence | None
    error: str | None

    def __post_init__(self) -> None:
        stable_id(self.job_id, "job_id")
        stable_id(self.request_id, "request_id")
        stable_id(self.archive_id, "archive_id")
        if self.status not in {"queued", "running", "completed", "cancelled", "failed"}:
            raise ValueError("job status is unsupported")
        _bounded_integer(self.total_trials, "total_trials", _MAX_RUNS)
        if (
            type(self.completed_trials) is not int
            or not 0 <= self.completed_trials <= self.total_trials
        ):
            raise ValueError("completed_trials is outside the job bounds")
        if type(self.cancel_requested) is not bool:
            raise TypeError("cancel_requested must be boolean")
        if self.error is not None and (
            not isinstance(self.error, str) or not self.error
        ):
            raise ValueError("error must be null or nonempty text")
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("error availability does not match job status")
        if (
            self.evidence is not None
            and self.evidence.archive.analyzed_trial_count != self.completed_trials
        ):
            raise ValueError("evidence prefix does not match job progress")
        if self.status == "completed" and self.completed_trials != self.total_trials:
            raise ValueError("completed job has incomplete progress")

    def to_json_dict(self) -> dict[str, Any]:
        """Return the exact client document without exposing server paths."""
        return {
            "schema_id": DURABLE_ENSEMBLE_JOB_SCHEMA_ID,
            "schema_version": DURABLE_ENSEMBLE_AUTHORITY_SCHEMA_VERSION,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "archive_id": self.archive_id,
            "status": self.status,
            "completed_trials": self.completed_trials,
            "total_trials": self.total_trials,
            "cancel_requested": self.cancel_requested,
            "evidence": (
                durable_ensemble_evidence_to_document(self.evidence)
                if self.evidence is not None
                else None
            ),
            "error": self.error,
        }


def parse_durable_ensemble_job(value: object) -> DurableEnsembleJobEnvelope:
    """Parse one exact path-free lifecycle response from the authority."""
    item = exact_mapping(value, _JOB_FIELDS, "durable ensemble job")
    if item["schema_id"] != DURABLE_ENSEMBLE_JOB_SCHEMA_ID:
        raise ValueError("durable ensemble job schema_id is unsupported")
    if item["schema_version"] != DURABLE_ENSEMBLE_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("durable ensemble job schema_version is unsupported")
    status = item["status"]
    if status not in {"queued", "running", "completed", "cancelled", "failed"}:
        raise ValueError("durable ensemble job status is unsupported")
    evidence_value = item["evidence"]
    evidence = (
        None
        if evidence_value is None
        else durable_ensemble_evidence_from_document(evidence_value)
    )
    error = item["error"]
    if error is not None and not isinstance(error, str):
        raise TypeError("durable ensemble job error must be text or null")
    return DurableEnsembleJobEnvelope(
        stable_id(item["job_id"], "job_id"),
        stable_id(item["request_id"], "request_id"),
        stable_id(item["archive_id"], "archive_id"),
        status,
        item["completed_trials"],
        item["total_trials"],
        item["cancel_requested"],
        evidence,
        error,
    )


__all__ = [
    "DURABLE_ENSEMBLE_AUTHORITY_SCHEMA_VERSION",
    "DURABLE_ENSEMBLE_JOB_SCHEMA_ID",
    "DURABLE_ENSEMBLE_REQUEST_SCHEMA_ID",
    "DURABLE_ENSEMBLE_REQUEST_SCHEMA_VERSION",
    "DURABLE_ENSEMBLE_SCOPE",
    "DurableEnsembleAuthorityRequest",
    "DurableEnsembleJobEnvelope",
    "durable_ensemble_request_document",
    "parse_durable_ensemble_job",
    "parse_durable_ensemble_request",
]
