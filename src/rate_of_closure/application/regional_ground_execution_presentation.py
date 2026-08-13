"""UI-neutral non-executing regional-ground presentation contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypedDict

from rate_of_closure.application.regional_ground_execution_job import (
    REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION,
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION,
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.capability import AuthorityCapability


class RegionalGroundPresentationState(StrEnum):
    """Exact states rendered without invoking a submitter."""

    IDLE = "idle"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class RegionalGroundExecutionSummaryWire(TypedDict):
    """Compact job evidence rendered identically by both clients."""

    schema_version: str
    model_id: str
    model_version: str
    producer: str
    producer_version: str
    source_revision: str
    input_sha256: str


@dataclass(frozen=True, slots=True)
class RegionalGroundExecutionSummary:
    """Validated compact identity and provenance summary."""

    schema_version: str
    model_id: str
    model_version: str
    producer: str
    producer_version: str
    source_revision: str
    input_sha256: str

    @classmethod
    def from_job(
        cls, job: RegionalGroundExecutionJob
    ) -> RegionalGroundExecutionSummary:
        """Project exact fields from a validated job without reinterpreting them."""
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("job must be an exact RegionalGroundExecutionJob")
        job.__post_init__()
        return cls(
            REGIONAL_GROUND_EXECUTION_JOB_SCHEMA_VERSION,
            job.flight.model_id,
            job.flight.model_version,
            job.provenance.producer,
            job.provenance.producer_version,
            job.provenance.source_revision,
            job.input_sha256,
        )

    def to_wire(self) -> RegionalGroundExecutionSummaryWire:
        """Return the exact matched client projection."""
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "source_revision": self.source_revision,
            "input_sha256": self.input_sha256,
        }


@dataclass(frozen=True, slots=True)
class RegionalGroundExecutionPresentation:
    """Immutable display-only state derived from exact controller evidence."""

    summary: RegionalGroundExecutionSummary
    job_sha256: str
    total: int
    execution_enabled: bool
    disabled_reason_code: str
    disabled_detail: str
    state: RegionalGroundPresentationState = RegionalGroundPresentationState.IDLE
    completed: int = 0
    failure_stage: str | None = None
    result_schema_version: str | None = None
    result_sha256: str | None = None

    @classmethod
    def initial(
        cls,
        job: RegionalGroundExecutionJob,
        capability: AuthorityCapability,
    ) -> RegionalGroundExecutionPresentation:
        """Build the false-capability presentation without a submitter."""
        if type(capability) is not AuthorityCapability:
            raise TypeError("capability must be exact")
        wire = capability.to_wire()
        return cls(
            RegionalGroundExecutionSummary.from_job(job),
            job.job_sha256,
            job.execution_options.max_trials,
            wire["regional_ground_execution"],
            wire["reason_code"],
            wire["detail"],
        )

    def with_progress(
        self, progress: GroundRegionalVariationProgress
    ) -> RegionalGroundExecutionPresentation:
        """Present exact monotonic progress from the existing controller."""
        if type(progress) is not GroundRegionalVariationProgress:
            raise TypeError("progress must be exact")
        if progress.total != self.total or progress.completed < self.completed:
            raise ValueError("progress must be monotonic and match the job")
        return replace(
            self,
            state=RegionalGroundPresentationState.RUNNING,
            completed=progress.completed,
        )

    def with_cancel_requested(self) -> RegionalGroundExecutionPresentation:
        """Present a cancellation request without invoking a controller."""
        return replace(self, state=RegionalGroundPresentationState.CANCEL_REQUESTED)

    def with_cancelled(
        self, terminal: GroundRegionalVariationCancelled
    ) -> RegionalGroundExecutionPresentation:
        """Present an exact typed cancellation terminal."""
        self._terminal_counts(terminal.completed, terminal.total)
        return replace(
            self,
            state=RegionalGroundPresentationState.CANCELLED,
            completed=terminal.completed,
        )

    def with_failure(
        self, terminal: GroundRegionalVariationFailed
    ) -> RegionalGroundExecutionPresentation:
        """Present only the stable failure stage and counts, never cause text."""
        self._terminal_counts(terminal.completed, terminal.total)
        return replace(
            self,
            state=RegionalGroundPresentationState.FAILED,
            completed=terminal.completed,
            failure_stage=terminal.stage.value,
        )

    def with_result(
        self, result: RegionalGroundExecutionResult
    ) -> RegionalGroundExecutionPresentation:
        """Present only a complete result carrying this presentation's identity."""
        if type(result) is not RegionalGroundExecutionResult:
            raise TypeError("result must be exact")
        result.__post_init__()
        if result.job_sha256 != self.job_sha256:
            raise ValueError("result job_sha256 must match presentation")
        return replace(
            self,
            state=RegionalGroundPresentationState.SUCCEEDED,
            completed=self.total,
            result_schema_version=REGIONAL_GROUND_EXECUTION_RESULT_SCHEMA_VERSION,
            result_sha256=result.dataset_sha256,
        )

    def _terminal_counts(self, completed: int, total: int) -> None:
        """Fail closed on stale or mismatched controller terminals."""
        if total != self.total or not self.completed <= completed <= total:
            raise ValueError("terminal counts must be monotonic and match the job")

    def to_wire(self) -> dict[str, object]:
        """Return the exact matched display record."""
        return {
            "summary": self.summary.to_wire(),
            "execution_enabled": self.execution_enabled,
            "disabled_reason_code": self.disabled_reason_code,
            "disabled_detail": self.disabled_detail,
            "status": self.state.value,
            "completed": self.completed,
            "total": self.total,
            "failure_stage": self.failure_stage,
            "result_schema_version": self.result_schema_version,
            "result_sha256": self.result_sha256,
        }


__all__ = [
    "RegionalGroundExecutionPresentation",
    "RegionalGroundExecutionSummary",
    "RegionalGroundExecutionSummaryWire",
    "RegionalGroundPresentationState",
]
