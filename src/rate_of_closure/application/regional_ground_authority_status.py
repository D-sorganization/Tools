"""Transport-neutral canonical regional-ground authority job status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast

from rate_of_closure.application._regional_ground_execution_job_values import digest
from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from shared.python.swing_sim.canonical_numeric_json import (
    MAX_CANONICAL_SAFE_INTEGER,
    canonical_numeric_json,
)
from shared.python.swing_sim.ground.strict_json import strict_json_object

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

AUTHORITY_JOB_STATUS_SCHEMA_VERSION: Final = (
    "rate-of-closure/regional-ground-authority-job-status/v1"
)
MAX_AUTHORITY_JOB_STATUS_BYTES: Final = 4_096

AuthorityFailureCode = Literal["execution_failed", "result_rejected"]
AuthorityFailureStage = Literal[
    "authority_restart",
    "cancellation_callback",
    "preflight",
    "executor",
    "validation",
    "progress_callback",
    "publication",
    "runner",
    "result_validation",
]
_FAILURE_CODES = frozenset({"execution_failed", "result_rejected"})
_FAILURE_STAGES = frozenset(
    {
        "authority_restart",
        "cancellation_callback",
        "preflight",
        "executor",
        "validation",
        "progress_callback",
        "publication",
        "runner",
        "result_validation",
    }
)
_FAILURE_FIELDS = frozenset({"code", "stage"})
_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "job_id",
        "job_sha256",
        "status",
        "completed",
        "total",
        "result_available",
        "failure",
    }
)


class AuthorityJobStatus(StrEnum):
    """Exact lifecycle states exposed by the local authority."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuthorityJobFailureWire(TypedDict):
    """Public failure record without raw exception text."""

    code: AuthorityFailureCode
    stage: AuthorityFailureStage


class AuthorityJobSnapshotWire(TypedDict):
    """Exact JSON-compatible job lifecycle projection."""

    schema_version: str
    job_id: str
    job_sha256: str
    status: str
    completed: int
    total: int
    result_available: bool
    failure: AuthorityJobFailureWire | None


def _bounded_integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= MAX_CANONICAL_SAFE_INTEGER:
        raise ValueError(f"{name} must lie within the cross-runtime safe range")
    return value


def _literal(value: object, allowed: frozenset[str], name: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True)
class AuthorityJobFailure:
    """Stable public failure identity with no internal exception detail."""

    code: AuthorityFailureCode
    stage: AuthorityFailureStage

    def __post_init__(self) -> None:
        _literal(self.code, _FAILURE_CODES, "failure code")
        _literal(self.stage, _FAILURE_STAGES, "failure stage")

    def to_wire(self) -> AuthorityJobFailureWire:
        """Return the exact bounded failure record."""
        self.__post_init__()
        return {"code": self.code, "stage": self.stage}


@dataclass(frozen=True, slots=True)
class AuthorityJobSnapshot:
    """Immutable validated point-in-time job status."""

    job_id: str
    job_sha256: str
    status: AuthorityJobStatus
    completed: int
    total: int
    result_available: bool = False
    failure: AuthorityJobFailure | None = None

    def __post_init__(self) -> None:
        stable_id(self.job_id, "job_id")
        digest(self.job_sha256, "job_sha256")
        if type(self.status) is not AuthorityJobStatus:
            raise TypeError("status must be an exact AuthorityJobStatus")
        _bounded_integer(self.completed, "completed", 0)
        _bounded_integer(self.total, "total", 1)
        if self.completed > self.total:
            raise ValueError("completed must not exceed total")
        if type(self.result_available) is not bool:
            raise TypeError("result_available must be a Boolean")
        if self.failure is not None and type(self.failure) is not AuthorityJobFailure:
            raise TypeError("failure must be an exact AuthorityJobFailure or None")
        self._validate_state_semantics()

    def _validate_state_semantics(self) -> None:
        if self.status is AuthorityJobStatus.QUEUED and self.completed != 0:
            raise ValueError("queued status must have zero completed")
        succeeded = self.status is AuthorityJobStatus.SUCCEEDED
        if self.result_available != succeeded or (
            succeeded and self.completed != self.total
        ):
            raise ValueError("succeeded status must expose a complete result")
        if (self.status is AuthorityJobStatus.FAILED) != (self.failure is not None):
            raise ValueError("failure is required only for failed status")

    def to_wire(self) -> AuthorityJobSnapshotWire:
        """Return the exact status wire projection."""
        self.__post_init__()
        return {
            "schema_version": AUTHORITY_JOB_STATUS_SCHEMA_VERSION,
            "job_id": self.job_id,
            "job_sha256": self.job_sha256,
            "status": self.status.value,
            "completed": self.completed,
            "total": self.total,
            "result_available": self.result_available,
            "failure": None if self.failure is None else self.failure.to_wire(),
        }


def _failure_from_wire(value: object) -> AuthorityJobFailure | None:
    if value is None:
        return None
    data = exact_mapping(value, _FAILURE_FIELDS, "failure")
    return AuthorityJobFailure(
        cast(
            AuthorityFailureCode, _literal(data["code"], _FAILURE_CODES, "failure code")
        ),
        cast(
            AuthorityFailureStage,
            _literal(data["stage"], _FAILURE_STAGES, "failure stage"),
        ),
    )


def _validate_expected_job(
    status: AuthorityJobSnapshot, expected_job: RegionalGroundExecutionJob
) -> None:
    if type(expected_job) is not RegionalGroundExecutionJob:
        raise TypeError("expected_job must be an exact RegionalGroundExecutionJob")
    expected_job.__post_init__()
    if status.job_id != expected_job.job_id:
        raise ValueError("job_id must match the expected execution job")
    if status.job_sha256 != expected_job.job_sha256:
        raise ValueError("job_sha256 must match the expected execution job")
    if status.total != expected_job.execution_options.max_trials:
        raise ValueError("total must match the expected execution job")


def regional_ground_authority_job_status_from_wire(
    value: object, expected_job: RegionalGroundExecutionJob | None = None
) -> AuthorityJobSnapshot:
    """Validate one exact status mapping and optionally bind it to its job."""
    data = exact_mapping(value, _STATUS_FIELDS, "regional-ground authority job status")
    if data["schema_version"] != AUTHORITY_JOB_STATUS_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    try:
        status_kind = AuthorityJobStatus(data["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid status") from exc
    status = AuthorityJobSnapshot(
        stable_id(data["job_id"], "job_id"),
        digest(data["job_sha256"], "job_sha256"),
        status_kind,
        _bounded_integer(data["completed"], "completed", 0),
        _bounded_integer(data["total"], "total", 1),
        data["result_available"],
        _failure_from_wire(data["failure"]),
    )
    if expected_job is not None:
        _validate_expected_job(status, expected_job)
    return status


def regional_ground_authority_job_status_from_json(
    text: str, expected_job: RegionalGroundExecutionJob | None = None
) -> AuthorityJobSnapshot:
    """Parse one bounded status and optionally bind it to its source job."""
    if type(text) is not str:
        raise TypeError("regional-ground authority job status JSON must be text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("job status must be valid UTF-8") from exc
    if len(encoded) > MAX_AUTHORITY_JOB_STATUS_BYTES:
        raise ValueError("job status exceeds maximum wire size")
    payload = strict_json_object(text)
    canonical_numeric_json(payload)
    return regional_ground_authority_job_status_from_wire(payload, expected_job)


def regional_ground_authority_job_status_to_json(
    status: AuthorityJobSnapshot, expected_job: RegionalGroundExecutionJob
) -> str:
    """Serialize one job-bound status with the shared canonical JSON policy."""
    if type(status) is not AuthorityJobSnapshot:
        raise TypeError("status must be an exact AuthorityJobSnapshot")
    status.__post_init__()
    _validate_expected_job(status, expected_job)
    text = str(canonical_numeric_json(status.to_wire()))
    if len(text.encode("utf-8")) > MAX_AUTHORITY_JOB_STATUS_BYTES:
        raise ValueError("job status exceeds maximum wire size")
    return text


__all__ = [
    "AUTHORITY_JOB_STATUS_SCHEMA_VERSION",
    "MAX_AUTHORITY_JOB_STATUS_BYTES",
    "AuthorityFailureCode",
    "AuthorityFailureStage",
    "AuthorityJobFailure",
    "AuthorityJobSnapshot",
    "AuthorityJobStatus",
    "regional_ground_authority_job_status_from_json",
    "regional_ground_authority_job_status_from_wire",
    "regional_ground_authority_job_status_to_json",
]
