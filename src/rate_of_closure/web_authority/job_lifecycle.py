"""Internal lifecycle types for the regional-ground authority manager."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobFailure,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationHooks,
)

PUBLIC_JOB_EXPORTS = [
    "AUTHORITY_JOB_STATUS_SCHEMA_VERSION",
    "AuthorityExecutionUnavailable",
    "AuthorityJobConflict",
    "AuthorityJobFailure",
    "AuthorityJobManager",
    "AuthorityJobResultUnavailable",
    "AuthorityJobSnapshot",
    "AuthorityJobStatus",
]


class AuthorityExecutionUnavailable(RuntimeError):
    """Raised when no qualified physical runner is attached."""


class AuthorityJobConflict(RuntimeError):
    """Raised when submission would violate the one-active-job invariant."""


class AuthorityJobResultUnavailable(RuntimeError):
    """Raised unless a complete validated result has been published."""


AuthorityJobRunner = Callable[
    [RegionalGroundExecutionJob, GroundRegionalVariationHooks],
    RegionalGroundExecutionResult,
]


@dataclass(slots=True)
class JobRecord:
    """Mutable manager-owned state protected by the manager condition."""

    job: RegionalGroundExecutionJob
    cancellation: threading.Event
    status: AuthorityJobStatus = AuthorityJobStatus.QUEUED
    completed: int = 0
    result: RegionalGroundExecutionResult | None = None
    failure: AuthorityJobFailure | None = None


TERMINAL_JOB_STATUSES = frozenset(
    {
        AuthorityJobStatus.SUCCEEDED,
        AuthorityJobStatus.FAILED,
        AuthorityJobStatus.CANCELLED,
    }
)


__all__ = [
    "AuthorityExecutionUnavailable",
    "AuthorityJobConflict",
    "AuthorityJobResultUnavailable",
    "AuthorityJobRunner",
    "JobRecord",
    "TERMINAL_JOB_STATUSES",
]
