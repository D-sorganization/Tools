"""Fail-closed startup reconciliation for durable authority records."""

from __future__ import annotations

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobFailure,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)

from .job_store import AuthorityJobStore, RetainedAuthorityJob

_TERMINAL = frozenset(
    {
        AuthorityJobStatus.SUCCEEDED,
        AuthorityJobStatus.FAILED,
        AuthorityJobStatus.CANCELLED,
    }
)


def recover_authority_jobs(
    store: AuthorityJobStore,
) -> tuple[RetainedAuthorityJob, ...]:
    """Make every interrupted record terminal without resuming execution."""
    loaded = store.load()
    reconciled: list[RetainedAuthorityJob] = []
    changed = False
    for record in loaded:
        if record.status.status in _TERMINAL:
            reconciled.append(record)
            continue
        changed = True
        if record.status.status is AuthorityJobStatus.CANCEL_REQUESTED:
            status = AuthorityJobSnapshot(
                job_id=record.job.job_id,
                job_sha256=record.job.job_sha256,
                status=AuthorityJobStatus.CANCELLED,
                completed=record.status.completed,
                total=record.status.total,
            )
            reconciled.append(RetainedAuthorityJob(record.job, status, None))
            continue
        status = AuthorityJobSnapshot(
            job_id=record.job.job_id,
            job_sha256=record.job.job_sha256,
            status=AuthorityJobStatus.FAILED,
            completed=record.status.completed,
            total=record.status.total,
            failure=AuthorityJobFailure("execution_failed", "authority_restart"),
        )
        reconciled.append(RetainedAuthorityJob(record.job, status, None))
    if len(reconciled) > store.max_retained_jobs:
        reconciled = reconciled[-store.max_retained_jobs :]
        changed = True
    recovered = tuple(reconciled)
    if changed:
        store.replace(recovered)
    return recovered


__all__ = ["recover_authority_jobs"]
