"""Sequential background transport for durable ensemble authority jobs."""

from __future__ import annotations

import threading
from typing import Protocol

from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.application.durable_ensemble.client import (
    DurableEnsembleCapability,
)
from rate_of_closure.application.durable_ensemble.contracts import (
    DurableEnsembleAuthorityRequest,
    DurableEnsembleJobEnvelope,
)


class DurableEnsembleAuthorityPort(Protocol):
    """Narrow client surface consumed by the Qt workflow."""

    def capability(self) -> DurableEnsembleCapability: ...

    def create(
        self, request: DurableEnsembleAuthorityRequest
    ) -> DurableEnsembleJobEnvelope: ...

    def status(self, job_id: str) -> DurableEnsembleJobEnvelope: ...

    def cancel(self, job_id: str) -> DurableEnsembleJobEnvelope: ...


class DurableEnsembleCapabilityWorker(QThread):
    """Probe authority availability without blocking the event loop."""

    available = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client: DurableEnsembleAuthorityPort) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:
        try:
            self.available.emit(self._client.capability())
        except Exception:
            self.failed.emit("Durable ensemble authority capability check failed")


class DurableEnsembleRunWorker(QThread):
    """Create/resume, poll, and cooperatively cancel one durable job."""

    jobChanged = pyqtSignal(int, object)  # noqa: N815 - Qt convention
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        client: DurableEnsembleAuthorityPort,
        request: DurableEnsembleAuthorityRequest,
        generation: int,
        poll_interval_ms: int,
    ) -> None:
        super().__init__()
        if poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be positive")
        self._client = client
        self._request = request
        self._generation = generation
        self._poll_interval_ms = poll_interval_ms
        self._cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        """Request one idempotent cancellation on this worker thread."""
        self._cancel_requested.set()

    def run(self) -> None:
        """Execute one identity-bound sequential transport lifecycle."""
        try:
            job = self._client.create(self._request)
            self._validate(job)
            self.jobChanged.emit(self._generation, job)
            job_id = job.job_id
            cancellation_sent = False
            while job.status not in {"completed", "cancelled", "failed"}:
                if self._cancel_requested.is_set() and not cancellation_sent:
                    job = self._client.cancel(job_id)
                    cancellation_sent = True
                else:
                    self.msleep(self._poll_interval_ms)
                    job = self._client.status(job_id)
                self._validate(job, job_id)
                self.jobChanged.emit(self._generation, job)
        except Exception:
            self.failed.emit(
                self._generation,
                "Durable ensemble request failed; verify the local authority, "
                "archive, and global perturbation plan.",
            )

    def _validate(
        self, job: DurableEnsembleJobEnvelope, job_id: str | None = None
    ) -> None:
        request = self._request
        if (
            job.request_id != request.request_id
            or job.archive_id != request.archive_id
            or job.total_trials != request.plan.n_runs
            or (job_id is not None and job.job_id != job_id)
        ):
            raise RuntimeError("durable ensemble response identity changed")


__all__ = [
    "DurableEnsembleAuthorityPort",
    "DurableEnsembleCapabilityWorker",
    "DurableEnsembleRunWorker",
]
