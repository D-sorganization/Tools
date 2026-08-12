"""Sequential background transport workers for the Morris PyQt workflow."""

from __future__ import annotations

import threading
from typing import Protocol

from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.application.morris._response_types import (
    MorrisCapability,
    MorrisResponseJob,
)
from rate_of_closure.application.morris.contracts import MorrisAuthorityRequest


class MorrisAuthorityPort(Protocol):
    """Narrow client surface used by the Qt workflow."""

    def capability(self) -> MorrisCapability: ...

    def create(self, request: MorrisAuthorityRequest) -> MorrisResponseJob: ...

    def status(self, job_id: str) -> MorrisResponseJob: ...

    def cancel(self, job_id: str) -> MorrisResponseJob: ...


class MorrisCapabilityWorker(QThread):
    """Perform one capability handshake without blocking Qt."""

    available = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client: MorrisAuthorityPort) -> None:
        super().__init__()
        self._client = client

    def run(self) -> None:
        """Fetch one strict capability document."""
        try:
            self.available.emit(self._client.capability())
        except Exception:  # transport boundary must become safe UI state
            self.failed.emit("Morris authority capability check failed")


class MorrisRunWorker(QThread):
    """Create, poll, and optionally cancel one authority-owned job."""

    jobChanged = pyqtSignal(int, object)  # noqa: N815 - Qt convention
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        client: MorrisAuthorityPort,
        request: MorrisAuthorityRequest,
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
        """Request one idempotent cancellation on the worker thread."""
        self._cancel_requested.set()

    def run(self) -> None:
        """Execute a strictly sequential create/poll/cancel lifecycle."""
        try:
            job = self._client.create(self._request)
            _validate_identity(job, self._request.request_id)
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
                _validate_identity(job, self._request.request_id, job_id)
                self.jobChanged.emit(self._generation, job)
        except Exception:  # strict client already sanitizes transport details
            self.failed.emit(
                self._generation,
                "Morris authority request failed; verify the local authority "
                "and inputs.",
            )


def _validate_identity(
    job: MorrisResponseJob,
    request_id: str,
    job_id: str | None = None,
) -> None:
    """Fail closed if an authority response crosses job/request identity."""
    if job.request_id != request_id or (job_id is not None and job.job_id != job_id):
        raise RuntimeError("Morris authority response identity changed")


__all__ = [
    "MorrisAuthorityPort",
    "MorrisCapabilityWorker",
    "MorrisRunWorker",
]
