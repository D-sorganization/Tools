"""Bounded asynchronous lifecycle for authority-owned durable ensembles."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from rate_of_closure.variation import DurableEnsembleEvidence
from shared.python.swing_sim.solver.solve import CancelledError

from .contracts import (
    DurableEnsembleAuthorityRequest,
    DurableEnsembleJobEnvelope,
    JobStatus,
)
from .service import EvidenceSink

_LOGGER = logging.getLogger(__name__)
_TERMINAL = frozenset({"completed", "cancelled", "failed"})


class DurableEnsembleExecutionService(Protocol):
    """Execution and inspection surface injected into the registry."""

    def execute(
        self,
        request: DurableEnsembleAuthorityRequest,
        cancel: threading.Event,
        progress: EvidenceSink,
    ) -> DurableEnsembleEvidence: ...

    def inspect(
        self, request: DurableEnsembleAuthorityRequest
    ) -> DurableEnsembleEvidence: ...


@dataclass(frozen=True, slots=True)
class DurableEnsembleRegistryOptions:
    """Resource and retention limits for one local authority."""

    max_active_jobs: int = 2
    max_body_bytes: int = 1_000_000
    terminal_ttl_s: float = 900.0
    max_retained_jobs: int = 128

    def __post_init__(self) -> None:
        if not 1 <= self.max_active_jobs <= 8:
            raise ValueError("max_active_jobs must be within [1, 8]")
        if self.max_body_bytes < 1024 or self.terminal_ttl_s <= 0.0:
            raise ValueError("body and retention limits must be positive")
        if self.max_retained_jobs < self.max_active_jobs:
            raise ValueError("retained job limit must cover active jobs")


@dataclass(slots=True)
class _Job:
    job_id: str
    request: DurableEnsembleAuthorityRequest
    status: JobStatus = "queued"
    cancel_requested: bool = False
    evidence: DurableEnsembleEvidence | None = None
    error: str | None = None
    terminal_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = None


_DEFAULT_OPTIONS = DurableEnsembleRegistryOptions()


class DurableEnsembleJobRegistry:
    """Lock-linearized jobs with one active writer allowed per archive."""

    def __init__(
        self,
        service: DurableEnsembleExecutionService,
        options: DurableEnsembleRegistryOptions = _DEFAULT_OPTIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hasattr(service, "execute") or not hasattr(service, "inspect"):
            raise TypeError("service must support execute and inspect")
        if not isinstance(options, DurableEnsembleRegistryOptions):
            raise TypeError("options must be DurableEnsembleRegistryOptions")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._service = service
        self._options = options
        self.max_body_bytes = options.max_body_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=options.max_active_jobs,
            thread_name_prefix="rate-durable-ensemble",
        )

    def create(
        self, request: DurableEnsembleAuthorityRequest
    ) -> DurableEnsembleJobEnvelope:
        """Submit a new or resumed job after enforcing writer capacity."""
        if not isinstance(request, DurableEnsembleAuthorityRequest):
            raise TypeError("request must be a DurableEnsembleAuthorityRequest")
        with self._lock:
            self._prune_locked()
            active = [job for job in self._jobs.values() if job.status not in _TERMINAL]
            if any(job.request.archive_id == request.archive_id for job in active):
                raise FileExistsError("archive already has an active writer")
            if len(active) >= self._options.max_active_jobs:
                raise OverflowError("durable ensemble authority is at capacity")
            job = _Job(str(uuid.uuid4()), request)
            self._jobs[job.job_id] = job
            job.future = self._pool.submit(self._run, job.job_id)
            return self._envelope(job)

    def status(self, job_id: str) -> DurableEnsembleJobEnvelope:
        """Return a detached incremental snapshot."""
        with self._lock:
            self._prune_locked()
            return self._envelope(self._known_locked(job_id))

    def cancel(self, job_id: str) -> DurableEnsembleJobEnvelope:
        """Request cancellation while retaining every committed prefix."""
        with self._lock:
            job = self._known_locked(job_id)
            if job.status in _TERMINAL:
                return self._envelope(job)
            job.cancel_requested = True
            job.cancel.set()
            if (
                job.status == "queued"
                and job.future is not None
                and job.future.cancel()
            ):
                self._terminal_locked(job, "cancelled")
            return self._envelope(job)

    def close(self) -> None:
        """Cancel active work and release the owned coordinator threads."""
        with self._lock:
            for job in self._jobs.values():
                if job.status not in _TERMINAL:
                    job.cancel_requested = True
                    job.cancel.set()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                self._terminal_locked(job, "cancelled")
                return
            job.status = "running"
        try:
            result = self._service.execute(
                job.request,
                job.cancel,
                lambda evidence: self._progress(job_id, evidence),
            )
        except CancelledError:
            self._finish_cancelled(job_id)
        except Exception:
            _LOGGER.exception("durable ensemble authority job failed")
            with self._lock:
                job = self._jobs[job_id]
                job.error = "durable ensemble execution failed"
                self._terminal_locked(job, "failed")
        else:
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    self._terminal_locked(job, "cancelled")
                else:
                    job.evidence = result
                    self._terminal_locked(job, "completed")

    def _finish_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
        try:
            evidence = self._service.inspect(job.request)
        except Exception:
            _LOGGER.exception("cancelled archive inspection failed")
            evidence = None
        with self._lock:
            job = self._jobs[job_id]
            job.evidence = evidence
            self._terminal_locked(job, "cancelled")

    def _progress(self, job_id: str, evidence: DurableEnsembleEvidence) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if (
                job.status == "running"
                and evidence.archive.trial_count == job.request.plan.n_runs
            ):
                job.evidence = evidence

    def _terminal_locked(self, job: _Job, status: JobStatus) -> None:
        job.status = status
        job.terminal_at = self._clock()

    def _known_locked(self, job_id: str) -> _Job:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return self._jobs[job_id]

    def _prune_locked(self) -> None:
        cutoff = self._clock() - self._options.terminal_ttl_s
        expired = [
            key
            for key, job in self._jobs.items()
            if job.terminal_at is not None and job.terminal_at < cutoff
        ]
        for key in expired:
            del self._jobs[key]
        terminal = sorted(
            (
                (job.terminal_at or float("inf"), key)
                for key, job in self._jobs.items()
                if job.terminal_at is not None
            )
        )
        excess = max(0, len(self._jobs) - self._options.max_retained_jobs)
        for _finished, key in terminal[:excess]:
            del self._jobs[key]

    @staticmethod
    def _envelope(job: _Job) -> DurableEnsembleJobEnvelope:
        evidence = job.evidence
        completed = evidence.archive.analyzed_trial_count if evidence else 0
        return DurableEnsembleJobEnvelope(
            job.job_id,
            job.request.request_id,
            job.request.archive_id,
            job.status,
            completed,
            job.request.plan.n_runs,
            job.cancel_requested,
            evidence,
            job.error,
        )


__all__ = [
    "DurableEnsembleExecutionService",
    "DurableEnsembleJobRegistry",
    "DurableEnsembleRegistryOptions",
]
