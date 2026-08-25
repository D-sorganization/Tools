"""Mountable FastAPI router and bounded in-memory Morris job registry."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rate_of_closure.application._strict_http_json import (
    StrictHttpFailure,
    strict_json_document,
)
from shared.python.swing_sim.variation import (
    CancelledError,
    MorrisObservationArchive,
    analyze_morris,
    morris_design_sha256,
)

from .contracts import (
    JobStatus,
    MorrisAuthorityRequest,
    MorrisJobEnvelope,
    parse_morris_request,
)
from .response_contract import parse_morris_report
from .service import (
    MorrisExecutionService,
    MorrisServiceResult,
    morris_request_sha256,
)

_LOGGER = logging.getLogger(__name__)
_TERMINAL = frozenset({"completed", "cancelled", "failed"})


@dataclass
class _Job:
    job_id: str
    request: MorrisAuthorityRequest
    status: JobStatus = "queued"
    completed_samples: int = 0
    cancel_requested: bool = False
    report: dict[str, Any] | None = None
    observations: MorrisObservationArchive | None = None
    error: dict[str, str] | None = None
    terminal_at: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    future: Future[None] | None = None


@dataclass(frozen=True)
class MorrisRegistryOptions:
    """Explicit resource and retention limits for one registry."""

    max_active_jobs: int = 2
    max_body_bytes: int = 64_000
    terminal_ttl_s: float = 900.0
    max_retained_jobs: int = 128
    max_total_study_workers: int = 64
    max_retained_observation_cells: int = 2_000_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_active_jobs <= 32:
            raise ValueError("max_active_jobs must be within [1, 32]")
        if self.max_body_bytes < 1_024 or self.terminal_ttl_s <= 0.0:
            raise ValueError("body and retention limits must be positive")
        if self.max_retained_jobs < self.max_active_jobs:
            raise ValueError("retained job limit must cover active jobs")
        if self.max_total_study_workers < 1:
            raise ValueError("study worker budget must be positive")
        if self.max_retained_observation_cells < 1:
            raise ValueError("observation cell budget must be positive")


_DEFAULT_REGISTRY_OPTIONS = MorrisRegistryOptions()


class MorrisJobRegistry:
    """Own a bounded executor and lock-linearized ephemeral job lifecycle."""

    def __init__(
        self,
        service: MorrisExecutionService,
        options: MorrisRegistryOptions = _DEFAULT_REGISTRY_OPTIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hasattr(service, "execute"):
            raise TypeError("service must implement execute")
        if not isinstance(options, MorrisRegistryOptions) or not callable(clock):
            raise TypeError("registry options and monotonic clock are required")
        self._service = service
        self._options = options
        self.max_body_bytes = options.max_body_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=options.max_active_jobs, thread_name_prefix="rate-morris"
        )

    def create(self, request: MorrisAuthorityRequest) -> MorrisJobEnvelope:
        """Register and submit one job, rejecting saturated capacity."""
        with self._lock:
            self._prune_locked()
            active = sum(job.status not in _TERMINAL for job in self._jobs.values())
            workers = sum(
                job.request.worker_count
                for job in self._jobs.values()
                if job.status not in _TERMINAL
            )
            saturated = active >= self._options.max_active_jobs
            worker_limit = (
                workers + request.worker_count > self._options.max_total_study_workers
            )
            if saturated or worker_limit:
                raise OverflowError("Morris authority is at capacity")
            job_id = str(uuid.uuid4())
            job = _Job(job_id, request)
            self._jobs[job_id] = job
            job.future = self._pool.submit(self._run, job_id)
            return self._envelope(job)

    def status(self, job_id: str) -> MorrisJobEnvelope:
        """Return one detached status envelope or reject an unknown job."""
        with self._lock:
            self._prune_locked()
            return self._envelope(self._known_locked(job_id))

    def cancel(self, job_id: str) -> MorrisJobEnvelope:
        """Idempotently register cancellation without prematurely terminating work."""
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

    def observations(self, job_id: str) -> MorrisObservationArchive:
        """Return a completed job's immutable raw authority or fail closed."""
        with self._lock:
            self._prune_locked()
            job = self._known_locked(job_id)
            if job.status != "completed" or job.observations is None:
                raise RuntimeError("Morris observations are not available")
            return job.observations

    def close(self) -> None:
        """Cancel outstanding work and release owned executor threads."""
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

            def progress(done: int, total: int) -> None:
                self._progress(job_id, done, total)

            extended = getattr(self._service, "execute_with_observations", None)
            result = (
                extended(job.request, job.cancel, progress)
                if callable(extended)
                else self._service.execute(job.request, job.cancel, progress)
            )
            if isinstance(result, MorrisServiceResult):
                self._validate_extended_result(job, result)
        except CancelledError:
            with self._lock:
                self._terminal_locked(job, "cancelled")
        except Exception:
            _LOGGER.exception("Morris job failed: job_id=%s", job_id)
            with self._lock:
                self._finish_failure_locked(job)
        else:
            with self._lock:
                self._finish_success_locked(job, result)

    def _finish_failure_locked(self, job: _Job) -> None:
        if job.cancel_requested:
            self._terminal_locked(job, "cancelled")
            return
        job.error = {
            "code": "execution_failed",
            "message": "Morris execution failed",
        }
        self._terminal_locked(job, "failed")

    def _finish_success_locked(
        self, job: _Job, result: dict[str, Any] | MorrisServiceResult
    ) -> None:
        if job.cancel_requested:
            self._terminal_locked(job, "cancelled")
            return
        job.completed_samples = job.request.total_samples
        if isinstance(result, MorrisServiceResult):
            job.report = dict(result.report)
            job.observations = result.observations
            self._enforce_observation_budget_locked(job)
        else:
            job.report = result
        self._terminal_locked(job, "completed")

    @staticmethod
    def _validate_extended_result(job: _Job, result: MorrisServiceResult) -> None:
        """Verify result transport integrity, provenance, and metric invariants.

        This integrity check confirms that the extended service result matches
        the original asynchronous job request, that observations and design
        provenance are uncorrupted across the execution/thread transport
        boundary, and that all reported Morris metric invariants hold. This is
        a transport/pipeline integrity guard against state corruption and
        cross-job misattribution, not an independent mathematical verification
        of the underlying elementary-effects estimation algorithm.
        """
        request = job.request
        archive = result.observations
        design = request.design()
        report_design = result.report.get("design")
        expected_report_design = {
            "trajectories": request.trajectories,
            "levels": request.levels,
            "seed": request.seed,
            "total_samples": request.total_samples,
        }
        recomputed_report = analyze_morris(
            archive.observations, request.minimum_effects
        ).to_json_dict()
        if (
            archive.study_id != request.request_id
            or archive.design_sha256 != morris_design_sha256(design)
            or archive.provenance.get("request_sha256")
            != morris_request_sha256(request)
            or not isinstance(report_design, dict)
            or any(
                report_design.get(key) != value
                for key, value in expected_report_design.items()
            )
            or result.report != recomputed_report
        ):
            raise ValueError("Morris service result does not match its job request")
        parse_morris_report(result.report)

    def _enforce_observation_budget_locked(self, current: _Job) -> None:
        """Evict oldest raw authorities until the weighted cell budget fits."""
        assert current.observations is not None
        candidates = sorted(
            (
                (job.terminal_at or float("inf"), job)
                for job in self._jobs.values()
                if job is not current and job.observations is not None
            ),
            key=lambda item: item[0],
        )
        total = sum(
            job.observations.observation_cells
            for job in self._jobs.values()
            if job.observations is not None
        )
        for _terminal_at, job in candidates:
            if total <= self._options.max_retained_observation_cells:
                break
            assert job.observations is not None
            total -= job.observations.observation_cells
            job.observations = None
        if total > self._options.max_retained_observation_cells:
            current.observations = None

    def _progress(self, job_id: str, done: int, total: int) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "running" and total == job.request.total_samples:
                job.completed_samples = max(job.completed_samples, min(done, total))

    def _terminal_locked(self, job: _Job, status: JobStatus) -> None:
        job.status = status
        job.terminal_at = self._clock()
        if status != "completed":
            job.report = None
            job.observations = None

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
                (job.terminal_at, key)
                for key, job in self._jobs.items()
                if job.terminal_at is not None
            ),
            key=lambda item: item[0],
        )
        excess = max(0, len(self._jobs) - self._options.max_retained_jobs)
        for _time, key in terminal[:excess]:
            del self._jobs[key]

    @staticmethod
    def _envelope(job: _Job) -> MorrisJobEnvelope:
        return MorrisJobEnvelope(
            job.job_id,
            job.request.request_id,
            job.status,
            job.completed_samples,
            job.request.total_samples,
            job.cancel_requested,
            job.report,
            job.error,
        )


def _response(envelope: MorrisJobEnvelope, status: int = 200) -> JSONResponse:
    return JSONResponse(envelope.to_json_dict(), status_code=status)


def create_morris_router(registry: MorrisJobRegistry) -> APIRouter:
    """Create a mountable router without host, CORS, or global state changes."""
    if not isinstance(registry, MorrisJobRegistry):
        raise TypeError("registry must be a MorrisJobRegistry")
    router = APIRouter()

    @router.post("/morris/jobs")
    async def create_job(request: Request) -> JSONResponse:
        try:
            document = await strict_json_document(request, registry.max_body_bytes)
            parsed = parse_morris_request(document)
            return _response(registry.create(parsed), 202)
        except StrictHttpFailure as exc:
            return JSONResponse({"error": exc.message}, status_code=exc.status)
        except (TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except OverflowError:
            return JSONResponse(
                {"error": "Morris authority is at capacity"}, status_code=429
            )

    @router.get("/morris/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        try:
            return _response(registry.status(job_id))
        except KeyError:
            return JSONResponse({"error": "unknown Morris job"}, status_code=404)

    @router.delete("/morris/jobs/{job_id}")
    async def cancel_job(job_id: str) -> JSONResponse:
        try:
            envelope = registry.cancel(job_id)
            return _response(envelope, 200 if envelope.status in _TERMINAL else 202)
        except KeyError:
            return JSONResponse({"error": "unknown Morris job"}, status_code=404)

    return router


__all__ = ["MorrisJobRegistry", "MorrisRegistryOptions", "create_morris_router"]
