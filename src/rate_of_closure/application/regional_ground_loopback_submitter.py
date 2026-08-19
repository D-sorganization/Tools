"""UI-neutral client for a future qualified loopback execution authority."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Final

from rate_of_closure.application.regional_ground_authority_failures import (
    authority_failure_stage,
    fail_regional_ground_authority,
)
from rate_of_closure.application.regional_ground_authority_policy import (
    DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY,
    RegionalGroundAuthorityClientError,
    RegionalGroundAuthorityPollPolicy,
)
from rate_of_closure.application.regional_ground_authority_status import (
    MAX_AUTHORITY_JOB_STATUS_BYTES,
    AuthorityJobSnapshot,
    AuthorityJobStatus,
    regional_ground_authority_job_status_from_json,
)
from rate_of_closure.application.regional_ground_authority_transport import (
    AuthorityHttpResponse,
    LoopbackAuthorityHttpTransport,
    RegionalGroundAuthorityTransport,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
    RegionalGroundExecutionResult,
    regional_ground_execution_result_from_json,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.api import JOB_COLLECTION_PATH
from rate_of_closure.web_authority.capability import AuthorityCapability
from rate_of_closure.web_authority.runtime import AuthorityRuntime

_JSON_MEDIA_TYPE: Final = "application/json"
_TERMINAL = frozenset(
    {
        AuthorityJobStatus.SUCCEEDED,
        AuthorityJobStatus.FAILED,
        AuthorityJobStatus.CANCELLED,
    }
)


class LoopbackRegionalGroundSubmitter:
    """Submit, poll, cancel, and retrieve one exact authority job at a time."""

    def __init__(
        self,
        transport: RegionalGroundAuthorityTransport,
        *,
        policy: RegionalGroundAuthorityPollPolicy = (
            DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY
        ),
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        """Bind injected transport, clock, and bounded lifecycle policy."""
        if not callable(getattr(transport, "request", None)) or not callable(
            getattr(transport, "close", None)
        ):
            raise TypeError("transport must implement request and close")
        if type(policy) is not RegionalGroundAuthorityPollPolicy:
            raise TypeError("policy must be exact")
        if not callable(monotonic) or not callable(sleeper):
            raise TypeError("clock functions must be callable")
        self._transport = transport
        self._policy = policy
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._shutdown = threading.Event()
        self._condition = threading.Condition(threading.RLock())
        self._active_job: RegionalGroundExecutionJob | None = None
        self._closed = False

    def __call__(
        self, job: RegionalGroundExecutionJob, hooks: GroundRegionalVariationHooks
    ) -> RegionalGroundExecutionResult:
        """Execute the strict loopback lifecycle or raise one typed terminal."""
        self._begin(job)
        try:
            return self._execute(job, hooks)
        except (GroundRegionalVariationCancelled, GroundRegionalVariationFailed):
            raise
        except Exception as error:
            fail_regional_ground_authority(
                GroundRegionalVariationFailureStage.EXECUTOR, 0, job, error
            )
        finally:
            self._finish()

    def request_shutdown(self) -> None:
        """Request cooperative cancellation of current or next in-flight work."""
        self._shutdown.set()

    def close(self, *, timeout_s: float | None = None) -> None:
        """Cancel active work, wait a bound, close transport, and reject reuse."""
        timeout = self._policy.shutdown_timeout_s if timeout_s is None else timeout_s
        if type(timeout) not in (int, float) or timeout <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.request_shutdown()
        deadline = self._monotonic() + float(timeout)
        with self._condition:
            while self._active_job is not None:
                remaining = deadline - self._monotonic()
                if remaining <= 0.0 or not self._condition.wait(remaining):
                    raise TimeoutError(
                        "authority submitter did not stop before timeout"
                    )
            self._closed = True
        self._transport.close()

    def _begin(self, job: RegionalGroundExecutionJob) -> None:
        """Reserve the one active slot for an exact validated job."""
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("job must be an exact RegionalGroundExecutionJob")
        job.__post_init__()
        with self._condition:
            if self._closed:
                fail_regional_ground_authority(
                    GroundRegionalVariationFailureStage.EXECUTOR,
                    0,
                    job,
                    RegionalGroundAuthorityClientError("client_closed"),
                )
            if self._active_job is not None:
                raise RuntimeError("authority submitter is already active")
            self._active_job = job

    def _finish(self) -> None:
        """Release the active slot and wake a closing owner."""
        with self._condition:
            self._active_job = None
            self._condition.notify_all()

    def _execute(
        self, job: RegionalGroundExecutionJob, hooks: GroundRegionalVariationHooks
    ) -> RegionalGroundExecutionResult:
        """Run the accepted remote lifecycle with late-publication suppression."""
        if type(hooks) is not GroundRegionalVariationHooks:
            raise TypeError("hooks must be exact")
        if self._cancel_requested(hooks, job, 0):
            raise GroundRegionalVariationCancelled(0, job.execution_options.max_trials)
        job_path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
        response = self._request(
            "POST",
            JOB_COLLECTION_PATH,
            regional_ground_execution_job_to_json(job).encode("utf-8"),
            MAX_AUTHORITY_JOB_STATUS_BYTES,
            expected_status=202,
        )
        cancel_posted = False
        try:
            status = self._status(response, job)
            self._report(status, hooks, job)
            deadline = self._monotonic() + self._policy.poll_timeout_s
            interval = self._policy.initial_interval_s
            while status.status not in _TERMINAL:
                requested = self._shutdown.is_set() or self._cancel_requested(
                    hooks, job, status.completed
                )
                if requested and not cancel_posted:
                    status = self._status(
                        self._request(
                            "POST",
                            f"{job_path}/cancel",
                            None,
                            MAX_AUTHORITY_JOB_STATUS_BYTES,
                            expected_status=202,
                        ),
                        job,
                    )
                    cancel_posted = True
                remaining = deadline - self._monotonic()
                if remaining <= 0.0:
                    raise RegionalGroundAuthorityClientError("poll_timeout")
                delay = min(interval, remaining)
                self._sleeper(delay)
                interval = min(
                    self._policy.maximum_interval_s,
                    interval * self._policy.backoff_multiplier,
                )
                status = self._status(
                    self._request(
                        "GET",
                        job_path,
                        None,
                        MAX_AUTHORITY_JOB_STATUS_BYTES,
                        expected_status=200,
                    ),
                    job,
                )
                self._report(status, hooks, job)
            if cancel_posted or self._shutdown.is_set():
                raise GroundRegionalVariationCancelled(status.completed, status.total)
            return self._resolve_terminal(status, job, job_path)
        except GroundRegionalVariationCancelled:
            raise
        except Exception:
            if not cancel_posted:
                self._best_effort_cancel(job_path)
            raise

    def _best_effort_cancel(self, job_path: str) -> None:
        """Attempt cleanup without masking the original client terminal."""
        try:
            self._request(
                "POST",
                f"{job_path}/cancel",
                None,
                MAX_AUTHORITY_JOB_STATUS_BYTES,
                expected_status=202,
            )
        except Exception:
            return

    def _resolve_terminal(
        self,
        status: AuthorityJobSnapshot,
        job: RegionalGroundExecutionJob,
        job_path: str,
    ) -> RegionalGroundExecutionResult:
        """Map a terminal snapshot to cancellation, failure, or exact result."""
        if status.status is AuthorityJobStatus.CANCELLED:
            raise GroundRegionalVariationCancelled(status.completed, status.total)
        if status.status is AuthorityJobStatus.FAILED:
            fail_regional_ground_authority(
                authority_failure_stage(status),
                status.completed,
                job,
                RegionalGroundAuthorityClientError("authority_job_failed"),
            )
        try:
            response = self._request(
                "GET",
                f"{job_path}/result",
                None,
                MAX_REGIONAL_GROUND_EXECUTION_RESULT_BYTES,
                expected_status=200,
            )
            text = response.body.decode("utf-8")
            return regional_ground_execution_result_from_json(text, expected_job=job)
        except GroundRegionalVariationFailed:
            raise
        except Exception as error:
            fail_regional_ground_authority(
                GroundRegionalVariationFailureStage.VALIDATION,
                status.completed,
                job,
                error,
            )

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None,
        maximum_bytes: int,
        *,
        expected_status: int,
    ) -> AuthorityHttpResponse:
        """Issue one transport request and enforce the expected response shape."""
        try:
            response = self._transport.request(method, path, body, maximum_bytes)
        except RegionalGroundAuthorityClientError:
            raise
        except Exception as error:
            raise RegionalGroundAuthorityClientError("transport_failure") from error
        if type(response) is not AuthorityHttpResponse:
            raise TypeError("transport response must be exact")
        if response.media_type != _JSON_MEDIA_TYPE:
            raise RegionalGroundAuthorityClientError("invalid_media_type")
        if len(response.body) > maximum_bytes:
            raise RegionalGroundAuthorityClientError("response_too_large")
        if response.status != expected_status:
            raise RegionalGroundAuthorityClientError(
                "authentication_required"
                if response.status == 401
                else "authority_request_failed"
            )
        return response

    @staticmethod
    def _status(
        response: AuthorityHttpResponse, job: RegionalGroundExecutionJob
    ) -> AuthorityJobSnapshot:
        """Decode and validate one canonical job-bound status snapshot."""
        try:
            text = response.body.decode("utf-8")
            return regional_ground_authority_job_status_from_json(text, job)
        except Exception as error:
            failure = GroundRegionalVariationFailed(
                GroundRegionalVariationFailureStage.VALIDATION,
                0,
                job.execution_options.max_trials,
                error,
            )
            failure.__cause__ = error
            raise failure from error

    @staticmethod
    def _report(
        status: AuthorityJobSnapshot,
        hooks: GroundRegionalVariationHooks,
        job: RegionalGroundExecutionJob,
    ) -> None:
        """Forward immutable progress or raise its typed callback terminal."""
        callback = hooks.progress_callback
        if callback is not None:
            try:
                callback(
                    GroundRegionalVariationProgress(status.completed, status.total)
                )
            except Exception as error:
                fail_regional_ground_authority(
                    GroundRegionalVariationFailureStage.PROGRESS_CALLBACK,
                    status.completed,
                    job,
                    error,
                )

    @staticmethod
    def _cancel_requested(
        hooks: GroundRegionalVariationHooks,
        job: RegionalGroundExecutionJob,
        completed: int,
    ) -> bool:
        """Read one exact cooperative-cancellation decision."""
        callback = hooks.cancellation_requested
        if callback is None:
            return False
        try:
            requested: bool = callback()
            if type(requested) is not bool:
                raise TypeError("cancellation callback must return an exact bool")
        except Exception as error:
            fail_regional_ground_authority(
                GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK,
                completed,
                job,
                error,
            )
        return requested


def regional_ground_submitter_if_available(
    *,
    runtime: AuthorityRuntime | None,
    capability: AuthorityCapability,
    policy: RegionalGroundAuthorityPollPolicy = (
        DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY
    ),
) -> LoopbackRegionalGroundSubmitter | None:
    """Return a client only when an exact runtime capability explicitly allows it."""
    if type(capability) is not AuthorityCapability:
        raise TypeError("capability must be exact")
    wire = capability.to_wire()
    if not wire["available"] or not wire["regional_ground_execution"]:
        return None
    if type(runtime) is not AuthorityRuntime:
        raise TypeError("an exact runtime is required for available execution")
    return LoopbackRegionalGroundSubmitter(
        LoopbackAuthorityHttpTransport(runtime, timeout_s=policy.request_timeout_s),
        policy=policy,
    )


__all__ = [
    "AuthorityHttpResponse",
    "DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY",
    "LoopbackRegionalGroundSubmitter",
    "RegionalGroundAuthorityClientError",
    "RegionalGroundAuthorityPollPolicy",
    "RegionalGroundAuthorityTransport",
    "regional_ground_submitter_if_available",
]
