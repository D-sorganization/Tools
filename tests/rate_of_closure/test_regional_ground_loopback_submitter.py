"""Strict transport contracts for the regional-ground Qt submitter port."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque

import pytest

from rate_of_closure.application.regional_ground_execution_job import (
    regional_ground_execution_job_to_json,
)
from rate_of_closure.application.regional_ground_execution_result import (
    regional_ground_execution_result_to_json,
)
from rate_of_closure.application.regional_ground_loopback_submitter import (
    AuthorityHttpResponse,
    LoopbackRegionalGroundSubmitter,
    RegionalGroundAuthorityPollPolicy,
    RegionalGroundAuthorityTransport,
    regional_ground_submitter_if_available,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationFailureStage,
    GroundRegionalVariationHooks,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.api import JOB_COLLECTION_PATH
from rate_of_closure.web_authority.capability import DEFAULT_UNAVAILABLE_CAPABILITY
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job, _result

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _status(
    state: str,
    completed: int,
    *,
    result_available: bool = False,
    failure: dict[str, str] | None = None,
) -> bytes:
    job = _job()
    return json.dumps(
        {
            "schema_version": (
                "rate-of-closure/regional-ground-authority-job-status/v1"
            ),
            "job_id": job.job_id,
            "job_sha256": job.job_sha256,
            "status": state,
            "completed": completed,
            "total": 4,
            "result_available": result_available,
            "failure": failure,
        },
        separators=(",", ":"),
    ).encode("utf-8")


class _ScriptedTransport(RegionalGroundAuthorityTransport):
    def __init__(self, responses: list[AuthorityHttpResponse]) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[str, str, bytes | None, int]] = []
        self.closed = False

    def request(
        self, method: str, path: str, body: bytes | None, maximum_bytes: int
    ) -> AuthorityHttpResponse:
        self.requests.append((method, path, body, maximum_bytes))
        assert not self.closed
        return self.responses.popleft()

    def close(self) -> None:
        self.closed = True


def _response(status: int, body: bytes) -> AuthorityHttpResponse:
    return AuthorityHttpResponse(status, "application/json", body)


def test_submitter_posts_canonical_job_polls_and_returns_exact_bound_result() -> None:
    job = _job()
    result_text = regional_ground_execution_result_to_json(_result(job)).encode()
    transport = _ScriptedTransport(
        [
            _response(202, _status("queued", 0)),
            _response(200, _status("running", 2)),
            _response(200, _status("succeeded", 4, result_available=True)),
            _response(200, result_text),
        ]
    )
    reports: list[GroundRegionalVariationProgress] = []
    sleeps: list[float] = []
    submitter = LoopbackRegionalGroundSubmitter(
        transport,
        policy=RegionalGroundAuthorityPollPolicy(
            poll_timeout_s=5.0,
            initial_interval_s=0.01,
            maximum_interval_s=0.02,
            backoff_multiplier=2.0,
        ),
        sleeper=sleeps.append,
    )

    result = submitter(
        job, GroundRegionalVariationHooks(progress_callback=reports.append)
    )

    job_path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
    assert result == _result(job)
    assert transport.requests == [
        (
            "POST",
            JOB_COLLECTION_PATH,
            regional_ground_execution_job_to_json(job).encode("utf-8"),
            4_096,
        ),
        ("GET", job_path, None, 4_096),
        ("GET", job_path, None, 4_096),
        ("GET", f"{job_path}/result", None, 8_388_608),
    ]
    assert [(item.completed, item.total) for item in reports] == [
        (0, 4),
        (2, 4),
        (4, 4),
    ]
    assert sleeps == [0.01, 0.02]


def test_cooperative_cancel_posts_once_and_suppresses_late_result() -> None:
    job = _job()
    transport = _ScriptedTransport(
        [
            _response(202, _status("queued", 0)),
            _response(202, _status("cancel_requested", 0)),
            _response(200, _status("succeeded", 4, result_available=True)),
        ]
    )
    checks = iter((False, True, True))
    submitter = LoopbackRegionalGroundSubmitter(transport, sleeper=lambda _delay: None)

    with pytest.raises(GroundRegionalVariationCancelled) as raised:
        submitter(
            job,
            GroundRegionalVariationHooks(cancellation_requested=lambda: next(checks)),
        )

    job_path = f"{JOB_COLLECTION_PATH}/{job.job_id}"
    assert (raised.value.completed, raised.value.total) == (4, 4)
    assert [request[:2] for request in transport.requests] == [
        ("POST", JOB_COLLECTION_PATH),
        ("POST", f"{job_path}/cancel"),
        ("GET", job_path),
    ]
    assert all(not path.endswith("/result") for _, path, _, _ in transport.requests)


def test_failed_status_becomes_typed_terminal_without_server_detail() -> None:
    secret = "Bearer secret-must-not-appear"
    transport = _ScriptedTransport(
        [
            _response(202, _status("queued", 0)),
            _response(
                200,
                _status(
                    "failed",
                    1,
                    failure={"code": "execution_failed", "stage": "runner"},
                ),
            ),
        ]
    )
    submitter = LoopbackRegionalGroundSubmitter(transport, sleeper=lambda _delay: None)

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        submitter(_job(), GroundRegionalVariationHooks())

    assert raised.value.stage is GroundRegionalVariationFailureStage.EXECUTOR
    assert (raised.value.completed, raised.value.total) == (1, 4)
    assert raised.value.cause_type == "RegionalGroundAuthorityClientError"
    assert secret not in str(raised.value)


def test_preflight_failure_preserves_its_public_typed_terminal() -> None:
    transport = _ScriptedTransport(
        [
            _response(
                202,
                _status(
                    "failed",
                    0,
                    failure={"code": "execution_failed", "stage": "preflight"},
                ),
            )
        ]
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        LoopbackRegionalGroundSubmitter(transport)(
            _job(), GroundRegionalVariationHooks()
        )

    assert raised.value.stage is GroundRegionalVariationFailureStage.PREFLIGHT


def test_stale_status_and_unbound_result_fail_validation_without_publication() -> None:
    stale = json.loads(_status("running", 1))
    stale["job_sha256"] = "0" * 64
    stale_transport = _ScriptedTransport(
        [_response(202, json.dumps(stale).encode("utf-8"))]
    )

    with pytest.raises(GroundRegionalVariationFailed) as stale_failure:
        LoopbackRegionalGroundSubmitter(stale_transport)(
            _job(), GroundRegionalVariationHooks()
        )
    assert stale_failure.value.stage is GroundRegionalVariationFailureStage.VALIDATION

    other = _job("different-job")
    unbound_transport = _ScriptedTransport(
        [
            _response(202, _status("succeeded", 4, result_available=True)),
            _response(
                200,
                regional_ground_execution_result_to_json(_result(other)).encode(),
            ),
        ]
    )
    with pytest.raises(GroundRegionalVariationFailed) as unbound_failure:
        LoopbackRegionalGroundSubmitter(unbound_transport)(
            _job(), GroundRegionalVariationHooks()
        )
    assert unbound_failure.value.stage is GroundRegionalVariationFailureStage.VALIDATION
    job_path = f"{JOB_COLLECTION_PATH}/{_job().job_id}"
    assert unbound_transport.requests[-1][:2] == ("POST", f"{job_path}/cancel")


def test_timeout_uses_bounded_backoff_and_returns_typed_failure() -> None:
    transport = _ScriptedTransport(
        [
            _response(202, _status("queued", 0)),
            _response(200, _status("running", 0)),
            _response(200, _status("running", 0)),
        ]
    )
    now = [0.0]
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    submitter = LoopbackRegionalGroundSubmitter(
        transport,
        policy=RegionalGroundAuthorityPollPolicy(
            poll_timeout_s=0.03,
            initial_interval_s=0.01,
            maximum_interval_s=0.02,
            backoff_multiplier=2.0,
        ),
        monotonic=lambda: now[0],
        sleeper=sleep,
    )

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        submitter(_job(), GroundRegionalVariationHooks())

    assert raised.value.stage is GroundRegionalVariationFailureStage.EXECUTOR
    assert raised.value.cause_type == "RegionalGroundAuthorityClientError"
    assert sleeps == [0.01, 0.019999999999999997]
    job_path = f"{JOB_COLLECTION_PATH}/{_job().job_id}"
    assert transport.requests[-1][:2] == ("POST", f"{job_path}/cancel")


def test_close_requests_active_cancel_waits_and_rejects_future_calls() -> None:
    entered = threading.Event()
    allow_cancel = threading.Event()

    class _BlockingTransport(_ScriptedTransport):
        def request(
            self, method: str, path: str, body: bytes | None, maximum_bytes: int
        ) -> AuthorityHttpResponse:
            if path.endswith("/cancel"):
                entered.set()
                assert allow_cancel.wait(2.0)
                return _response(202, _status("cancel_requested", 0))
            return super().request(method, path, body, maximum_bytes)

    transport = _BlockingTransport(
        [
            _response(202, _status("queued", 0)),
            _response(200, _status("cancelled", 0)),
        ]
    )
    submitter = LoopbackRegionalGroundSubmitter(transport, sleeper=lambda _delay: None)
    terminals: list[BaseException] = []
    thread = threading.Thread(
        target=lambda: _capture_terminal(submitter, terminals), daemon=True
    )
    thread.start()
    submitter.request_shutdown()
    assert entered.wait(2.0)
    allow_cancel.set()
    submitter.close(timeout_s=2.0)
    thread.join(2.0)

    assert isinstance(terminals[0], GroundRegionalVariationCancelled)
    assert transport.closed
    with pytest.raises(GroundRegionalVariationFailed):
        submitter(_job(), GroundRegionalVariationHooks())


def _capture_terminal(
    submitter: LoopbackRegionalGroundSubmitter, terminals: list[BaseException]
) -> None:
    try:
        submitter(_job(), GroundRegionalVariationHooks())
    except BaseException as error:  # test helper records the exact terminal
        terminals.append(error)


def test_unavailable_capability_keeps_submitter_unregistered() -> None:
    assert (
        regional_ground_submitter_if_available(
            runtime=None, capability=DEFAULT_UNAVAILABLE_CAPABILITY
        )
        is None
    )


def test_transport_and_client_errors_never_log_authority_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "never-log-this-authority-token"
    transport = _ScriptedTransport(
        [_response(503, b'{"code":"execution_unavailable","detail":"no"}')]
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(GroundRegionalVariationFailed):
        LoopbackRegionalGroundSubmitter(transport)(
            _job(), GroundRegionalVariationHooks()
        )

    assert token not in caplog.text


def test_transport_exception_text_is_not_published() -> None:
    secret = "transport-secret-must-not-escape"

    class _FailingTransport(_ScriptedTransport):
        def request(
            self, method: str, path: str, body: bytes | None, maximum_bytes: int
        ) -> AuthorityHttpResponse:
            raise OSError(f"connection failed with {secret}")

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        LoopbackRegionalGroundSubmitter(_FailingTransport([]))(
            _job(), GroundRegionalVariationHooks()
        )

    assert raised.value.cause_type == "RegionalGroundAuthorityClientError"
    assert secret not in str(raised.value)


@pytest.mark.parametrize(
    ("callback_name", "stage"),
    [
        (
            "cancellation_requested",
            GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK,
        ),
        ("progress_callback", GroundRegionalVariationFailureStage.PROGRESS_CALLBACK),
    ],
)
def test_callback_errors_keep_their_typed_failure_stage(
    callback_name: str, stage: GroundRegionalVariationFailureStage
) -> None:
    transport = _ScriptedTransport([_response(202, _status("queued", 0))])

    def fail_callback(*_args: object) -> bool:
        raise LookupError("callback failed")

    hooks = GroundRegionalVariationHooks(**{callback_name: fail_callback})
    with pytest.raises(GroundRegionalVariationFailed) as raised:
        LoopbackRegionalGroundSubmitter(transport)(_job(), hooks)

    assert raised.value.stage is stage
    assert (raised.value.completed, raised.value.total) == (0, 4)
    if callback_name == "progress_callback":
        job_path = f"{JOB_COLLECTION_PATH}/{_job().job_id}"
        assert transport.requests[-1][:2] == ("POST", f"{job_path}/cancel")


def test_non_boolean_cancellation_answer_is_a_typed_callback_failure() -> None:
    hooks = GroundRegionalVariationHooks(cancellation_requested=lambda: 1)  # type: ignore[arg-type,return-value]

    with pytest.raises(GroundRegionalVariationFailed) as raised:
        LoopbackRegionalGroundSubmitter(_ScriptedTransport([]))(_job(), hooks)

    assert (
        raised.value.stage is GroundRegionalVariationFailureStage.CANCELLATION_CALLBACK
    )
