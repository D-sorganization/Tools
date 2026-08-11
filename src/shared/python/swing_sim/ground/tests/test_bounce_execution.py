"""UI-neutral repeated-bounce request execution contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    RepeatedBounceRequest,
    RepeatedBounceRequestResultPair,
    execute_repeated_bounce_request,
    repeated_bounce_result_from_json,
    repeated_bounce_result_to_json,
)

from ._support import _surface_run_request

_GOLDEN_REQUEST = (
    Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__"
    / "ground_repeated_bounce_request_wire_golden_v1.json"
)


def _execution_request(*, capture_speed_m_s: float = 0.05) -> RepeatedBounceRequest:
    """Return the golden analytic request with a controllable capture threshold."""
    return RepeatedBounceRequest(_surface_run_request(), capture_speed_m_s)


def test_executor_requires_exact_request_and_callable_or_none_cancellation() -> None:
    request = _execution_request()

    class RequestSubclass(RepeatedBounceRequest):
        """Prove that nominal subclasses cannot bypass the exact-type boundary."""

    with pytest.raises(ValueError, match="exact RepeatedBounceRequest"):
        execute_repeated_bounce_request(cast(RepeatedBounceRequest, object()))
    with pytest.raises(ValueError, match="exact RepeatedBounceRequest"):
        execute_repeated_bounce_request(RequestSubclass(request.ground_request))
    with pytest.raises(ValueError, match="callable or None"):
        execute_repeated_bounce_request(request, is_cancelled=cast(Any, 1))

    pair = execute_repeated_bounce_request(request, is_cancelled=None)
    assert type(pair) is RepeatedBounceRequestResultPair


def test_executor_returns_identity_safe_pair_for_golden_execution() -> None:
    request = _execution_request()
    fixture = cast(
        dict[str, Any], json.loads(_GOLDEN_REQUEST.read_text(encoding="utf-8"))
    )

    pair = execute_repeated_bounce_request(request)

    assert type(pair) is RepeatedBounceRequestResultPair
    assert pair.request is request
    assert pair.execution_input_sha256 == request.execution_input_sha256
    assert pair.execution_input_sha256 == fixture["request"]["execution_input_sha256"]
    assert pair.result.request_fingerprint_sha256 == request.ground_request_sha256
    assert pair.result.request_id == request.request_id
    assert pair.result.surface_id == request.surface_id
    assert pair.result.frame is request.frame
    assert (pair.result.model_id, pair.result.model_version) == (
        request.model_id,
        request.model_version,
    )


def test_executor_consumes_capture_threshold_from_request() -> None:
    bounce = execute_repeated_bounce_request(_execution_request(capture_speed_m_s=0.05))
    captured = execute_repeated_bounce_request(
        _execution_request(capture_speed_m_s=0.2)
    )

    assert bounce.result.impacts[0].effective_restitution == pytest.approx(0.42)
    assert captured.result.impacts[0].effective_restitution == pytest.approx(0.0)
    assert captured.result.termination.reason is BounceTerminationReason.SETTLED_TO_SKID
    assert len(captured.result.events) == 1
    assert bounce.execution_input_sha256 != captured.execution_input_sha256


def test_executor_preflight_cancellation_returns_valid_pair() -> None:
    request = _execution_request()

    pair = execute_repeated_bounce_request(request, is_cancelled=lambda: True)

    assert type(pair) is RepeatedBounceRequestResultPair
    assert pair.request is request
    assert pair.result.termination.reason is BounceTerminationReason.CANCELLED
    assert pair.result.termination.elapsed_time_s == pytest.approx(0.0)
    assert pair.result.trajectory == ()
    assert pair.result.events == ()
    assert pair.result.impacts == ()
    assert pair.result.airborne_segments == ()
    assert pair.result.handoff_state is None


def test_serialized_result_round_trip_can_be_paired_with_execution_request() -> None:
    request = _execution_request()
    pair = execute_repeated_bounce_request(request)
    serialized = repeated_bounce_result_to_json(pair.result)

    restored = repeated_bounce_result_from_json(serialized)
    restored_pair = RepeatedBounceRequestResultPair(request, restored)

    assert repeated_bounce_result_to_json(restored) == serialized
    assert restored.request_fingerprint_sha256 == pair.result.request_fingerprint_sha256
    assert restored_pair.execution_input_sha256 == pair.execution_input_sha256
