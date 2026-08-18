"""UI-neutral execution binding for validated repeated-bounce requests."""

from __future__ import annotations

from .bounce_request_wire import (
    RepeatedBounceRequest,
    RepeatedBounceRequestResultPair,
)
from .bounce_simulation import simulate_repeated_bounce
from .bounce_types import CancellationCheck


def execute_repeated_bounce_request(
    request: RepeatedBounceRequest,
    *,
    is_cancelled: CancellationCheck | None = None,
) -> RepeatedBounceRequestResultPair:
    """Execute one exact request and return its identity-validated result pair."""
    if type(request) is not RepeatedBounceRequest:
        raise ValueError("request must be an exact RepeatedBounceRequest")
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("is_cancelled must be callable or None")

    result = simulate_repeated_bounce(
        request.ground_request,
        request.settings,
        is_cancelled=is_cancelled,
    )
    return RepeatedBounceRequestResultPair(request, result)


__all__ = ["execute_repeated_bounce_request"]
