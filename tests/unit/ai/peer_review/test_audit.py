"""Tests for ai.peer_review._audit (Tools #4493).

Pins the package-structure fix: ``_audit_event`` lives in its own
``_audit`` module (matching UpstreamDrift) instead of being defined
inline in ``coordinator.py``, so it is importable independently of the
coordinator.
"""

from __future__ import annotations

from shared.python.ai.peer_review._audit import _audit_event
from shared.python.ai.peer_review.coordinator import (
    _audit_event as coordinator_audit_event,
)


def test_audit_event_is_defined_in_its_own_module() -> None:
    """`_audit_event` must be importable from `_audit`, not just `coordinator`."""
    event = _audit_event("started", request_id="req-1")

    assert event["kind"] == "started"
    assert event["request_id"] == "req-1"
    assert "timestamp" in event
    assert "message" not in event
    assert "extra" not in event


def test_audit_event_includes_optional_fields() -> None:
    event = _audit_event(
        "completed",
        request_id="req-2",
        message="done",
        extra={"votes": 3},
    )

    assert event["message"] == "done"
    assert event["extra"] == {"votes": 3}


def test_coordinator_reexports_the_same_audit_event_function() -> None:
    """coordinator.py imports `_audit_event` from `_audit` rather than redefining it."""
    assert coordinator_audit_event is _audit_event
