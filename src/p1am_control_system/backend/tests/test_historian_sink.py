"""Unit tests for the historian sink seam.

Covers the forwarding contract that protects the control path: a broken remote
historian must not reduce local durability, must not raise into the scan loop,
and must not change what the throttle decides.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

# Backend deps (sqlmodel/sqlalchemy) aren't installed in the shared CI `tests`
# job, so skip this module there rather than erroring on collection.
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian_sink import (  # noqa: E402
    HistorianSink,
    HistorianWriter,
    NullHistorianSink,
)

pytestmark = pytest.mark.unit

_TS = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)


class _RecordingSink:
    """Captures forwarded scans."""

    def __init__(self) -> None:
        self.scans: list[tuple[dict[str, float], datetime]] = []
        self.closed = False

    def write_scan(self, tags: Any, timestamp: datetime) -> int:
        self.scans.append((dict(tags), timestamp))
        return len(tags)

    def close(self) -> None:
        self.closed = True


class _ExplodingSink:
    """Fails every way a remote historian can fail."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("historian unreachable")
        self.calls = 0

    def write_scan(self, tags: Any, timestamp: datetime) -> int:
        self.calls += 1
        raise self.exc

    def close(self) -> None:
        raise self.exc


def _always_due() -> bool:
    return True


def _never_due() -> bool:
    return False


def _fake_log_scan(recorder: list[Any]) -> Any:
    def _inner(
        session: Any,
        tags: dict[str, float],
        *,
        timestamp: Any = None,
        signal_frame: Any = None,
    ) -> int:
        recorder.append((session, dict(tags), timestamp))
        return len(tags)

    return _inner


# --------------------------------------------------------------- NullSink ---


def test_null_sink_satisfies_the_protocol() -> None:
    assert isinstance(NullHistorianSink(), HistorianSink)


def test_null_sink_accepts_and_discards() -> None:
    sink = NullHistorianSink()
    assert sink.write_scan({"TAG_0": 1.0}, _TS) == 0
    assert sink.close() is None


# ----------------------------------------------------------------- writer ---


def test_writer_persists_locally_and_forwards_the_same_timestamp() -> None:
    """A sample must be correlatable across the two stores exactly."""
    calls: list[Any] = []
    sink = _RecordingSink()
    writer = HistorianWriter(
        due=_always_due,
        sink=sink,
        log_scan=_fake_log_scan(calls),
        clock=lambda: _TS,
    )

    written = writer.write(object(), {"TAG_0": 1.5, "TAG_1": 2.5})

    assert written == 2
    assert len(calls) == 1
    _, local_tags, local_ts = calls[0]
    assert local_tags == {"TAG_0": 1.5, "TAG_1": 2.5}
    assert local_ts == _TS
    assert sink.scans == [({"TAG_0": 1.5, "TAG_1": 2.5}, _TS)]


def test_writer_skips_both_stores_when_throttle_declines() -> None:
    """Local and remote stay in lockstep so the two stores stay comparable."""
    calls: list[Any] = []
    sink = _RecordingSink()
    writer = HistorianWriter(
        due=_never_due,
        sink=sink,
        log_scan=_fake_log_scan(calls),
        clock=lambda: _TS,
    )

    assert writer.write(object(), {"TAG_0": 1.0}) == 0
    assert calls == []
    assert sink.scans == []


def test_writer_consults_the_throttle_exactly_once_per_scan() -> None:
    """`due` consumes the throttle window; calling it twice would double-consume."""
    hits = 0

    def counting_due() -> bool:
        nonlocal hits
        hits += 1
        return True

    writer = HistorianWriter(
        due=counting_due,
        sink=_RecordingSink(),
        log_scan=_fake_log_scan([]),
        clock=lambda: _TS,
    )
    writer.write(object(), {"TAG_0": 1.0})

    assert hits == 1


def test_remote_failure_does_not_reach_the_scan_loop() -> None:
    """The whole point of the seam: a dead historian cannot fault a scan."""
    calls: list[Any] = []
    sink = _ExplodingSink()
    writer = HistorianWriter(
        due=_always_due,
        sink=sink,
        log_scan=_fake_log_scan(calls),
        clock=lambda: _TS,
    )

    written = writer.write(object(), {"TAG_0": 9.0})

    assert written == 1, "local write must still have happened"
    assert len(calls) == 1
    assert sink.calls == 1


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("boom"),
        ConnectionRefusedError("no route"),
        TimeoutError("slow"),
        MemoryError("driver blew up"),
    ],
)
def test_any_remote_exception_type_is_contained(exc: Exception) -> None:
    writer = HistorianWriter(
        due=_always_due,
        sink=_ExplodingSink(exc),
        log_scan=_fake_log_scan([]),
        clock=lambda: _TS,
    )
    assert writer.write(object(), {"TAG_0": 1.0}) == 1


def test_writer_returns_local_row_count_not_forwarded_count() -> None:
    """Callers must not be able to confuse 'unreachable' with 'not recorded'."""
    writer = HistorianWriter(
        due=_always_due,
        sink=_ExplodingSink(),
        log_scan=_fake_log_scan([]),
        clock=lambda: _TS,
    )
    assert writer.write(object(), {"TAG_0": 1.0, "TAG_1": 2.0}) == 2


def test_writer_without_a_sink_still_persists_locally() -> None:
    calls: list[Any] = []
    writer = HistorianWriter(due=_always_due, log_scan=_fake_log_scan(calls))

    assert writer.write(object(), {"TAG_0": 1.0}) == 1
    assert len(calls) == 1
    assert isinstance(writer.sink, NullHistorianSink)


def test_writer_forwards_signal_frame_to_the_local_write() -> None:
    """Signal quality must survive the historian-forwarding seam.

    Regression guard for the #4065 + #4091 consolidation: ``poll_runtime``
    invokes its injected ``ScanLogger`` with ``signal_frame=``, so a
    ``HistorianWriter`` that swallowed the keyword would silently drop quality
    metadata from every persisted sample (and, before the writer accepted it,
    raise ``TypeError`` inside the scan loop).
    """
    seen: list[Any] = []

    def _recording_log_scan(
        _session: Any,
        tags: dict[str, float],
        *,
        timestamp: Any = None,
        signal_frame: Any = None,
    ) -> int:
        seen.append(signal_frame)
        return len(tags)

    frame = object()
    writer = HistorianWriter(
        due=_always_due,
        sink=_RecordingSink(),
        log_scan=_recording_log_scan,
        clock=lambda: _TS,
    )

    assert writer.write(object(), {"TAG_0": 1.0}, signal_frame=frame) == 1
    assert seen == [frame]


def test_writer_defaults_signal_frame_to_none() -> None:
    """Two-argument callers (the pre-#4091 shape) still work."""
    seen: list[Any] = []

    def _recording_log_scan(
        _session: Any,
        tags: dict[str, float],
        *,
        timestamp: Any = None,
        signal_frame: Any = None,
    ) -> int:
        seen.append(signal_frame)
        return len(tags)

    writer = HistorianWriter(due=_always_due, log_scan=_recording_log_scan)

    assert writer.write(object(), {"TAG_0": 1.0}) == 1
    assert seen == [None]


def test_sink_never_receives_signal_frame() -> None:
    """The forwarding contract stays ``{tag: value}`` + timestamp."""
    sink = _RecordingSink()
    writer = HistorianWriter(
        due=_always_due,
        sink=sink,
        log_scan=_fake_log_scan([]),
        clock=lambda: _TS,
    )

    writer.write(object(), {"TAG_0": 1.0}, signal_frame=object())

    assert sink.scans == [({"TAG_0": 1.0}, _TS)]


def test_close_forwards_to_the_sink() -> None:
    sink = _RecordingSink()
    HistorianWriter(due=_always_due, sink=sink).close()
    assert sink.closed is True


def test_close_swallows_sink_failure() -> None:
    """Shutdown must not fail because the historian is unreachable."""
    HistorianWriter(due=_always_due, sink=_ExplodingSink()).close()


# -------------------------------------------------------------------- DbC ---


def test_rejects_non_callable_due() -> None:
    with pytest.raises(TypeError, match="due must be callable"):
        bad: Any = "nope"
        HistorianWriter(due=bad)


def test_rejects_non_callable_log_scan() -> None:
    with pytest.raises(TypeError, match="log_scan must be callable"):
        bad: Any = object()
        HistorianWriter(due=_always_due, log_scan=bad)


def test_rejects_non_callable_clock() -> None:
    with pytest.raises(TypeError, match="clock must be callable"):
        bad: Any = 123
        HistorianWriter(due=_always_due, clock=bad)


def test_rejects_a_sink_that_is_not_a_sink() -> None:
    with pytest.raises(TypeError, match="sink must implement HistorianSink"):
        bad: Any = object()
        HistorianWriter(due=_always_due, sink=bad)
