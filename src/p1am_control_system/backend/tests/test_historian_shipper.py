"""Unit tests for the store-and-forward shipper.

The properties under test are safety properties, not performance ones: the
producer side must never block, never raise, and never grow without bound, no
matter what the remote destination does.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian_shipper import (  # noqa: E402
    RemoteHistorianWriter,
    Sample,
    StoreAndForwardSink,
)
from historian_sink import HistorianSink  # noqa: E402

pytestmark = pytest.mark.unit

_TS = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)

# Bound every wait so a regression that reintroduces blocking fails fast rather
# than hanging the suite.
_WAIT_TIMEOUT_S = 5.0


class _FakeRemote:
    """A cooperative remote writer with controllable failure modes."""

    def __init__(
        self,
        *,
        fail_connect: bool = False,
        fail_write: bool = False,
        block_write: threading.Event | None = None,
    ) -> None:
        self.fail_connect = fail_connect
        self.fail_write = fail_write
        self.block_write = block_write
        self.batches: list[list[Sample]] = []
        self.connects = 0
        self.closes = 0
        self._lock = threading.Lock()
        self.wrote = threading.Event()

    def connect(self) -> None:
        with self._lock:
            self.connects += 1
        if self.fail_connect:
            raise ConnectionRefusedError("historian down")

    def write_batch(self, samples: Any) -> int:
        if self.block_write is not None:
            self.block_write.wait(_WAIT_TIMEOUT_S)
        if self.fail_write:
            raise RuntimeError("write failed")
        with self._lock:
            self.batches.append(list(samples))
        self.wrote.set()
        return len(samples)

    def close(self) -> None:
        with self._lock:
            self.closes += 1

    def total_written(self) -> int:
        with self._lock:
            return sum(len(b) for b in self.batches)


def _no_jitter() -> float:
    return 0.0


def _wait_for(predicate: Any, timeout_s: float = _WAIT_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------ contract ---


def test_shipper_satisfies_the_sink_protocol() -> None:
    sink = StoreAndForwardSink(_FakeRemote())
    assert isinstance(sink, HistorianSink)


def test_fake_remote_satisfies_the_remote_protocol() -> None:
    assert isinstance(_FakeRemote(), RemoteHistorianWriter)


# ----------------------------------------------------------------- happy path ---


def test_samples_reach_the_remote() -> None:
    remote = _FakeRemote()
    sink = StoreAndForwardSink(remote, batch_size=10, flush_interval_s=0.05)
    sink.start()
    try:
        assert sink.write_scan({"TAG_0": 1.0, "TAG_1": 2.0}, _TS) == 2
        assert _wait_for(lambda: remote.total_written() == 2)
    finally:
        sink.close(timeout_s=2.0)

    flat = [s for batch in remote.batches for s in batch]
    assert sorted(s[1] for s in flat) == ["TAG_0", "TAG_1"]
    assert all(s[0] == _TS for s in flat)


def test_non_numeric_values_are_skipped_not_fatal() -> None:
    """The local historian already rejects these loudly; forwarding just skips."""
    sink = StoreAndForwardSink(_FakeRemote())
    tags: Any = {"TAG_0": 1.0, "TAG_1": "oops"}
    assert sink.write_scan(tags, _TS) == 1


# ------------------------------------------------------- producer never blocks ---


def test_enqueue_does_not_block_when_the_remote_hangs() -> None:
    """The property that protects the 10 Hz scan loop."""
    blocker = threading.Event()
    remote = _FakeRemote(block_write=blocker)
    sink = StoreAndForwardSink(remote, queue_max=50, flush_interval_s=0.01)
    sink.start()
    try:
        start = time.monotonic()
        for _ in range(200):
            sink.write_scan({"TAG_0": 1.0}, _TS)
        elapsed = time.monotonic() - start
        # 200 enqueues against a wedged remote. Generous bound — the point is
        # that this is not gated on the blocked writer at all.
        assert elapsed < 1.0, f"enqueue blocked for {elapsed:.3f}s"
    finally:
        blocker.set()
        sink.close(timeout_s=2.0)


def test_enqueue_never_raises_when_the_remote_is_dead() -> None:
    sink = StoreAndForwardSink(_FakeRemote(fail_connect=True), queue_max=10)
    sink.start()
    try:
        for _ in range(100):
            sink.write_scan({"TAG_0": 1.0}, _TS)
    finally:
        sink.close(timeout_s=2.0)


def test_writes_work_before_start_is_called() -> None:
    """Ordering must not matter; a scan before the worker starts is not an error."""
    sink = StoreAndForwardSink(_FakeRemote(), queue_max=10)
    assert sink.write_scan({"TAG_0": 1.0}, _TS) == 1


# ---------------------------------------------------------------- boundedness ---


def test_queue_is_bounded_and_drops_oldest() -> None:
    """An unbounded queue on the control Pi is an OOM crash of the controller."""
    sink = StoreAndForwardSink(_FakeRemote(fail_connect=True), queue_max=10)
    # Not started: nothing drains, so overflow is deterministic.
    for i in range(100):
        sink.write_scan({f"TAG_{i}": float(i)}, _TS)

    stats = sink.stats()
    assert stats.queue_depth <= 10
    assert stats.dropped_total >= 90


def test_drop_counter_is_accurate() -> None:
    sink = StoreAndForwardSink(_FakeRemote(fail_connect=True), queue_max=5)
    for i in range(25):
        sink.write_scan({f"TAG_{i}": 1.0}, _TS)

    stats = sink.stats()
    assert stats.queue_depth + stats.dropped_total == 25


# -------------------------------------------------------------------- failure ---


def test_reconnects_after_the_remote_recovers() -> None:
    remote = _FakeRemote(fail_connect=True)
    sink = StoreAndForwardSink(remote, flush_interval_s=0.01, jitter=_no_jitter)
    sink.start()
    try:
        assert _wait_for(lambda: remote.connects >= 1)
        assert not sink.stats().connected

        remote.fail_connect = False
        assert _wait_for(lambda: sink.stats().connected)

        sink.write_scan({"TAG_0": 42.0}, _TS)
        assert _wait_for(lambda: remote.total_written() >= 1)
    finally:
        sink.close(timeout_s=2.0)


def test_write_failure_marks_disconnected_and_counts_drops() -> None:
    remote = _FakeRemote(fail_write=True)
    sink = StoreAndForwardSink(
        remote, batch_size=5, flush_interval_s=0.01, jitter=_no_jitter
    )
    sink.start()
    try:
        for _ in range(5):
            sink.write_scan({"TAG_0": 1.0}, _TS)
        assert _wait_for(lambda: sink.stats().dropped_total > 0)
        assert _wait_for(lambda: remote.closes >= 1)
    finally:
        sink.close(timeout_s=2.0)


def test_stats_report_lag_after_a_success() -> None:
    remote = _FakeRemote()
    sink = StoreAndForwardSink(remote, flush_interval_s=0.01)
    sink.start()
    try:
        sink.write_scan({"TAG_0": 1.0}, _TS)
        assert _wait_for(lambda: sink.stats().last_success_ts is not None)
        stats = sink.stats()
        assert stats.lag_s is not None
        assert stats.lag_s >= 0.0
        assert stats.shipped_total >= 1
    finally:
        sink.close(timeout_s=2.0)


def test_stats_before_any_activity_are_a_clean_zero() -> None:
    stats = StoreAndForwardSink(_FakeRemote(), queue_max=7).stats()
    assert stats.enabled is True
    assert stats.connected is False
    assert stats.queue_depth == 0
    assert stats.queue_max == 7
    assert stats.shipped_total == 0
    assert stats.dropped_total == 0
    assert stats.last_success_ts is None
    assert stats.lag_s is None


def test_stats_as_dict_is_json_serialisable() -> None:
    import json

    payload = StoreAndForwardSink(_FakeRemote()).stats().as_dict()
    json.loads(json.dumps(payload))
    assert payload["enabled"] is True


# ------------------------------------------------------------------- shutdown ---


def test_close_is_bounded_when_the_remote_hangs() -> None:
    """Shutdown must not hang on an unreachable historian."""
    blocker = threading.Event()
    remote = _FakeRemote(block_write=blocker)
    sink = StoreAndForwardSink(remote, flush_interval_s=0.01)
    sink.start()
    try:
        sink.write_scan({"TAG_0": 1.0}, _TS)
        time.sleep(0.1)
        start = time.monotonic()
        sink.close(timeout_s=0.5)
        elapsed = time.monotonic() - start
        assert elapsed < 3.0, f"close took {elapsed:.2f}s"
    finally:
        blocker.set()


def test_close_is_idempotent() -> None:
    sink = StoreAndForwardSink(_FakeRemote())
    sink.start()
    sink.close(timeout_s=1.0)
    sink.close(timeout_s=1.0)


def test_start_is_idempotent() -> None:
    sink = StoreAndForwardSink(_FakeRemote())
    sink.start()
    sink.start()
    try:
        assert _wait_for(lambda: sink.stats().connected)
    finally:
        sink.close(timeout_s=2.0)


# ------------------------------------------------------------------------ DbC ---


def test_rejects_a_writer_that_is_not_a_remote_writer() -> None:
    with pytest.raises(TypeError, match="writer must implement"):
        bad: Any = object()
        StoreAndForwardSink(bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_non_positive_queue_max(bad: int) -> None:
    with pytest.raises(ValueError, match="queue_max must be >= 1"):
        StoreAndForwardSink(_FakeRemote(), queue_max=bad)


@pytest.mark.parametrize("bad", [0, -5])
def test_rejects_non_positive_batch_size(bad: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        StoreAndForwardSink(_FakeRemote(), batch_size=bad)


def test_rejects_non_int_queue_max() -> None:
    with pytest.raises(TypeError, match="queue_max must be an int"):
        bad: Any = 1.5
        StoreAndForwardSink(_FakeRemote(), queue_max=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_bad_flush_interval(bad: float) -> None:
    with pytest.raises(ValueError, match="flush_interval_s"):
        StoreAndForwardSink(_FakeRemote(), flush_interval_s=bad)
