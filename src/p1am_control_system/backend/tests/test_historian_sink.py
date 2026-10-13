"""Unit tests for the HistorianSink protocol slice (issue #5540, H1).

Enforces:
1. Protocol contract: @runtime_checkable HistorianSink with submit,
   drain_once, stats, close.
2. Fake in-memory sink satisfies the protocol structurally and at runtime.
3. HistorianWriter implements the protocol with zero behavior change and
   exposes stats().
4. ThrottledHistorianSink works with any HistorianSink backend.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian import HistorianRecord, HistorianWriter, ThrottledHistorianSink
from historian_sink import HistorianSink


class FakeHistorianSink:
    """In-memory sink test double satisfying HistorianSink."""

    def __init__(self) -> None:
        self.submitted: list[HistorianRecord] = []
        self.closed: bool = False
        self._dropped: int = 0
        self._drained: int = 0

    def submit(self, record: HistorianRecord) -> bool:
        if not isinstance(record, HistorianRecord):
            raise TypeError(
                f"record must be a HistorianRecord, got {type(record).__name__}"
            )
        if self.closed:
            self._dropped += 1
            return False
        if record.is_empty:
            return False
        self.submitted.append(record)
        return True

    async def drain_once(self) -> int:
        count = len(self.submitted)
        self._drained += count
        self.submitted.clear()
        return count

    def stats(self) -> Mapping[str, int]:
        return {
            "submitted": len(self.submitted),
            "drained": self._drained,
            "dropped": self._dropped,
        }

    def close(self) -> None:
        self.closed = True


class _MinimalSession:
    def add(self, _row: Any) -> None:
        pass

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def _dummy_session_factory() -> Iterator[_MinimalSession]:
    yield _MinimalSession()


def test_fake_in_memory_sink_satisfies_protocol() -> None:
    fake = FakeHistorianSink()
    assert isinstance(fake, HistorianSink)

    # Verify basic behavior
    rec = HistorianRecord(tags={"tag1": 42.0}, quality="live")
    assert fake.submit(rec) is True
    assert len(fake.submitted) == 1

    # Empty record is not enqueued
    empty_rec = HistorianRecord(tags=None)
    assert fake.submit(empty_rec) is False

    stats = fake.stats()
    assert isinstance(stats, Mapping)
    assert stats["submitted"] == 1

    fake.close()
    assert fake.closed is True
    assert fake.submit(rec) is False


@pytest.mark.asyncio
async def test_fake_in_memory_sink_drain() -> None:
    fake = FakeHistorianSink()
    fake.submit(HistorianRecord(tags={"t": 1.0}, quality="live"))
    fake.submit(HistorianRecord(tags={"t": 2.0}, quality="live"))
    drained = await fake.drain_once()
    assert drained == 2
    assert len(fake.submitted) == 0


def test_historian_writer_implements_protocol() -> None:
    writer = HistorianWriter(_dummy_session_factory)
    assert isinstance(writer, HistorianSink)

    # Check stats() mapping
    stats = writer.stats()
    assert isinstance(stats, Mapping)
    expected_keys = {
        "write_failures",
        "dropped_samples",
        "rows_written",
        "events_written",
        "retries",
        "queue_depth",
    }
    assert expected_keys.issubset(stats.keys())
    assert all(isinstance(v, (int, float)) for v in stats.values())

    # Check close() is idempotent and does not raise
    writer.close()
    writer.close()


def test_throttled_historian_sink_with_any_sink() -> None:
    fake = FakeHistorianSink()
    is_due = True
    throttled = ThrottledHistorianSink(fake, lambda: is_due)

    # Passes through when sample is due
    rec_with_tags = HistorianRecord(tags={"T1": 100.0}, events=("ev1",), quality="live")
    throttled.submit(rec_with_tags)
    assert len(fake.submitted) == 1
    assert fake.submitted[0].tags == {"T1": 100.0}
    assert fake.submitted[0].events == ("ev1",)

    # Strips tags when sample is NOT due, but preserves non-resamplable events
    is_due = False
    fake.submitted.clear()
    throttled.submit(rec_with_tags)
    assert len(fake.submitted) == 1
    assert fake.submitted[0].tags is None
    assert fake.submitted[0].events == ("ev1",)

    # Drops completely when no events and not due
    fake.submitted.clear()
    rec_only_tags = HistorianRecord(tags={"T1": 100.0}, quality="live")
    throttled.submit(rec_only_tags)
    assert len(fake.submitted) == 0

    # ThrottledHistorianSink itself also satisfies HistorianSink
    assert isinstance(throttled, HistorianSink)
    assert isinstance(throttled.stats(), Mapping)
    throttled.close()
    assert fake.closed is True


@pytest.mark.asyncio
async def test_poll_once_writes_to_historian_sink() -> None:
    from poll_runtime import poll_once

    fake_sink = FakeHistorianSink()

    class _SimplePLC:
        connected = True

        async def read_tags(self) -> dict[str, float]:
            return {"TAG_0": 42.0}

        async def trigger_estop(self) -> None:
            pass

    class _SimpleWS:
        async def broadcast(self, _payload: Any) -> None:
            pass

    class _SimpleAlicats:
        def get_devices_data(self) -> list[Any]:
            return []

    class _SimplePower:
        async def poll(self, _tags: Any) -> Any:
            return type("Status", (), {"model_dump": lambda _s: {}})()

    class _SimpleAlarmEngine:
        def update_tag(self, _name: str, _value: float) -> list[Any]:
            return []

    payload = await poll_once(
        plc=_SimplePLC(),
        latest_tag_values={},
        ws=_SimpleWS(),
        alicats=_SimpleAlicats(),
        power_supply=_SimplePower(),
        alarm_engine=_SimpleAlarmEngine(),
        active_alarm_map={},
        estop_active=False,
        historian=fake_sink,
        simulated=False,
    )

    assert payload["data_source"] == "live"
    assert len(fake_sink.submitted) == 1
    assert fake_sink.submitted[0].tags == {"TAG_0": 42.0}
    assert fake_sink.submitted[0].quality == "live"
