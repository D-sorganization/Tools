"""Off-loop, batched, retrying historian writer (issue #4023).

The scan loop must not open a SQLite session, insert and commit on the event
loop. Tag samples are resamplable and may be dropped under pressure; alarm
transitions are not and must survive a ``VACUUM`` lock.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from poll_runtime import HistorianRecord, HistorianWriter  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402


class _Event:
    def __init__(self, description: str) -> None:
        self.description = description


class _Session:
    """Minimal session double recording the transaction shape."""

    def __init__(self, *, fail_times: int = 0, fail_with: type = OperationalError):
        self.added: list[Any] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self._fail_times = fail_times
        self._fail_with = fail_with

    def add(self, row: Any) -> None:
        self.added.append(row)

    def commit(self) -> None:
        if self._fail_times > 0:
            self._fail_times -= 1
            raise self._fail_with("database is locked", None, None)
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class _Factory:
    """Session factory that hands out a fresh generator per attempt."""

    def __init__(self, *sessions: _Session) -> None:
        self._sessions = list(sessions)
        self.opened = 0

    def __call__(self) -> Iterator[_Session]:
        self.opened += 1
        session = self._sessions[min(self.opened - 1, len(self._sessions) - 1)]
        yield session


async def _immediate(func: Callable[..., Any], *args: Any) -> Any:
    """``asyncio.to_thread`` stand-in that runs inline for deterministic tests."""
    return func(*args)


def _writer(factory: _Factory, **kwargs: Any) -> HistorianWriter:
    scans: list[dict[str, Any]] = []

    def fake_log_scan(
        _session: Any, tags: dict[str, float], *, quality: str = "live"
    ) -> int:
        scans.append({"quality": quality, **tags})
        return len(tags)

    kwargs.setdefault("log_scan", fake_log_scan)
    kwargs.setdefault("to_thread", _immediate)
    kwargs.setdefault("sleep", lambda _s: None)
    writer = HistorianWriter(factory, **kwargs)
    writer.test_scans = scans
    return writer


@pytest.mark.asyncio
async def test_batches_several_scans_into_one_transaction() -> None:
    session = _Session()
    factory = _Factory(session)
    writer = _writer(factory)

    for i in range(4):
        writer.submit(
            HistorianRecord(tags={f"TAG_{i}": float(i)}, events=(), quality="live")
        )

    written = await writer.drain_once()

    assert written == 4
    assert factory.opened == 1, "one session for the whole batch"
    assert session.commits == 1
    assert len(writer.test_scans) == 4


@pytest.mark.asyncio
async def test_no_session_is_opened_when_nothing_is_queued() -> None:
    factory = _Factory(_Session())
    writer = _writer(factory)

    assert await writer.drain_once() == 0
    assert factory.opened == 0


@pytest.mark.asyncio
async def test_full_queue_drops_oldest_samples_but_carries_alarms_forward() -> None:
    factory = _Factory(_Session())
    writer = _writer(factory, queue_size=2)
    alarm = _Event("HiHi on TAG_3")

    writer.submit(HistorianRecord(tags={"TAG_0": 0.0}, events=(alarm,), quality="live"))
    writer.submit(HistorianRecord(tags={"TAG_1": 1.0}, events=(), quality="live"))
    # Queue is full: the oldest record is evicted, but its alarm event rides
    # along on the newcomer — an alarm transition is not resamplable.
    writer.submit(HistorianRecord(tags={"TAG_2": 2.0}, events=(), quality="live"))

    assert writer.dropped_samples == 1
    await writer.drain_once()
    assert alarm in factory._sessions[0].added


@pytest.mark.asyncio
async def test_operational_error_is_retried_so_alarms_survive_a_vacuum_lock() -> None:
    locked_then_ok = _Session(fail_times=1)
    factory = _Factory(locked_then_ok)
    writer = _writer(factory, retry_attempts=3)
    alarm = _Event("HiHi on TAG_3")

    writer.submit(HistorianRecord(tags=None, events=(alarm,), quality="live"))
    await writer.drain_once()

    assert locked_then_ok.commits == 1
    assert locked_then_ok.rollbacks == 1
    assert writer.write_failures == 0


@pytest.mark.asyncio
async def test_persistent_failure_is_counted_not_silently_swallowed() -> None:
    always_locked = _Session(fail_times=99)
    factory = _Factory(always_locked)
    writer = _writer(factory, retry_attempts=2)

    writer.submit(HistorianRecord(tags=None, events=(_Event("HiHi"),), quality="live"))
    await writer.drain_once()

    assert writer.write_failures == 1
    assert always_locked.commits == 0
    assert always_locked.rollbacks == 2


@pytest.mark.asyncio
async def test_run_drains_until_stopped() -> None:
    session = _Session()
    writer = _writer(_Factory(session), poll_timeout_s=0.01)
    stop = asyncio.Event()

    task = asyncio.create_task(writer.run(stop))
    writer.submit(HistorianRecord(tags={"TAG_0": 1.0}, events=(), quality="live"))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert session.commits >= 1


class TestContracts:
    def test_record_rejects_a_non_mapping_tags(self) -> None:
        with pytest.raises(TypeError):
            HistorianRecord(tags=[("TAG_0", 1.0)], events=(), quality="live")

    def test_record_rejects_a_blank_quality(self) -> None:
        with pytest.raises(ValueError):
            HistorianRecord(tags=None, events=(), quality="")

    def test_writer_rejects_a_non_callable_factory(self) -> None:
        with pytest.raises(TypeError):
            HistorianWriter(object())

    def test_writer_rejects_a_non_positive_queue(self) -> None:
        with pytest.raises(ValueError):
            HistorianWriter(lambda: iter(()), queue_size=0)

    def test_submit_rejects_a_foreign_object(self) -> None:
        writer = _writer(_Factory(_Session()))
        with pytest.raises(TypeError):
            writer.submit({"tags": {}})


class TestThrottledSink:
    def _sink(self, due: bool) -> tuple[Any, list[Any]]:
        from poll_runtime import ThrottledHistorianSink

        seen: list[Any] = []
        writer = type("W", (), {"submit": lambda _self, r: seen.append(r)})()
        return ThrottledHistorianSink(writer, lambda: due), seen

    def test_sample_is_forwarded_when_due(self) -> None:
        sink, seen = self._sink(True)
        sink.submit(HistorianRecord(tags={"TAG_0": 1.0}, events=(), quality="live"))
        assert seen[0].tags == {"TAG_0": 1.0}

    def test_sample_is_dropped_but_events_survive_when_not_due(self) -> None:
        sink, seen = self._sink(False)
        alarm = _Event("HiHi")
        sink.submit(
            HistorianRecord(tags={"TAG_0": 1.0}, events=(alarm,), quality="live")
        )
        assert seen[0].tags is None
        assert seen[0].events == (alarm,)

    def test_nothing_is_queued_when_nothing_is_due(self) -> None:
        """#4023: a suppressed scan must not even open a session."""
        sink, seen = self._sink(False)
        sink.submit(HistorianRecord(tags={"TAG_0": 1.0}, events=(), quality="live"))
        assert seen == []

    def test_rejects_a_writer_without_a_submit_seam(self) -> None:
        from poll_runtime import ThrottledHistorianSink

        with pytest.raises(TypeError):
            ThrottledHistorianSink(object(), lambda: True)

    def test_rejects_a_non_callable_throttle(self) -> None:
        from poll_runtime import ThrottledHistorianSink

        with pytest.raises(TypeError):
            ThrottledHistorianSink(type("W", (), {"submit": lambda s, r: None})(), True)


def test_loop_diagnostics_reports_every_health_counter() -> None:
    from poll_runtime import loop_diagnostics

    diag = loop_diagnostics(
        scheduler=type(
            "S", (), {"period_s": 0.1, "overrun_count": 2, "last_overrun_s": 0.05}
        )(),
        perf=type("P", (), {"broadcast_every_n": 20})(),
        writer=type("W", (), {"write_failures": 1, "dropped_samples": 3})(),
        ws=type("C", (), {"frames_dropped": 4})(),
    )

    assert diag == {
        "scan_interval_s": 0.1,
        "scan_overruns": 2,
        "last_overrun_s": 0.05,
        "broadcast_every_n": 20,
        "historian_write_failures": 1,
        "historian_samples_dropped": 3,
        "ws_frames_dropped": 4,
    }
