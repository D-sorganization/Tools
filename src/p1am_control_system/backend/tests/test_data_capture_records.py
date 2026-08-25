"""Unit tests for the historian data-capture footprint, size capping, and retention.

Covers database footprint estimation, size capping and retention policies,
off-loop background retention worker, and incremental vacuum page reclamation.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import sys
import threading
import time
from collections.abc import Generator
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017

import data_capture  # noqa: E402
from data_capture import (  # noqa: E402
    TableFootprint,
    capture_stats,
    enforce_size_cap,
    historian_footprint,
)
from models import EventLog, TagLog  # noqa: E402
from settings import P1AMSettings  # noqa: E402
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402


@pytest.fixture
def session() -> Generator[Session, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _fake_footprint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    total: int,
    taglog: int,
    eventlog: int,
) -> None:
    """Pin both the on-disk size and its per-table split for deterministic math."""
    monkeypatch.setattr(data_capture, "_db_size_bytes", lambda: total)
    monkeypatch.setattr(
        data_capture,
        "historian_footprint",
        lambda _session, total_bytes=None: TableFootprint(
            total_bytes=total if total_bytes is None else int(total_bytes),
            taglog_bytes=taglog,
            eventlog_bytes=eventlog,
            measured=True,
        ),
    )


class TestEnforceSizeCap:
    def test_under_cap_is_noop(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(10):
            session.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
        session.commit()
        _fake_footprint(monkeypatch, total=100, taglog=100, eventlog=0)
        result = enforce_size_cap(session, max_bytes=1000)
        assert result.over_cap is False
        assert result.rows_deleted == 0
        assert capture_stats(session).total_rows == 10

    def test_over_cap_purges_oldest_keeps_newest(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        # ids are monotonic with insert order; value encodes age (0=oldest).
        for i in range(10):
            session.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
        session.commit()
        # The whole 1000 B file is taglog: 100 B/row. Cap 500 leaves the tag
        # budget 450 (90% of the cap after the 10% event reserve); 450*0.9=405
        # => keep 4 rows, purge the 6 oldest.
        _fake_footprint(monkeypatch, total=1000, taglog=1000, eventlog=0)
        result = enforce_size_cap(session, max_bytes=500)
        assert result.over_cap is True
        assert result.rows_deleted == 6
        remaining = sorted(r.value for r in session.exec(select(TagLog)).all())
        assert remaining == [6.0, 7.0, 8.0, 9.0]  # newest kept

    def test_event_log_bulk_does_not_wipe_the_tag_historian(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression for #4027: sizing bytes_per_row from the WHOLE file while
        # counting only TagLog rows inflates the estimate ~10x here and deletes
        # the entire tag historian even though TagLog is well inside its budget.
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(10):
            session.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
        session.commit()
        # 1000 B file, of which only 100 B is taglog and 900 B is the event log.
        # Tag budget is 900*... => taglog (100 B) is comfortably under it.
        _fake_footprint(monkeypatch, total=1000, taglog=100, eventlog=900)
        result = enforce_size_cap(session, max_bytes=800)
        assert result.over_cap is True
        assert result.rows_deleted == 0  # taglog is inside its own budget
        assert capture_stats(session).total_rows == 10

    def test_event_log_over_budget_is_purged_oldest_first(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(10):
            session.add(
                EventLog(
                    event_type="ALARM",
                    description=f"e{i}",
                    severity=1,
                    timestamp=base + _dt.timedelta(seconds=i),
                )
            )
        session.commit()
        # Event budget = 10% of 1000 = 100 B; the event log holds 1000 B over
        # 10 rows (100 B/row) so it must shed down to 100*0.9 = 90 B. A size
        # purge always keeps at least the newest row rather than emptying the
        # log. The age pass is disabled here to isolate the size budget.
        _fake_footprint(monkeypatch, total=1000, taglog=0, eventlog=1000)
        result = enforce_size_cap(session, max_bytes=1000, event_max_age_s=None)
        assert result.event_rows_deleted > 0
        survivors = [e.description for e in session.exec(select(EventLog)).all()]
        assert "e0" not in survivors  # oldest went first
        assert "e9" in survivors  # newest retained

    def test_event_log_age_retention_runs_even_when_under_cap(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regression for #4027: EventLog had NO automatic retention at all.
        now = _dt.datetime(2026, 6, 1, tzinfo=UTC)
        session.add(
            EventLog(
                event_type="SYSTEM",
                description="ancient",
                timestamp=now - _dt.timedelta(days=400),
            )
        )
        session.add(
            EventLog(
                event_type="SYSTEM",
                description="recent",
                timestamp=now - _dt.timedelta(days=1),
            )
        )
        session.commit()
        _fake_footprint(monkeypatch, total=10, taglog=5, eventlog=5)
        result = enforce_size_cap(
            session,
            max_bytes=1_000_000,
            event_max_age_s=30 * 86400.0,
            now=now,
        )
        assert result.over_cap is False
        assert result.event_rows_deleted == 1
        survivors = [e.description for e in session.exec(select(EventLog)).all()]
        assert survivors == ["recent"]

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            enforce_size_cap(object(), max_bytes=1000)

    def test_rejects_nonpositive_max_bytes(self, session: Session) -> None:
        with pytest.raises(ValueError):
            enforce_size_cap(session, max_bytes=0)

    def test_rejects_bad_headroom(self, session: Session) -> None:
        with pytest.raises(ValueError):
            enforce_size_cap(session, max_bytes=1000, headroom=1.5)

    def test_rejects_bad_event_max_age(self, session: Session) -> None:
        with pytest.raises(ValueError):
            enforce_size_cap(session, max_bytes=1000, event_max_age_s=-1.0)
        with pytest.raises(TypeError):
            enforce_size_cap(session, max_bytes=1000, event_max_age_s="30d")


class TestHistorianFootprint:
    def test_splits_taglog_and_eventlog(self, session: Session) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(200):
            session.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
        session.add(EventLog(event_type="ALARM", description="x", timestamp=base))
        session.commit()
        fp = historian_footprint(session)
        assert isinstance(fp, TableFootprint)
        assert fp.taglog_bytes > 0
        # The tag historian must be attributed strictly less than the whole DB —
        # otherwise the size cap over-charges TagLog for other tables' pages.
        assert fp.taglog_bytes < fp.total_bytes
        assert fp.eventlog_bytes >= 0

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            historian_footprint(object())


class TestHistorianRetentionLoop:
    """#4006: the periodic sweep must never run on the asyncio event loop."""

    @staticmethod
    def _settings() -> P1AMSettings:
        return P1AMSettings(historian_max_bytes=1024, historian_retention_interval_s=60)

    def test_sweep_runs_on_a_worker_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, int] = {}

        def _sweep(engine: object, **_kw: object) -> None:
            seen["thread"] = threading.get_ident()

        monkeypatch.setattr(data_capture, "run_retention_sweep", _sweep)

        async def _drive() -> int:
            shutdown = asyncio.Event()
            task = asyncio.create_task(
                data_capture.historian_retention_loop(
                    shutdown_event=shutdown,
                    engine=object(),
                    logger=logging.getLogger("test.retention"),
                    interval_s=0.01,
                    settings=self._settings(),
                )
            )
            while "thread" not in seen:
                await asyncio.sleep(0.01)
            shutdown.set()
            task.cancel()
            return threading.get_ident()

        loop_thread = asyncio.run(_drive())
        assert seen["thread"] != loop_thread

    def test_slow_sweep_does_not_stall_the_event_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _slow_sweep(engine: object, **_kw: object) -> None:
            time.sleep(0.30)  # stands in for VACUUM on a 1 GiB SD-card DB

        monkeypatch.setattr(data_capture, "run_retention_sweep", _slow_sweep)

        async def _drive() -> int:
            shutdown = asyncio.Event()
            ticks = 0

            async def _heartbeat() -> None:
                nonlocal ticks
                while not shutdown.is_set():
                    await asyncio.sleep(0.005)
                    ticks += 1

            beat = asyncio.create_task(_heartbeat())
            task = asyncio.create_task(
                data_capture.historian_retention_loop(
                    shutdown_event=shutdown,
                    engine=object(),
                    logger=logging.getLogger("test.retention"),
                    interval_s=0.01,
                    settings=self._settings(),
                )
            )
            await asyncio.sleep(0.35)
            shutdown.set()
            task.cancel()
            beat.cancel()
            return ticks

        # A blocking sweep on the loop thread yields ~3 ticks over the 350 ms
        # window (only the 50 ms outside the freeze); an offloaded one keeps
        # servicing coroutines throughout — E-stop included. The threshold is
        # well clear of both, allowing for Windows' ~15 ms timer granularity.
        assert asyncio.run(_drive()) >= 10

    def test_disabled_when_cap_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def _sweep(engine: object, **_kw: object) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(data_capture, "run_retention_sweep", _sweep)

        async def _drive() -> None:
            shutdown = asyncio.Event()
            await asyncio.wait_for(
                data_capture.historian_retention_loop(
                    shutdown_event=shutdown,
                    engine=object(),
                    logger=logging.getLogger("test.retention"),
                    interval_s=0.01,
                    settings=P1AMSettings(historian_max_bytes=0),
                ),
                timeout=1.0,
            )

        asyncio.run(_drive())
        assert called is False

    def test_sweep_failure_is_logged_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(engine: object, **_kw: object) -> None:
            raise RuntimeError("disk gone")

        monkeypatch.setattr(data_capture, "run_retention_sweep", _boom)
        records: list[str] = []

        class _Logger(logging.Logger):
            def error(self, msg: object, *args: object, **kw: object) -> None:
                records.append(str(msg) % args if args else str(msg))

        async def _drive() -> None:
            shutdown = asyncio.Event()
            task = asyncio.create_task(
                data_capture.historian_retention_loop(
                    shutdown_event=shutdown,
                    engine=object(),
                    logger=_Logger("t"),
                    interval_s=0.01,
                    settings=self._settings(),
                )
            )
            while not records:
                await asyncio.sleep(0.01)
            shutdown.set()
            task.cancel()

        asyncio.run(asyncio.wait_for(_drive(), timeout=2.0))
        assert any("disk gone" in r for r in records)


class TestReclaimFreePages:
    def test_incremental_vacuum_is_bounded_and_best_effort(
        self, tmp_path: Path
    ) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'vac.db'}")
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA auto_vacuum=INCREMENTAL")
        SQLModel.metadata.create_all(engine)
        with Session(engine) as s:
            base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
            for i in range(2000):
                s.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
            s.commit()
            s.exec(select(TagLog))  # keep the session warm
        with Session(engine) as s:
            s.connection().exec_driver_sql("DELETE FROM taglog")
            s.commit()
        with engine.connect() as conn:
            free_before = int(conn.exec_driver_sql("PRAGMA freelist_count").scalar())
        assert free_before > 8, "seed did not create enough free pages to bound"

        reclaimed = data_capture._reclaim_free_pages(engine, max_pages=8)
        # Bounded: never reclaims more than the caller allowed, however many
        # free pages are waiting. This is the whole point over a full VACUUM.
        assert reclaimed == 8
        with engine.connect() as conn:
            free_after = int(conn.exec_driver_sql("PRAGMA freelist_count").scalar())
        assert free_after == free_before - 8

    def test_rejects_bad_max_pages(self, tmp_path: Path) -> None:
        engine = create_engine(f"sqlite:///{tmp_path / 'vac2.db'}")
        with pytest.raises(ValueError):
            data_capture._reclaim_free_pages(engine, max_pages=0)
        with pytest.raises(TypeError):
            data_capture._reclaim_free_pages(engine, max_pages="lots")
