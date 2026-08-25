"""Unit tests for the historian data-capture core helpers and stats.

Runs against an in-memory SQLite database. Covers stats on empty/populated
historians, clear-all vs clear-events vs rolling purge, export query parsing,
and the monotonic capture throttle.
"""

from __future__ import annotations

import datetime as _dt
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017

import data_capture  # noqa: E402
from data_capture import (  # noqa: E402
    CaptureStats,
    capture_stats,
    clear_capture,
    parse_query_bound,
    parse_tag_names,
    stream_tag_export_csv,
)
from models import EventLog, TagLog  # noqa: E402
from pydantic import ValidationError  # noqa: E402
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


def _seed(s: Session, *, base: _dt.datetime, n: int = 5) -> None:
    for i in range(n):
        s.add(TagLog(tag_name=f"TAG_{i % 2}", value=float(i), timestamp=base))
    s.add(EventLog(event_type="ALARM", description="x", severity=2, timestamp=base))
    s.commit()


class TestCaptureStats:
    def test_empty_historian(self, session: Session) -> None:
        stats = capture_stats(session)
        assert isinstance(stats, CaptureStats)
        assert stats.total_rows == 0
        assert stats.distinct_tags == 0
        assert stats.oldest_timestamp is None
        assert stats.newest_timestamp is None
        assert stats.span_seconds == 0.0
        assert stats.capturing is True

    def test_counts_and_span(self, session: Session) -> None:
        t0 = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(4):
            session.add(
                TagLog(
                    tag_name=f"TAG_{i % 2}",
                    value=float(i),
                    timestamp=t0 + _dt.timedelta(seconds=i * 10),
                )
            )
        session.commit()
        stats = capture_stats(session)
        assert stats.total_rows == 4
        assert stats.distinct_tags == 2  # TAG_0, TAG_1
        assert stats.span_seconds == pytest.approx(30.0)  # 0..30 s

    def test_capturing_flag_passthrough(self, session: Session) -> None:
        assert capture_stats(session, capturing=False).capturing is False

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            capture_stats(object())

    def test_rejects_non_bool_capturing(self, session: Session) -> None:
        with pytest.raises(TypeError):
            capture_stats(session, capturing="yes")


class TestClearCapture:
    def test_clear_all_tags_keeps_events_by_default(self, session: Session) -> None:
        _seed(session, base=_dt.datetime(2026, 1, 1, tzinfo=UTC), n=5)
        result = clear_capture(session)
        assert result.tag_rows_deleted == 5
        assert result.event_rows_deleted == 0
        assert capture_stats(session).total_rows == 0
        assert capture_stats(session).event_rows == 1  # event kept

    def test_clear_includes_events_when_requested(self, session: Session) -> None:
        _seed(session, base=_dt.datetime(2026, 1, 1, tzinfo=UTC), n=3)
        result = clear_capture(session, include_events=True)
        assert result.tag_rows_deleted == 3
        assert result.event_rows_deleted == 1
        assert capture_stats(session).event_rows == 0

    def test_rolling_purge_before_only_deletes_older(self, session: Session) -> None:
        old = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        new = _dt.datetime(2026, 1, 2, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.0, timestamp=old))
        session.add(TagLog(tag_name="TAG_0", value=2.0, timestamp=new))
        session.commit()
        result = clear_capture(
            session,
            before=_dt.datetime(2026, 1, 1, 12, tzinfo=UTC),
        )
        assert result.tag_rows_deleted == 1
        assert capture_stats(session).total_rows == 1  # the newer row survives

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            clear_capture(object())

    def test_rejects_non_bool_include_events(self, session: Session) -> None:
        with pytest.raises(TypeError):
            clear_capture(session, include_events=1)

    def test_rejects_bad_before_type(self, session: Session) -> None:
        with pytest.raises(TypeError):
            clear_capture(session, before="2026-01-01")


class TestParseQueryBound:
    def test_z_suffix_is_utc(self) -> None:
        dt = parse_query_bound("2026-01-01T00:00:00Z")
        assert dt.tzinfo is not None
        assert dt.utcoffset() == _dt.timedelta(0)

    def test_explicit_offset_normalized_to_utc(self) -> None:
        # 02:00 at +02:00 == 00:00 UTC
        dt = parse_query_bound("2026-01-01T02:00:00+02:00")
        assert dt.hour == 0 and dt.utcoffset() == _dt.timedelta(0)

    def test_naive_assumed_utc(self) -> None:
        dt = parse_query_bound("2026-01-01T12:00:00")
        assert dt.tzinfo is not None
        assert dt.hour == 12 and dt.utcoffset() == _dt.timedelta(0)

    def test_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            parse_query_bound(12345)

    def test_rejects_bad_iso(self) -> None:
        with pytest.raises(ValueError):
            parse_query_bound("not-a-date")


class TestExportHelpers:
    def test_parse_tag_names_normalizes_numeric_ids(self) -> None:
        assert parse_tag_names("1, TAG_A, ,2") == ["TAG_1", "TAG_A", "TAG_2"]

    def test_parse_tag_names_rejects_non_str(self) -> None:
        with pytest.raises(TypeError):
            parse_tag_names(12345)

    def test_stream_tag_export_csv_yields_header_and_rows(
        self, session: Session
    ) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.5, timestamp=base))
        session.add(TagLog(tag_name="TAG_1", value=2.5, timestamp=base))
        session.commit()

        statement = select(TagLog).order_by(TagLog.tag_name)
        rows = list(stream_tag_export_csv(session.get_bind(), statement, chunk_rows=1))

        assert rows[0] == "Timestamp,Tag Name,Value\r\n"
        assert "TAG_0,1.5" in rows[1]
        assert "TAG_1,2.5" in rows[2]


class _FakeClock:
    """Deterministic monotonic clock for throttle tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestCaptureThrottle:
    def test_first_call_is_always_due(self) -> None:
        thr = data_capture.CaptureThrottle(5.0, clock=_FakeClock())
        assert thr.due() is True

    def test_holds_off_until_interval_elapses(self) -> None:
        clk = _FakeClock()
        thr = data_capture.CaptureThrottle(5.0, clock=clk)
        assert thr.due() is True  # t=0 -> log
        clk.t = 3.0
        assert thr.due() is False  # 3 s < 5 s
        clk.t = 5.0
        assert thr.due() is True  # 5 s >= 5 s (inclusive) -> log
        clk.t = 9.0
        assert thr.due() is False  # 4 s since last
        clk.t = 10.0
        assert thr.due() is True  # 5 s since last -> log

    def test_zero_interval_logs_every_call(self) -> None:
        thr = data_capture.CaptureThrottle(0.0, clock=_FakeClock())
        assert [thr.due(), thr.due(), thr.due()] == [True, True, True]

    def test_set_interval_takes_effect(self) -> None:
        clk = _FakeClock()
        thr = data_capture.CaptureThrottle(5.0, clock=clk)
        assert thr.due() is True
        thr.set_interval_s(1.0)
        assert thr.interval_s == 1.0
        clk.t = 1.0
        assert thr.due() is True

    def test_set_interval_validates(self) -> None:
        thr = data_capture.CaptureThrottle(5.0)
        with pytest.raises(TypeError):
            thr.set_interval_s("nope")
        with pytest.raises(TypeError):
            thr.set_interval_s(True)  # bool is not an accepted numeric
        with pytest.raises(ValueError):
            thr.set_interval_s(-1.0)
        with pytest.raises(ValueError):
            thr.set_interval_s(float("inf"))
        with pytest.raises(ValueError):
            thr.set_interval_s(float("nan"))

    def test_config_model_validates_bounds(self) -> None:
        assert data_capture.CaptureConfig(interval_s=5.0).interval_s == 5.0
        with pytest.raises(ValidationError):
            data_capture.CaptureConfig(interval_s=-1.0)
        with pytest.raises(ValidationError):
            data_capture.CaptureConfig(interval_s=10_000.0)  # exceeds le=3600
