"""Unit tests for the historian data-capture service.

Runs against an in-memory SQLite database so no live PLC or on-disk file is
needed. Covers stats on empty/populated historians, clear-all vs clear-events
vs rolling purge, and the DbC input validation.
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
    TRENDS_MAX_MAX_POINTS,
    TRENDS_MIN_MAX_POINTS,
    CaptureStats,
    capture_stats,
    clear_capture,
    enforce_size_cap,
    parse_query_bound,
    parse_tag_names,
    query_trend_series,
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


class TestEnforceSizeCap:
    def test_under_cap_is_noop(
        self, session: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(10):
            session.add(TagLog(tag_name="TAG_0", value=float(i), timestamp=base))
        session.commit()
        monkeypatch.setattr(data_capture, "_db_size_bytes", lambda: 100)
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
        # 1000 bytes / 10 rows = 100 B/row; cap 500 * 0.9 headroom => keep 4.
        monkeypatch.setattr(data_capture, "_db_size_bytes", lambda: 1000)
        result = enforce_size_cap(session, max_bytes=500)
        assert result.over_cap is True
        assert result.rows_deleted == 6
        remaining = sorted(r.value for r in session.exec(select(TagLog)).all())
        assert remaining == [6.0, 7.0, 8.0, 9.0]  # newest kept

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            enforce_size_cap(object(), max_bytes=1000)

    def test_rejects_nonpositive_max_bytes(self, session: Session) -> None:
        with pytest.raises(ValueError):
            enforce_size_cap(session, max_bytes=0)

    def test_rejects_bad_headroom(self, session: Session) -> None:
        with pytest.raises(ValueError):
            enforce_size_cap(session, max_bytes=1000, headroom=1.5)


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
            thr.set_interval_s("nope")  # type: ignore[arg-type]
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


def _bare(dt: _dt.datetime) -> _dt.datetime:
    """Drop tzinfo for comparison — SQLite round-trips datetimes tz-naive."""
    return dt.replace(tzinfo=None)


def _seed_series(
    s: Session,
    *,
    tag_name: str,
    base: _dt.datetime,
    n: int,
    step_s: float = 1.0,
    reverse: bool = False,
) -> tuple[_dt.datetime, _dt.datetime]:
    """Insert ``n`` evenly-spaced rows for ``tag_name`` and return (start, end).

    ``value`` encodes the sample index so ascending-order assertions can verify
    which physical row was picked. ``reverse`` inserts newest-first to prove the
    query re-sorts by timestamp rather than trusting insertion order.
    """
    order = range(n - 1, -1, -1) if reverse else range(n)
    for i in order:
        s.add(
            TagLog(
                tag_name=tag_name,
                value=float(i),
                timestamp=base + _dt.timedelta(seconds=i * step_s),
            )
        )
    s.commit()
    return base, base + _dt.timedelta(seconds=(n - 1) * step_s)


class TestQueryTrendSeries:
    _BASE = _dt.datetime(2026, 1, 1, tzinfo=UTC)

    def test_small_range_returns_everything_ascending_untruncated(
        self, session: Session
    ) -> None:
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=5)
        timestamps, values, truncated = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end
        )
        assert truncated is False
        assert len(timestamps) == 5
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0]  # ascending by time
        assert timestamps == sorted(timestamps)
        assert _bare(timestamps[0]) == _bare(start)
        assert _bare(timestamps[-1]) == _bare(end)

    def test_reversed_insert_still_returns_ascending(self, session: Session) -> None:
        start, end = _seed_series(
            session, tag_name="TAG_0", base=self._BASE, n=6, reverse=True
        )
        timestamps, values, _ = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end
        )
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert timestamps == sorted(timestamps)

    def test_empty_range_returns_empty_untruncated(self, session: Session) -> None:
        timestamps, values, truncated = query_trend_series(
            session,
            tag_name="TAG_0",
            start=self._BASE,
            end=self._BASE + _dt.timedelta(hours=1),
        )
        assert timestamps == []
        assert values == []
        assert truncated is False

    def test_large_range_spans_whole_window_not_newest_slice(
        self, session: Session
    ) -> None:
        # Regression guard for the DESC+LIMIT bug: 1000 rows over ~1000 s,
        # decimated to 100. A newest-slice implementation would return points
        # clustered at the END (value ~900..999); a correct decimation spans the
        # WHOLE window, so the first point must be at/near the start.
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=1000)
        timestamps, values, truncated = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end, max_points=100
        )
        assert truncated is True
        assert 50 <= len(timestamps) <= 101  # ~max_points (+1 forced endpoint)
        assert timestamps == sorted(timestamps)  # ascending, de-duplicated stride
        # The whole span is covered: first sample at the range start, last at end.
        assert _bare(timestamps[0]) == _bare(start)
        assert values[0] == 0.0
        assert _bare(timestamps[-1]) == _bare(end)
        assert values[-1] == 999.0
        # The series must reach deep into the FIRST half — impossible if it were
        # clipped to only the newest ~100 rows.
        assert values[1] < 100.0

    def test_five_x_max_points_is_truncated_and_spans(self, session: Session) -> None:
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=500)
        timestamps, values, truncated = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end, max_points=100
        )
        assert truncated is True
        assert len(timestamps) <= 101
        assert _bare(timestamps[0]) == _bare(start)
        assert _bare(timestamps[-1]) == _bare(end)

    def test_exact_max_points_returns_all_untruncated(self, session: Session) -> None:
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=100)
        timestamps, _, truncated = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end, max_points=100
        )
        assert truncated is False
        assert len(timestamps) == 100

    def test_one_over_max_points_is_truncated(self, session: Session) -> None:
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=101)
        _, _, truncated = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end, max_points=100
        )
        assert truncated is True

    def test_only_selected_tag_is_returned(self, session: Session) -> None:
        start, end = _seed_series(session, tag_name="TAG_0", base=self._BASE, n=4)
        _seed_series(session, tag_name="TAG_1", base=self._BASE, n=4)
        _, values, _ = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end
        )
        assert len(values) == 4  # TAG_1 rows excluded by the filter

    # -- DbC ---------------------------------------------------------------- #

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            query_trend_series(
                object(),
                tag_name="TAG_0",
                start=self._BASE,
                end=self._BASE,
            )

    def test_rejects_non_str_tag_name(self, session: Session) -> None:
        with pytest.raises(TypeError):
            query_trend_series(
                session,
                tag_name=123,
                start=self._BASE,
                end=self._BASE,
            )

    def test_rejects_non_datetime_bounds(self, session: Session) -> None:
        with pytest.raises(TypeError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start="2026-01-01",
                end=self._BASE,
            )
        with pytest.raises(TypeError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start=self._BASE,
                end="2026-01-01",
            )

    def test_rejects_bool_max_points(self, session: Session) -> None:
        # bool is an int subclass; it must not slip through as max_points=1.
        with pytest.raises(TypeError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start=self._BASE,
                end=self._BASE,
                max_points=True,
            )

    def test_rejects_out_of_range_max_points(self, session: Session) -> None:
        with pytest.raises(ValueError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start=self._BASE,
                end=self._BASE,
                max_points=TRENDS_MIN_MAX_POINTS - 1,
            )
        with pytest.raises(ValueError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start=self._BASE,
                end=self._BASE,
                max_points=TRENDS_MAX_MAX_POINTS + 1,
            )

    def test_rejects_start_after_end(self, session: Session) -> None:
        with pytest.raises(ValueError):
            query_trend_series(
                session,
                tag_name="TAG_0",
                start=self._BASE + _dt.timedelta(hours=1),
                end=self._BASE,
            )
