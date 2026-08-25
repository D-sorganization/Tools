"""Unit tests for the historian query and timezone handling.

Covers trend series query decimation, limit bounding, UTC timezone
normalization, and offset preservation across models and endpoints.
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

from data_capture import (  # noqa: E402
    TRENDS_MAX_MAX_POINTS,
    TRENDS_MIN_MAX_POINTS,
    capture_stats,
    query_trend_series,
    stream_tag_export_csv,
)
from models import EventLog, TagLog  # noqa: E402
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


def _assert_aware_utc(value: _dt.datetime) -> None:
    """Every historian datetime crossing the ORM boundary must be aware UTC."""
    assert value.tzinfo is not None, f"{value!r} is tz-naive"
    assert value.utcoffset() == _dt.timedelta(0), f"{value!r} is not UTC"


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
        assert timestamps[0] == start
        assert timestamps[-1] == end

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
        assert timestamps[0] == start
        assert values[0] == 0.0
        assert timestamps[-1] == end
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
        assert timestamps[0] == start
        assert timestamps[-1] == end

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


class TestTimestampTimezone:
    """#4025: historian datetimes must round-trip as aware UTC, not naive."""

    def test_taglog_round_trips_aware_utc(self, session: Session) -> None:
        base = _dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.0, timestamp=base))
        session.commit()
        session.expunge_all()
        row = session.exec(select(TagLog)).one()
        _assert_aware_utc(row.timestamp)
        assert row.timestamp == base

    def test_eventlog_round_trips_aware_utc(self, session: Session) -> None:
        base = _dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        session.add(EventLog(event_type="ALARM", description="x", timestamp=base))
        session.commit()
        session.expunge_all()
        row = session.exec(select(EventLog)).one()
        _assert_aware_utc(row.timestamp)
        assert row.timestamp == base

    def test_non_utc_offset_is_normalized_on_write(self, session: Session) -> None:
        # A UTC-7 host writing 05:00-07:00 means 12:00Z; the naive-bind bug
        # stored the local wall clock (05:00) and read it back as 05:00Z.
        local = _dt.timezone(_dt.timedelta(hours=-7))
        session.add(
            TagLog(
                tag_name="TAG_0",
                value=1.0,
                timestamp=_dt.datetime(2026, 1, 1, 5, 0, tzinfo=local),
            )
        )
        session.commit()
        session.expunge_all()
        row = session.exec(select(TagLog)).one()
        _assert_aware_utc(row.timestamp)
        assert row.timestamp == _dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def test_naive_write_is_read_back_as_aware_utc(self, session: Session) -> None:
        session.add(
            TagLog(
                tag_name="TAG_0",
                value=1.0,
                timestamp=_dt.datetime(2026, 1, 1, 12, 0),
            )
        )
        session.commit()
        session.expunge_all()
        row = session.exec(select(TagLog)).one()
        _assert_aware_utc(row.timestamp)
        assert row.timestamp == _dt.datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def test_range_filter_honours_the_bound_offset(self, session: Session) -> None:
        # Rows at 12:00Z and 20:00Z. A bound of 13:00-07:00 == 20:00Z must keep
        # only the later row; a tz-dropping bind would compare against 13:00.
        for hour in (12, 20):
            session.add(
                TagLog(
                    tag_name="TAG_0",
                    value=float(hour),
                    timestamp=_dt.datetime(2026, 1, 1, hour, tzinfo=UTC),
                )
            )
        session.commit()
        local = _dt.timezone(_dt.timedelta(hours=-7))
        _, values, _ = query_trend_series(
            session,
            tag_name="TAG_0",
            start=_dt.datetime(2026, 1, 1, 13, 0, tzinfo=local),
            end=_dt.datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
        )
        assert values == [20.0]

    def test_capture_stats_emits_explicit_offset(self, session: Session) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.0, timestamp=base))
        session.commit()
        stats = capture_stats(session)
        assert stats.oldest_timestamp is not None
        assert stats.newest_timestamp is not None
        # An offset-less string is re-parsed as LOCAL time by the browser, which
        # is exactly how the export window drifted by the host's UTC offset.
        assert stats.oldest_timestamp.endswith("+00:00")
        assert stats.newest_timestamp.endswith("+00:00")

    def test_export_csv_emits_explicit_offset(self, session: Session) -> None:
        base = _dt.datetime(2026, 1, 1, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.5, timestamp=base))
        session.commit()
        statement = select(TagLog)
        rows = list(stream_tag_export_csv(session.get_bind(), statement))
        assert "+00:00" in rows[1]

    def test_query_trend_series_returns_aware_timestamps(
        self, session: Session
    ) -> None:
        start, end = _seed_series(
            session, tag_name="TAG_0", base=_dt.datetime(2026, 1, 1, tzinfo=UTC), n=3
        )
        timestamps, _, _ = query_trend_series(
            session, tag_name="TAG_0", start=start, end=end
        )
        for ts in timestamps:
            _assert_aware_utc(ts)
