"""Unit tests for the historian data-capture service.

Runs against an in-memory SQLite database so no live PLC or on-disk file is
needed. Covers stats on empty/populated historians, clear-all vs clear-events
vs rolling purge, and the DbC input validation.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_capture import (  # noqa: E402
    CaptureStats,
    capture_stats,
    clear_capture,
)
from models import EventLog, TagLog  # noqa: E402
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402


@pytest.fixture
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed(s: Session, *, base: datetime, n: int = 5) -> None:
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
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        for i in range(4):
            session.add(
                TagLog(
                    tag_name=f"TAG_{i % 2}",
                    value=float(i),
                    timestamp=t0 + timedelta(seconds=i * 10),
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
            capture_stats(object())  # type: ignore[arg-type]

    def test_rejects_non_bool_capturing(self, session: Session) -> None:
        with pytest.raises(TypeError):
            capture_stats(session, capturing="yes")  # type: ignore[arg-type]


class TestClearCapture:
    def test_clear_all_tags_keeps_events_by_default(self, session: Session) -> None:
        _seed(session, base=datetime(2026, 1, 1, tzinfo=UTC), n=5)
        result = clear_capture(session)
        assert result.tag_rows_deleted == 5
        assert result.event_rows_deleted == 0
        assert capture_stats(session).total_rows == 0
        assert capture_stats(session).event_rows == 1  # event kept

    def test_clear_includes_events_when_requested(self, session: Session) -> None:
        _seed(session, base=datetime(2026, 1, 1, tzinfo=UTC), n=3)
        result = clear_capture(session, include_events=True)
        assert result.tag_rows_deleted == 3
        assert result.event_rows_deleted == 1
        assert capture_stats(session).event_rows == 0

    def test_rolling_purge_before_only_deletes_older(self, session: Session) -> None:
        old = datetime(2026, 1, 1, tzinfo=UTC)
        new = datetime(2026, 1, 2, tzinfo=UTC)
        session.add(TagLog(tag_name="TAG_0", value=1.0, timestamp=old))
        session.add(TagLog(tag_name="TAG_0", value=2.0, timestamp=new))
        session.commit()
        result = clear_capture(session, before=datetime(2026, 1, 1, 12, tzinfo=UTC))
        assert result.tag_rows_deleted == 1
        assert capture_stats(session).total_rows == 1  # the newer row survives

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            clear_capture(object())  # type: ignore[arg-type]

    def test_rejects_non_bool_include_events(self, session: Session) -> None:
        with pytest.raises(TypeError):
            clear_capture(session, include_events=1)  # type: ignore[arg-type]

    def test_rejects_bad_before_type(self, session: Session) -> None:
        with pytest.raises(TypeError):
            clear_capture(session, before="2026-01-01")  # type: ignore[arg-type]
