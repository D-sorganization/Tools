"""Unit tests for the historian bulk write path.

Runs against in-memory SQLite. Covers the bulk insert, the shared timestamp,
empty-scan handling, and DbC input validation.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

# Backend deps (sqlmodel/sqlalchemy) aren't installed in the shared CI `tests`
# job, so skip this module there rather than erroring on collection.
pytest.importorskip("sqlmodel")

sys.path.insert(0, str(Path(__file__).parent.parent))

from historian import log_scan  # noqa: E402
from models import TagLog  # noqa: E402
from sqlalchemy import StaticPool, func  # noqa: E402
from sqlmodel import Session, SQLModel, col, create_engine, select  # noqa: E402


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


class TestLogScan:
    def test_inserts_one_row_per_tag(self, session: Session) -> None:
        n = log_scan(session, {"TAG_0": 1.0, "TAG_1": 2.5, "TAG_2": -3.0})
        session.commit()
        assert n == 3
        assert session.exec(select(func.count()).select_from(TagLog)).one() == 3

    def test_values_and_names_persisted(self, session: Session) -> None:
        log_scan(session, {"TAG_7": 12.25})
        session.commit()
        row = session.exec(select(TagLog).where(col(TagLog.tag_name) == "TAG_7")).one()
        assert row.value == pytest.approx(12.25)

    def test_shared_timestamp(self, session: Session) -> None:
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        log_scan(session, {"TAG_0": 1.0, "TAG_1": 2.0}, timestamp=ts)
        session.commit()
        stamps = {r.timestamp for r in session.exec(select(TagLog)).all()}
        assert len(stamps) == 1  # every row shares the one scan timestamp

    def test_empty_scan_writes_nothing(self, session: Session) -> None:
        assert log_scan(session, {}) == 0
        session.commit()
        assert session.exec(select(func.count()).select_from(TagLog)).one() == 0

    def test_coerces_numeric_strings(self, session: Session) -> None:
        # float("3.5") works — the value is coerced, not rejected.
        assert log_scan(session, {"TAG_0": "3.5"}) == 1

    def test_rejects_non_session(self) -> None:
        with pytest.raises(TypeError):
            log_scan(object(), {"TAG_0": 1.0})

    def test_rejects_non_dict_tags(self, session: Session) -> None:
        with pytest.raises(TypeError):
            log_scan(session, [("TAG_0", 1.0)])

    def test_rejects_bad_timestamp(self, session: Session) -> None:
        with pytest.raises(TypeError):
            log_scan(session, {"TAG_0": 1.0}, timestamp="2026")

    def test_rejects_non_numeric_value(self, session: Session) -> None:
        with pytest.raises(ValueError, match="non-numeric"):
            log_scan(session, {"TAG_0": "oops"})


class TestSampleQuality:
    def test_rows_default_to_live_quality(self, session: Session) -> None:
        log_scan(session, {"TAG_0": 1.0})
        session.commit()
        assert session.exec(select(TagLog.quality)).one() == "live"

    def test_quality_is_stamped_on_every_row(self, session: Session) -> None:
        """#4004: a bench run must be distinguishable from a real measurement."""
        log_scan(session, {"TAG_0": 1.0, "TAG_1": 2.0}, quality="simulated")
        session.commit()
        assert set(session.exec(select(TagLog.quality)).all()) == {"simulated"}

    def test_quality_must_be_a_non_empty_string(self, session: Session) -> None:
        with pytest.raises(ValueError):
            log_scan(session, {"TAG_0": 1.0}, quality="  ")
        with pytest.raises(TypeError):
            log_scan(session, {"TAG_0": 1.0}, quality=None)
