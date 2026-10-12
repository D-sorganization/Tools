"""Stored sample quality in ``/api/trends`` and the CSV export (issue #5529).

``TagLog.quality`` records each historian sample's provenance. The trends JSON
gains an additive ``quality`` array parallel to ``values`` and the CSV export
gains an appended ``Quality`` column; NULL / unrecognised stored values surface
as the explicit ``unknown`` marker rather than being dropped or guessed.
"""

from __future__ import annotations

import datetime as _dt
import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

os.environ.setdefault("PLC_DRIVER", "modbus")

sys.path.insert(0, str(Path(__file__).parent.parent))

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017

import main  # noqa: E402
from data_capture import (  # noqa: E402
    QUALITY_UNKNOWN,
    normalize_quality,
    query_trend_samples,
    query_trend_series,
    stream_tag_export_csv,
)
from models import (  # noqa: E402
    DATA_SOURCE_FAULT,
    DATA_SOURCE_HELD,
    DATA_SOURCE_LIVE,
    DATA_SOURCE_SIMULATED,
    TagLog,
)
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine, select  # noqa: E402

_BASE = _dt.datetime(2026, 1, 1, tzinfo=UTC)
_MIXED = [
    DATA_SOURCE_LIVE,
    DATA_SOURCE_SIMULATED,
    DATA_SOURCE_HELD,
    DATA_SOURCE_FAULT,
    DATA_SOURCE_LIVE,
]


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


def _seed(s: Session, qualities: list[str], tag: str = "TAG_0") -> None:
    for i, q in enumerate(qualities):
        s.add(
            TagLog(
                tag_name=tag,
                value=float(i),
                quality=q,
                timestamp=_BASE + _dt.timedelta(seconds=i),
            )
        )
    s.commit()


def _iso(offset_s: float) -> str:
    return (_BASE + _dt.timedelta(seconds=offset_s)).isoformat()


def test_trends_quality_aligned_with_values(session: Session) -> None:
    _seed(session, _MIXED)
    result = main.get_trends(
        tag_id="TAG_0", start_time=_iso(0), end_time=_iso(4), db=session
    )
    assert result["quality"] == _MIXED
    assert len(result["quality"]) == len(result["values"])
    assert len(result["quality"]) == len(result["timestamps"])


def test_downsampling_keeps_quality_aligned(session: Session) -> None:
    # Quality encodes the sample index parity so alignment is checkable.
    n = 100
    qualities = [DATA_SOURCE_FAULT if i % 2 else DATA_SOURCE_LIVE for i in range(n)]
    _seed(session, qualities)
    ts, values, quals, truncated = query_trend_samples(
        session,
        tag_name="TAG_0",
        start=_BASE,
        end=_BASE + _dt.timedelta(seconds=n - 1),
        max_points=10,
    )
    assert truncated is True
    assert len(ts) == len(values) == len(quals) < n
    for v, q in zip(values, quals, strict=True):
        assert q == (DATA_SOURCE_FAULT if int(v) % 2 else DATA_SOURCE_LIVE)
    assert values[-1] == float(n - 1)  # final sample still forced in


def test_query_trend_series_signature_unchanged(session: Session) -> None:
    _seed(session, _MIXED)
    out = query_trend_series(
        session, tag_name="TAG_0", start=_BASE, end=_BASE + _dt.timedelta(seconds=4)
    )
    assert len(out) == 3


def test_unrecognised_stored_quality_reported_unknown(session: Session) -> None:
    _seed(session, [DATA_SOURCE_LIVE, "bogus"])
    _, _, quals, _ = query_trend_samples(
        session, tag_name="TAG_0", start=_BASE, end=_BASE + _dt.timedelta(seconds=1)
    )
    assert quals == [DATA_SOURCE_LIVE, QUALITY_UNKNOWN]


def test_normalize_quality_contract() -> None:
    assert normalize_quality(None) == QUALITY_UNKNOWN
    assert normalize_quality("") == QUALITY_UNKNOWN
    assert normalize_quality(DATA_SOURCE_HELD) == DATA_SOURCE_HELD
    assert QUALITY_UNKNOWN not in {
        DATA_SOURCE_LIVE,
        DATA_SOURCE_SIMULATED,
        DATA_SOURCE_HELD,
        DATA_SOURCE_FAULT,
    }


def test_csv_header_pinned_with_quality_appended(session: Session) -> None:
    _seed(session, [DATA_SOURCE_HELD, "bogus"])
    rows = list(
        stream_tag_export_csv(
            session.get_bind(), select(TagLog).order_by(TagLog.timestamp)
        )
    )
    assert rows[0] == "Timestamp,Tag Name,Value,Quality\r\n"
    assert rows[1].rstrip().endswith(f",0.0,{DATA_SOURCE_HELD}")
    assert rows[2].rstrip().endswith(f",1.0,{QUALITY_UNKNOWN}")
    # Existing column order preserved: first three fields unchanged.
    assert rows[1].split(",")[1:3] == ["TAG_0", "0.0"]
