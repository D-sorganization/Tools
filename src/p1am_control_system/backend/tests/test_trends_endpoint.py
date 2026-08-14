"""Endpoint-level tests for ``GET /api/trends`` (``main.get_trends``).

The handler is a plain ``def`` that maps the ``query_trend_series`` contract onto
the HTTP response schema ``{timestamps, values, truncated}``. These tests call it
directly with an in-memory SQLModel session (no app lifespan / PLC boot / auth),
covering the whole-window decimation fix, numeric tag resolution, the response
schema the frontend depends on, and the ``ValueError``/``TypeError`` -> HTTP 400
mapping.
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

# ``main`` builds a process-wide PLC client at import time; mirror the driver
# selection used by the functional suite so importing it here cannot leave the
# shared client in a state that breaks sibling suites (collection order varies).
os.environ.setdefault("PLC_DRIVER", "modbus")

sys.path.insert(0, str(Path(__file__).parent.parent))

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017

import main  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from models import TagLog  # noqa: E402
from sqlalchemy import StaticPool  # noqa: E402
from sqlmodel import Session, SQLModel, create_engine  # noqa: E402

_BASE = _dt.datetime(2026, 1, 1, tzinfo=UTC)


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


def _seed(s: Session, *, tag_name: str, n: int, step_s: float = 1.0) -> None:
    for i in range(n):
        s.add(
            TagLog(
                tag_name=tag_name,
                value=float(i),
                timestamp=_BASE + _dt.timedelta(seconds=i * step_s),
            )
        )
    s.commit()


def _iso(offset_s: float) -> str:
    return (_BASE + _dt.timedelta(seconds=offset_s)).isoformat()


def test_response_schema_and_small_range(session: Session) -> None:
    _seed(session, tag_name="TAG_0", n=5)
    result = main.get_trends(
        tag_id="TAG_0",
        start_time=_iso(0),
        end_time=_iso(4),
        db=session,
    )
    assert set(result) == {
        "timestamps",
        "values",
        "qualities",
        "diagnostic_reasons",
        "source_timestamps",
        "sequences",
        "sources",
        "truncated",
    }
    assert result["truncated"] is False
    assert result["values"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert len(result["timestamps"]) == 5
    assert all(isinstance(t, str) for t in result["timestamps"])  # ISO strings
    assert result["qualities"] == ["uncertain"] * 5
    assert result["diagnostic_reasons"] == ["legacy_unqualified"] * 5
    assert len(result["source_timestamps"]) == 5
    assert result["sequences"] == [0] * 5
    assert result["sources"] == ["legacy.adapter"] * 5


def test_numeric_tag_id_resolves_to_tag_name(session: Session) -> None:
    _seed(session, tag_name="TAG_7", n=3)
    result = main.get_trends(
        tag_id="7",  # numeric id -> TAG_7
        start_time=_iso(0),
        end_time=_iso(2),
        db=session,
    )
    assert result["values"] == [0.0, 1.0, 2.0]


def test_large_range_spans_whole_window_via_endpoint(session: Session) -> None:
    # The actual bug: a long window must not be clipped to only its newest slice.
    _seed(session, tag_name="TAG_0", n=1000)
    result = main.get_trends(
        tag_id="TAG_0",
        start_time=_iso(0),
        end_time=_iso(999),
        max_points=100,
        db=session,
    )
    assert result["truncated"] is True
    assert len(result["timestamps"]) <= 101
    # First returned value is the range-start sample (index 0); a newest-slice
    # bug would start near index ~900 instead.
    assert result["values"][0] == 0.0
    assert result["values"][-1] == 999.0


def test_invalid_date_returns_400(session: Session) -> None:
    with pytest.raises(HTTPException) as exc:
        main.get_trends(
            tag_id="TAG_0",
            start_time="not-a-date",
            end_time=_iso(4),
            db=session,
        )
    assert exc.value.status_code == 400


def test_out_of_range_max_points_returns_400(session: Session) -> None:
    _seed(session, tag_name="TAG_0", n=5)
    with pytest.raises(HTTPException) as exc:
        main.get_trends(
            tag_id="TAG_0",
            start_time=_iso(0),
            end_time=_iso(4),
            max_points=9,  # below the sane floor
            db=session,
        )
    assert exc.value.status_code == 400


def test_start_after_end_returns_400(session: Session) -> None:
    _seed(session, tag_name="TAG_0", n=5)
    with pytest.raises(HTTPException) as exc:
        main.get_trends(
            tag_id="TAG_0",
            start_time=_iso(10),
            end_time=_iso(0),
            db=session,
        )
    assert exc.value.status_code == 400
