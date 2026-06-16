"""Historian data-capture service: status + retention control.

The polling loop in ``main.py`` already logs every tag value to the ``taglog``
table on every scan, so capture is continuous and automatic whenever the
backend is running. This module owns the *operator-facing* side of that data
set — reporting how much has been captured and clearing it down so the storage
device does not fill up during long test campaigns.

Design notes:
    - DbC: every public function validates its inputs (``TypeError`` for wrong
      types) and documents pre/postconditions.
    - LOD: this module talks only to the ORM models and the DB file path. It
      imports nothing from the FastAPI layer or the PLC clients, so it stays
      unit-testable against a plain in-memory SQLModel session.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from database import DB_FILE
from models import EventLog, TagLog
from pydantic import BaseModel, Field
from sqlalchemy import delete, func
from sqlmodel import Session, col, select


class CaptureStats(BaseModel):
    """Snapshot of the captured historian for the operator."""

    capturing: bool = Field(
        description="True whenever the backend is running (the poll loop logs "
        "every scan). The HMI shows this as the live REC indicator.",
    )
    total_rows: int = Field(ge=0, description="Rows in the tag historian.")
    distinct_tags: int = Field(ge=0, description="Distinct tags captured.")
    oldest_timestamp: str | None = Field(
        default=None, description="ISO time of the earliest sample, or null."
    )
    newest_timestamp: str | None = Field(
        default=None, description="ISO time of the latest sample, or null."
    )
    span_seconds: float = Field(
        ge=0.0, description="Seconds between oldest and newest sample."
    )
    db_bytes: int = Field(ge=0, description="On-disk size of the capture DB file.")
    event_rows: int = Field(ge=0, description="Rows in the event log.")


class ClearResult(BaseModel):
    """Outcome of a retention-clear operation."""

    tag_rows_deleted: int = Field(ge=0)
    event_rows_deleted: int = Field(ge=0)
    db_bytes_before: int = Field(ge=0)
    db_bytes_after: int = Field(ge=0)


def _db_size_bytes() -> int:
    """Best-effort on-disk size of the SQLite capture file (0 if absent)."""
    try:
        return os.path.getsize(DB_FILE)
    except OSError:
        return 0


def capture_stats(session: Session, *, capturing: bool = True) -> CaptureStats:
    """Return a snapshot of the captured historian.

    Args:
        session: An active SQLModel session.
        capturing: Whether the capture loop is currently running. Defaults True
            (the poll loop logs whenever the backend is up).

    Returns:
        CaptureStats with row counts, time span, and on-disk size.

    Raises:
        TypeError: If ``session`` is not a Session or ``capturing`` is not bool.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(capturing, bool):
        raise TypeError(f"capturing must be bool, got {type(capturing).__name__}")

    total_rows = session.exec(select(func.count()).select_from(TagLog)).one()
    event_rows = session.exec(select(func.count()).select_from(EventLog)).one()
    distinct_tags = session.exec(
        select(func.count(func.distinct(col(TagLog.tag_name))))
    ).one()
    oldest = session.exec(select(func.min(col(TagLog.timestamp)))).one()
    newest = session.exec(select(func.max(col(TagLog.timestamp)))).one()

    span = 0.0
    if oldest is not None and newest is not None:
        span = max(0.0, (newest - oldest).total_seconds())

    return CaptureStats(
        capturing=capturing,
        total_rows=int(total_rows or 0),
        distinct_tags=int(distinct_tags or 0),
        oldest_timestamp=oldest.isoformat() if oldest is not None else None,
        newest_timestamp=newest.isoformat() if newest is not None else None,
        span_seconds=span,
        db_bytes=_db_size_bytes(),
        event_rows=int(event_rows or 0),
    )


def clear_capture(
    session: Session,
    *,
    include_events: bool = False,
    before: datetime | None = None,
) -> ClearResult:
    """Delete captured rows to reclaim storage, then VACUUM to free disk.

    Args:
        session: An active SQLModel session.
        include_events: Also clear the event log when True.
        before: If given, only delete samples strictly older than this instant
            (a rolling-retention purge). When None, clears everything.

    Returns:
        ClearResult with rows deleted and before/after on-disk size.

    Raises:
        TypeError: If arguments are of the wrong type.

    Postcondition: a plain DELETE does not shrink a SQLite file, so this runs
    ``VACUUM`` afterward to actually return space to the storage device.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(include_events, bool):
        raise TypeError(
            f"include_events must be bool, got {type(include_events).__name__}"
        )
    if before is not None and not isinstance(before, datetime):
        raise TypeError(f"before must be a datetime or None, got {type(before)}")

    bytes_before = _db_size_bytes()

    tag_stmt = delete(TagLog)
    if before is not None:
        tag_stmt = tag_stmt.where(col(TagLog.timestamp) < before)
    tag_deleted = session.exec(tag_stmt).rowcount

    event_deleted = 0
    if include_events:
        event_stmt = delete(EventLog)
        if before is not None:
            event_stmt = event_stmt.where(col(EventLog.timestamp) < before)
        event_deleted = session.exec(event_stmt).rowcount

    session.commit()

    # Reclaim disk. VACUUM cannot run inside a transaction, so use a raw
    # connection outside the session's transaction scope.
    try:
        raw = session.connection().connection
        raw.execute("VACUUM")
    except Exception:  # pragma: no cover - VACUUM is best-effort
        pass

    return ClearResult(
        tag_rows_deleted=int(tag_deleted or 0),
        event_rows_deleted=int(event_deleted or 0),
        db_bytes_before=bytes_before,
        db_bytes_after=_db_size_bytes(),
    )


def utcnow() -> datetime:
    """Timezone-aware current UTC instant (wrapper for easy test patching)."""
    return datetime.now(UTC)
