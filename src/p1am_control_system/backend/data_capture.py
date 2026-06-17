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

import asyncio
import csv
import datetime as _dt
import io
import logging
import os
from collections.abc import Iterator
from typing import Any

from database import DB_FILE
from models import EventLog, TagLog
from pydantic import BaseModel, Field
from sqlalchemy import Engine, delete, func
from sqlmodel import Session, col, select

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017
TRENDS_MAX_POINTS = 50_000
EXPORT_CHUNK_ROWS = 5_000
HISTORIAN_RETENTION_INTERVAL_S = 300.0


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


class RetentionResult(BaseModel):
    """Outcome of a size-cap retention sweep."""

    over_cap: bool = Field(description="True if the DB exceeded the cap.")
    rows_deleted: int = Field(ge=0, description="Oldest tag rows purged.")
    db_bytes_before: int = Field(ge=0)
    db_bytes_after: int = Field(ge=0)


def _db_size_bytes() -> int:
    """Total on-disk footprint of the capture DB: main file + WAL/SHM sidecars.

    Under WAL, recent commits live in the ``-wal`` file until a checkpoint, so
    counting only the main file under-reports actual usage — which would make
    both the status display and the size-cap check inaccurate.
    """
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += os.path.getsize(DB_FILE + suffix)
        except OSError:
            pass
    return total


def parse_query_bound(value: str) -> _dt.datetime:
    """Parse an ISO time-range bound into an aware UTC datetime.

    Accepts a trailing ``Z`` or an explicit offset; a tz-naive string is assumed
    to be UTC. Normalizing to aware-UTC is required because the historian stores
    aware-UTC timestamps — comparing a naive bound against them silently
    mis-filters range queries.

    Raises:
        TypeError: If ``value`` is not a str.
        ValueError: If ``value`` is not a valid ISO datetime.
    """
    if not isinstance(value, str):
        raise TypeError(f"value must be a str, got {type(value).__name__}")
    parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_tag_names(tag_ids: str) -> list[str]:
    """Normalize comma-separated tag ids/names for historian queries."""
    if not isinstance(tag_ids, str):
        raise TypeError(f"tag_ids must be a str, got {type(tag_ids).__name__}")
    return [
        f"TAG_{tag_id}" if tag_id.isdigit() else tag_id
        for raw in tag_ids.split(",")
        if (tag_id := raw.strip())
    ]


def stream_tag_export_csv(
    bind: Any,
    statement: Any,
    *,
    chunk_rows: int = EXPORT_CHUNK_ROWS,
) -> Iterator[str]:
    """Yield export CSV rows without materializing the whole result set."""
    header = io.StringIO()
    csv.writer(header).writerow(["Timestamp", "Tag Name", "Value"])
    yield header.getvalue()
    with Session(bind) as stream_db:
        result = stream_db.exec(statement).yield_per(chunk_rows)
        for row in result:
            line = io.StringIO()
            csv.writer(line).writerow(
                [row.timestamp.isoformat(), row.tag_name, row.value]
            )
            yield line.getvalue()


def historian_max_bytes() -> int:
    """Size cap for the historian (default 2 GiB). 0 disables auto-purge."""
    try:
        return int(os.environ.get("P1AM_HISTORIAN_MAX_BYTES", str(2 * 1024**3)))
    except ValueError:
        return 2 * 1024**3


async def historian_retention_loop(
    *,
    shutdown_event: asyncio.Event,
    engine: Any,
    logger: logging.Logger,
    interval_s: float = HISTORIAN_RETENTION_INTERVAL_S,
) -> None:
    """Periodically enforce the historian size cap off the request hot path."""
    max_bytes = historian_max_bytes()
    if max_bytes <= 0:
        logger.info("Historian size-cap auto-purge disabled (max_bytes<=0).")
        return
    logger.info("Historian retention active: cap %d bytes.", max_bytes)
    while not shutdown_event.is_set():
        await asyncio.sleep(interval_s)
        if shutdown_event.is_set():
            break
        try:
            with Session(engine) as session:
                enforce_size_cap(session, max_bytes)
        except Exception as ret_err:
            logger.error("Historian retention sweep failed: %s", ret_err)


def _vacuum(session: Session) -> None:
    """VACUUM on a dedicated AUTOCOMMIT connection (never the caller's session).

    VACUUM cannot run inside a transaction and rewrites the whole file, so it
    must not borrow the poll loop's connection. Best-effort: logged, not raised.
    """
    try:
        bind = session.get_bind()
        engine = bind if isinstance(bind, Engine) else bind.engine
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").exec_driver_sql(
                "VACUUM"
            )
    except Exception as exc:  # pragma: no cover - best-effort maintenance
        logging.getLogger("dcs_backend.data_capture").warning("VACUUM failed: %s", exc)


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
    before: _dt.datetime | None = None,
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
    if before is not None and not isinstance(before, _dt.datetime):
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
    _vacuum(session)

    return ClearResult(
        tag_rows_deleted=int(tag_deleted or 0),
        event_rows_deleted=int(event_deleted or 0),
        db_bytes_before=bytes_before,
        db_bytes_after=_db_size_bytes(),
    )


def enforce_size_cap(
    session: Session,
    max_bytes: int,
    *,
    headroom: float = 0.9,
) -> RetentionResult:
    """Keep the historian under ``max_bytes`` by purging the oldest samples.

    When the on-disk file exceeds the cap, this estimates how many of the oldest
    rows (by insert order / id) to drop to land at ``headroom`` × cap, deletes
    them in one ranged statement, then VACUUMs to actually return the space. A
    no-op (and cheap) while under the cap, so it is safe to call periodically.

    Args:
        session: An active SQLModel session.
        max_bytes: Target maximum on-disk size of the capture DB.
        headroom: Fraction of the cap to target after a purge (0 < headroom < 1),
            leaving room before the next sweep.

    Returns:
        RetentionResult describing whether it acted and how much it freed.

    Raises:
        TypeError: If ``session``/``max_bytes`` are the wrong type.
        ValueError: If ``max_bytes`` <= 0 or ``headroom`` not in (0, 1).
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        raise TypeError(f"max_bytes must be an int, got {type(max_bytes).__name__}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if not 0.0 < headroom < 1.0:
        raise ValueError(f"headroom must be in (0, 1), got {headroom}")

    bytes_before = _db_size_bytes()
    total_rows = int(session.exec(select(func.count()).select_from(TagLog)).one() or 0)
    if bytes_before <= max_bytes or total_rows == 0:
        return RetentionResult(
            over_cap=False,
            rows_deleted=0,
            db_bytes_before=bytes_before,
            db_bytes_after=bytes_before,
        )

    bytes_per_row = bytes_before / total_rows
    target_rows = int((max_bytes * headroom) / bytes_per_row)
    to_delete = max(0, total_rows - target_rows)
    if to_delete <= 0:
        return RetentionResult(
            over_cap=True,
            rows_deleted=0,
            db_bytes_before=bytes_before,
            db_bytes_after=bytes_before,
        )

    # Find the id cutoff for the oldest `to_delete` rows, then delete by range —
    # avoids materializing a huge id list.
    cutoff = session.exec(
        select(col(TagLog.id))
        .order_by(col(TagLog.id).asc())
        .offset(to_delete - 1)
        .limit(1)
    ).first()
    deleted = 0
    if cutoff is not None:
        deleted = session.exec(delete(TagLog).where(col(TagLog.id) <= cutoff)).rowcount
        session.commit()
        _vacuum(session)

    logging.getLogger("dcs_backend.data_capture").warning(
        "Retention: DB over cap (%d > %d bytes); purged %d oldest rows.",
        bytes_before,
        max_bytes,
        int(deleted or 0),
    )
    return RetentionResult(
        over_cap=True,
        rows_deleted=int(deleted or 0),
        db_bytes_before=bytes_before,
        db_bytes_after=_db_size_bytes(),
    )


def utcnow() -> _dt.datetime:
    """Timezone-aware current UTC instant (wrapper for easy test patching)."""
    return _dt.datetime.now(UTC)
