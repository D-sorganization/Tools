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
import time
from collections.abc import Callable, Iterator
from typing import Any

from database import DB_FILE
from models import EventLog, TagLog, ensure_utc
from pydantic import BaseModel, Field
from settings import P1AMSettings, get_settings
from sqlalchemy import Engine, delete, func
from sqlmodel import Session, col, select

UTC = getattr(_dt, "UTC", _dt.timezone.utc)  # noqa: UP017
TRENDS_MAX_POINTS = 50_000
# Inclusive bounds for a caller-supplied ``max_points`` override on the trends
# read. The floor keeps a series legible; the ceiling caps the response so a
# pathological override cannot materialize an unbounded payload.
TRENDS_MIN_MAX_POINTS = 10
TRENDS_MAX_MAX_POINTS = 200_000
# yield_per batch size while streaming the decimation cursor (memory hint only).
TRENDS_STREAM_CHUNK_ROWS = 10_000
EXPORT_CHUNK_ROWS = 5_000
HISTORIAN_RETENTION_INTERVAL_S = 300.0

# --- Retention budgets (issues #4006 / #4027) ------------------------------- #
# The size cap is split into two *independently enforced* budgets. Before this
# split the sweep charged the whole file (main + WAL + SHM) against the TagLog
# row count, so a fat event log inflated bytes-per-row several-fold and the
# sweep kept over-deleting trend history until the tag historian was empty.
EVENTLOG_BUDGET_FRACTION = 0.10
# Age-based EventLog retention. The event log previously had no automatic
# retention of any kind, so it grew without bound for the life of the unit.
HISTORIAN_EVENT_MAX_AGE_S = 30.0 * 86_400.0
# Free pages returned to the filesystem per sweep. At SQLite's 4 KiB default
# page size, 2000 pages is ~8 MiB — a bounded, sub-second lock on SD storage,
# unlike a full VACUUM which rewrites the entire file under an exclusive lock.
HISTORIAN_VACUUM_PAGES = 2_000
# Fallback per-row on-disk cost (row + its index entries) used only when the
# ``dbstat`` virtual table is not compiled into the host's SQLite.
TAGLOG_EST_BYTES_PER_ROW = 96.0
EVENTLOG_EST_BYTES_PER_ROW = 192.0


class CaptureStats(BaseModel):
    """Snapshot of the captured historian for the operator."""

    capturing: bool = Field(
        description="True whenever the backend is running (the poll loop logs "
        "every scan). The HMI shows this as the live REC indicator.",
    )
    total_rows: int = Field(ge=0, description="Rows in the tag historian.")
    distinct_tags: int = Field(ge=0, description="Distinct tags captured.")
    oldest_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 UTC time (explicit offset) of the earliest sample.",
    )
    newest_timestamp: str | None = Field(
        default=None,
        description="ISO-8601 UTC time (explicit offset) of the latest sample.",
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


class TableFootprint(BaseModel):
    """How the historian's on-disk bytes split between its two log tables.

    ``taglog_bytes``/``eventlog_bytes`` include each table's own index pages, so
    they can be charged directly against that table's retention budget.
    """

    total_bytes: int = Field(ge=0, description="Whole DB footprint incl. WAL/SHM.")
    taglog_bytes: int = Field(ge=0, description="TagLog table + its indexes.")
    eventlog_bytes: int = Field(ge=0, description="EventLog table + its indexes.")
    measured: bool = Field(
        description="True when read from dbstat; False when estimated per-row."
    )


class RetentionResult(BaseModel):
    """Outcome of a size-cap retention sweep."""

    over_cap: bool = Field(description="True if the DB exceeded the cap.")
    rows_deleted: int = Field(ge=0, description="Oldest tag rows purged.")
    event_rows_deleted: int = Field(
        default=0, ge=0, description="Event rows purged (age + size budget)."
    )
    db_bytes_before: int = Field(ge=0)
    db_bytes_after: int = Field(ge=0)
    taglog_bytes_before: int = Field(default=0, ge=0)
    eventlog_bytes_before: int = Field(default=0, ge=0)


class CaptureConfig(BaseModel):
    """Operator-configurable historian sampling parameters."""

    interval_s: float = Field(
        ge=0.0,
        le=3600.0,
        description=(
            "Minimum seconds between historian writes. 0 logs every scan; "
            "larger values shrink the data files. The live stream is unaffected."
        ),
    )


class CaptureThrottle:
    """Rate-limits historian writes to at most one per ``interval_s``.

    The scan loop calls :meth:`due` once per scan; it returns True only when at
    least ``interval_s`` of wall-clock has elapsed since the last write it
    approved (the first call always approves). This decouples how often data is
    *persisted* from how often the PLC is *polled*, so the DB grows at a bounded,
    operator-chosen rate without slowing the control/stream loop.

    A clock is injected so the timing is deterministic in tests (DbC: the
    interval setter validates type and range).
    """

    def __init__(
        self,
        interval_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._last: float | None = None
        self._interval_s = 0.0
        self.set_interval_s(interval_s)

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def set_interval_s(self, value: float) -> None:
        """Update the minimum write period.

        Raises:
            TypeError: if value is not numeric.
            ValueError: if value is negative or non-finite.
        """
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"interval_s must be numeric, got {type(value).__name__}")
        v = float(value)
        if not (v >= 0.0) or v != v or v == float("inf"):
            raise ValueError(f"interval_s must be a finite value >= 0, got {value}")
        self._interval_s = v

    def due(self) -> bool:
        """Return True (and arm the next window) when a write is allowed now."""
        now = self._clock()
        if self._last is None or (now - self._last) >= self._interval_s:
            self._last = now
            return True
        return False


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


def _iso_utc(value: _dt.datetime) -> str:
    """Render a historian timestamp as ISO-8601 carrying an explicit offset.

    An offset-less string is re-parsed by the browser (per ECMAScript) as
    *local* time, which is how an "export everything" window silently started
    hours late on a non-UTC host (issue #4025). Every timestamp this module
    puts on the API boundary goes through here.
    """
    # Annotated locally: the flat sibling import makes ``ensure_utc`` opaque to
    # mypy in some invocations, and an unannotated return would leak Any.
    aware: _dt.datetime = ensure_utc(value)
    return aware.isoformat()


def parse_query_bound(value: str) -> _dt.datetime:
    """Parse an ISO time-range bound into an aware UTC datetime.

    Accepts a trailing ``Z`` or an explicit offset; a tz-naive string is assumed
    to be UTC. Normalizing to aware-UTC keeps the bound on the same clock as the
    stored rows: ``TagLog.timestamp``/``EventLog.timestamp`` use the
    :class:`models.UtcDateTime` column type, which converts to UTC on write and
    re-attaches UTC on read. (Historically the columns were plain ``DATETIME``
    and SQLite dropped the offset on both ends, so this normalization was the
    only thing standing between a naive bound and a silently mis-filtered
    range query.)

    Raises:
        TypeError: If ``value`` is not a str.
        ValueError: If ``value`` is not a valid ISO datetime.
    """
    if not isinstance(value, str):
        raise TypeError(f"value must be a str, got {type(value).__name__}")
    parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    normalized: _dt.datetime = ensure_utc(parsed)
    return normalized


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
                [_iso_utc(row.timestamp), row.tag_name, row.value]
            )
            yield line.getvalue()


def query_trend_series(
    session: Session,
    *,
    tag_name: str,
    start: _dt.datetime,
    end: _dt.datetime,
    max_points: int = TRENDS_MAX_POINTS,
) -> tuple[list[_dt.datetime], list[float], bool]:
    """Fetch a tag's historian samples over ``[start, end]``, decimated to fit.

    A naive ``DESC + LIMIT`` clips a long window to only its newest
    ``max_points`` rows, so a multi-hour or multi-day request would silently show
    just the most-recent slice. This instead strides evenly across the *entire*
    ascending range, so the returned series always spans the whole window.

    Contract:
        - When the range holds at most ``max_points`` samples: returns them all,
          ascending by timestamp, with ``truncated=False`` (unchanged behavior).
        - When it holds more: returns approximately ``max_points`` samples evenly
          spanning the whole ``[start, end]`` window (the first and last in-range
          samples are always included), ascending, with ``truncated=True``.

    The decimation is a ``COUNT``-then-stride over a streamed ascending cursor
    (``yield_per``), so peak memory stays ``O(max_points)`` rather than
    ``O(rows-in-range)`` — a 24 h window at 10 Hz is never materialized at once.

    Args:
        session: An active SQLModel session.
        tag_name: The resolved historian tag name (e.g. ``"TAG_0"``).
        start: Inclusive lower time bound (aware datetime).
        end: Inclusive upper time bound (aware datetime).
        max_points: Maximum samples to return. Defaults to ``TRENDS_MAX_POINTS``.

    Returns:
        ``(timestamps, values, truncated)`` — two equal-length, ascending lists
        (datetimes and floats) plus the downsample flag.

    Raises:
        TypeError: If any argument is of the wrong type.
        ValueError: If ``max_points`` is outside ``[TRENDS_MIN_MAX_POINTS,
            TRENDS_MAX_MAX_POINTS]`` or ``start`` is after ``end``.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(tag_name, str):
        raise TypeError(f"tag_name must be a str, got {type(tag_name).__name__}")
    if not isinstance(start, _dt.datetime):
        raise TypeError(f"start must be a datetime, got {type(start).__name__}")
    if not isinstance(end, _dt.datetime):
        raise TypeError(f"end must be a datetime, got {type(end).__name__}")
    # bool is an int subclass; reject it so ``max_points=True`` cannot pose as 1.
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError(f"max_points must be an int, got {type(max_points).__name__}")
    if not TRENDS_MIN_MAX_POINTS <= max_points <= TRENDS_MAX_MAX_POINTS:
        raise ValueError(
            f"max_points must be within [{TRENDS_MIN_MAX_POINTS}, "
            f"{TRENDS_MAX_MAX_POINTS}], got {max_points}"
        )
    if start > end:
        raise ValueError(
            f"start ({start.isoformat()}) must not be after end ({end.isoformat()})"
        )

    # One shared filter tuple keeps the COUNT and the row cursor in lock-step.
    where_clauses = (
        col(TagLog.tag_name) == tag_name,
        col(TagLog.timestamp) >= start,
        col(TagLog.timestamp) <= end,
    )
    total = int(
        session.exec(
            select(func.count()).select_from(TagLog).where(*where_clauses)
        ).one()
        or 0
    )
    ordered = (
        select(TagLog.timestamp, TagLog.value)
        .where(*where_clauses)
        .order_by(col(TagLog.timestamp).asc())
    )

    timestamps: list[_dt.datetime] = []
    values: list[float] = []

    if total <= max_points:
        # Fits the budget: return every in-range sample verbatim, ascending.
        for ts, value in session.exec(ordered):
            timestamps.append(ts)
            values.append(float(value))
        return timestamps, values, False

    # Over budget: stride across the *whole* ascending cursor. Ceil division
    # keeps the emitted count <= max_points (+1 for the forced final sample).
    stride = (total + max_points - 1) // max_points
    last_index = total - 1
    last_emitted = -1
    last_ts: _dt.datetime | None = None
    last_value = 0.0
    cursor = session.exec(ordered).yield_per(TRENDS_STREAM_CHUNK_ROWS)
    for i, (ts, value) in enumerate(cursor):
        last_ts, last_value = ts, value
        if i % stride == 0:
            timestamps.append(ts)
            values.append(float(value))
            last_emitted = i
    # Always include the final in-range sample so the series reaches ``end``
    # rather than stopping up to one stride short of it.
    if last_emitted != last_index and last_ts is not None:
        timestamps.append(last_ts)
        values.append(float(last_value))
    return timestamps, values, True


def historian_max_bytes(settings: P1AMSettings | None = None) -> int:
    """Size cap for the historian (default 2 GiB). 0 disables auto-purge."""
    return int((settings or get_settings()).historian_max_bytes)


def run_retention_sweep(
    engine: Any,
    *,
    max_bytes: int,
    logger: logging.Logger,
    event_max_age_s: float | None = HISTORIAN_EVENT_MAX_AGE_S,
) -> RetentionResult:
    """One complete historian maintenance pass. **Blocking — never `await` it.**

    Enforces the two retention budgets, returns a bounded chunk of free pages to
    the filesystem, then truncates the WAL. Every step drives synchronous SQLite
    calls, so this must be executed on a worker thread (see
    :func:`historian_retention_loop`, which dispatches it via
    ``asyncio.to_thread``). Called directly on the event loop it would freeze
    the control loop, the websocket broadcast and every HTTP endpoint —
    including E-stop — for the duration (issue #4006).

    Args:
        engine: The SQLAlchemy engine bound to the historian DB.
        max_bytes: Total on-disk size cap for the capture DB.
        logger: Where the sweep reports what it deleted and why.
        event_max_age_s: Age-based EventLog retention, or None to disable it.

    Returns:
        The :class:`RetentionResult` for this sweep.
    """
    with Session(engine) as session:
        result = enforce_size_cap(
            session, max_bytes, event_max_age_s=event_max_age_s, logger=logger
        )
    # Return free pages in a bounded chunk rather than rewriting the whole file.
    _reclaim_free_pages(engine, max_pages=HISTORIAN_VACUUM_PAGES, logger=logger)
    # Reclaim WAL disk: a long-held reader can bloat the WAL between sweeps;
    # TRUNCATE checkpoints it back to journal_size_limit.
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    return result


async def historian_retention_loop(
    *,
    shutdown_event: asyncio.Event,
    engine: Any,
    logger: logging.Logger,
    interval_s: float | None = None,
    settings: P1AMSettings | None = None,
) -> None:
    """Periodically enforce the historian retention budgets, off the event loop.

    This coroutine is scheduled with ``asyncio.create_task``, which means its
    body runs *on the event loop thread* — so the sweep itself is dispatched to
    a worker thread with ``asyncio.to_thread``. Only the sleeping and the
    dispatch happen on the loop; the SQLite work never blocks E-stop.

    A failed sweep is logged and the loop continues: retention is maintenance,
    and must never take the controller down.
    """
    settings = settings or get_settings()
    max_bytes = historian_max_bytes(settings)
    if max_bytes <= 0:
        logger.info("Historian size-cap auto-purge disabled (max_bytes<=0).")
        return
    logger.info("Historian retention active: cap %d bytes.", max_bytes)
    interval = (
        interval_s
        if interval_s is not None
        else settings.historian_retention_interval_s
    )
    while not shutdown_event.is_set():
        await asyncio.sleep(interval)
        if shutdown_event.is_set():
            break
        try:
            await asyncio.to_thread(
                run_retention_sweep,
                engine,
                max_bytes=max_bytes,
                logger=logger,
            )
        except Exception as ret_err:
            logger.error("Historian retention sweep failed: %s", ret_err)


def _engine_of(bind: Any) -> Engine:
    """Resolve an Engine from an Engine, Connection or Session bind."""
    if isinstance(bind, Engine):
        return bind
    engine = getattr(bind, "engine", None)
    if isinstance(engine, Engine):
        return engine
    raise TypeError(f"cannot resolve an Engine from {type(bind).__name__}")


def _reclaim_free_pages(
    bind: Any,
    *,
    max_pages: int = HISTORIAN_VACUUM_PAGES,
    logger: logging.Logger | None = None,
) -> int:
    """Return at most ``max_pages`` free pages to the filesystem.

    Incremental vacuuming moves a *bounded* number of pages, so the longest lock
    this maintenance step can take is proportional to ``max_pages`` — unlike
    ``VACUUM``, which rewrites the whole file and on a 1 GiB DB on SD storage
    holds an exclusive lock for tens of seconds (issue #4006). Measured cost:
    ~60 ms per 1000 pages (~4 MiB) in WAL + ``synchronous=NORMAL``.

    Requires the file to be in ``auto_vacuum=INCREMENTAL`` mode
    (:func:`database._enable_incremental_autovacuum` converts legacy files at
    startup); on a NONE-mode file the pragma is a harmless no-op returning 0.

    Implementation note: ``PRAGMA incremental_vacuum(N)`` yields one result row
    per page freed, and pysqlite advances a no-column statement exactly one step
    per ``execute()``. So passing a large ``N`` frees a single page — the pragma
    has to be issued once per page. The loop below is still bounded by
    ``min(freelist, max_pages)``, which is the property that matters.

    Args:
        bind: An Engine, Connection or Session bind for the historian DB.
        max_pages: Upper bound on pages reclaimed by this call. Must be > 0.
        logger: Optional logger for the best-effort failure path.

    Returns:
        The number of free pages actually reclaimed (0 when nothing to do).

    Raises:
        TypeError: If ``max_pages`` is not an int.
        ValueError: If ``max_pages`` is not positive.
    """
    if isinstance(max_pages, bool) or not isinstance(max_pages, int):
        raise TypeError(f"max_pages must be an int, got {type(max_pages).__name__}")
    if max_pages <= 0:
        raise ValueError(f"max_pages must be positive, got {max_pages}")
    log = logger or logging.getLogger("dcs_backend.data_capture")
    try:
        engine = _engine_of(bind)
        # AUTOCOMMIT from the first statement: incremental_vacuum must not run
        # inside the caller's transaction.
        connection = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        with connection as conn:
            before = int(conn.exec_driver_sql("PRAGMA freelist_count").scalar() or 0)
            for _ in range(min(before, max_pages)):
                conn.exec_driver_sql("PRAGMA incremental_vacuum(1)")
            after = int(conn.exec_driver_sql("PRAGMA freelist_count").scalar() or 0)
        return max(0, before - after)
    except Exception as exc:  # pragma: no cover - best-effort maintenance
        log.warning("Incremental vacuum skipped: %s", exc)
        return 0


def _vacuum(session: Session) -> None:
    """Full VACUUM on a dedicated AUTOCOMMIT connection (never the caller's).

    VACUUM cannot run inside a transaction and rewrites the whole file, so it
    must not borrow the poll loop's connection. Reserved for the *explicit*
    operator "clear capture" action, where a one-off whole-file rewrite is the
    point; the periodic sweep uses :func:`_reclaim_free_pages` instead so no
    unattended maintenance step takes an unbounded lock. Best-effort: logged,
    not raised.
    """
    try:
        engine = _engine_of(session.get_bind())
        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").exec_driver_sql(
                "VACUUM"
            )
    except Exception as exc:  # pragma: no cover - best-effort maintenance
        logging.getLogger("dcs_backend.data_capture").warning("VACUUM failed: %s", exc)


def _table_index_names(table: str) -> tuple[str, ...]:
    """Return the dbstat ``name`` values that bill to ``table`` (data + indexes)."""
    return (table, f"ix_{table}_", f"sqlite_autoindex_{table}_")


def _dbstat_bytes(session: Session) -> dict[str, int] | None:
    """Per-object page bytes from the ``dbstat`` virtual table, or None.

    ``dbstat`` requires ``SQLITE_ENABLE_DBSTAT_VTAB`` in the host's SQLite. When
    it is missing the caller falls back to a per-row estimate.
    """
    try:
        rows = (
            session.connection()
            .exec_driver_sql("SELECT name, SUM(pgsize) FROM dbstat GROUP BY name")
            .fetchall()
        )
    except Exception:  # pragma: no cover - depends on the host SQLite build
        return None
    return {str(name): int(size or 0) for name, size in rows}


def _sum_for_table(stats: dict[str, int], table: str) -> int:
    """Sum the data + index page bytes attributable to one table."""
    exact, *prefixes = _table_index_names(table)
    return sum(
        size
        for name, size in stats.items()
        if name == exact or any(name.startswith(p) for p in prefixes)
    )


def historian_footprint(
    session: Session, *, total_bytes: int | None = None
) -> TableFootprint:
    """Split the historian's on-disk footprint between TagLog and EventLog.

    The size cap must be charged per table: sizing a purge from the *whole* file
    while counting only TagLog rows inflates bytes-per-row by however much the
    event log occupies, which makes the sweep over-delete trend history and
    still never get under the cap (issue #4027).

    Uses ``dbstat`` for real page accounting when the host SQLite provides it,
    and falls back to row-count x nominal row width otherwise. Because
    ``dbstat`` sees only the main file, any excess in ``total_bytes`` (the WAL
    and SHM sidecars) is apportioned across the tables in the same ratio.

    Args:
        session: An active SQLModel session.
        total_bytes: Pre-measured whole-DB footprint; measured if omitted.

    Returns:
        A :class:`TableFootprint`. ``taglog_bytes + eventlog_bytes`` is always
        <= ``total_bytes`` (the remainder is the other tables' pages).

    Raises:
        TypeError: If ``session`` is not a Session or ``total_bytes`` not an int.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if total_bytes is not None and (
        isinstance(total_bytes, bool) or not isinstance(total_bytes, int)
    ):
        raise TypeError(
            f"total_bytes must be an int or None, got {type(total_bytes).__name__}"
        )

    total = _db_size_bytes() if total_bytes is None else int(total_bytes)
    stats = _dbstat_bytes(session)
    if stats:
        measured_total = sum(stats.values())
        # The file can never be smaller than the pages it holds; an in-memory or
        # not-yet-flushed DB reports 0 on disk.
        total = max(total, measured_total)
        scale = (total / measured_total) if measured_total > 0 else 1.0
        return TableFootprint(
            total_bytes=total,
            taglog_bytes=int(_sum_for_table(stats, "taglog") * scale),
            eventlog_bytes=int(_sum_for_table(stats, "eventlog") * scale),
            measured=True,
        )

    # Fallback: apportion the measured file size by estimated row widths so the
    # two budgets still move relative to each other as the tables grow.
    tag_rows = int(session.exec(select(func.count()).select_from(TagLog)).one() or 0)
    event_rows = int(
        session.exec(select(func.count()).select_from(EventLog)).one() or 0
    )
    tag_est = tag_rows * TAGLOG_EST_BYTES_PER_ROW
    event_est = event_rows * EVENTLOG_EST_BYTES_PER_ROW
    est_total = tag_est + event_est
    if est_total <= 0:
        return TableFootprint(
            total_bytes=total, taglog_bytes=0, eventlog_bytes=0, measured=False
        )
    scale = min(1.0, total / est_total) if total > 0 else 1.0
    return TableFootprint(
        total_bytes=total,
        taglog_bytes=int(tag_est * scale),
        eventlog_bytes=int(event_est * scale),
        measured=False,
    )


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
        oldest_timestamp=_iso_utc(oldest) if oldest is not None else None,
        newest_timestamp=_iso_utc(newest) if newest is not None else None,
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


def _purge_oldest_by_id(
    session: Session,
    # Deliberately ``Any``: this is a table *class* dispatched dynamically, and
    # its ``id`` attribute resolves through SQLModel's descriptor machinery
    # differently depending on whether the type checker can see the ORM stubs.
    model: Any,
    *,
    rows: int,
    current_bytes: int,
    budget_bytes: int,
    headroom: float,
) -> int:
    """Drop the oldest rows of one table until it fits ``budget_bytes``.

    ``bytes_per_row`` is derived from **that table's own** footprint, so a
    neighbouring table's growth can never inflate the estimate and over-delete
    here (issue #4027). At least one row is always retained, so a size purge can
    never empty a log outright.

    Args:
        session: An active SQLModel session.
        model: The table to purge; must expose a monotonically-increasing ``id``.
        rows: Current row count of that table.
        current_bytes: That table's measured/estimated on-disk footprint.
        budget_bytes: The table's share of the size cap.
        headroom: Fraction of the budget to target after the purge.

    Returns:
        The number of rows deleted (0 when already inside the budget).
    """
    if rows <= 0 or current_bytes <= budget_bytes:
        return 0
    bytes_per_row = current_bytes / rows
    if bytes_per_row <= 0.0:
        return 0
    target_rows = max(1, int((budget_bytes * headroom) / bytes_per_row))
    to_delete = max(0, rows - target_rows)
    if to_delete <= 0:
        return 0
    # Find the id cutoff for the oldest `to_delete` rows, then delete by range —
    # avoids materializing a huge id list.
    id_col = col(model.id)
    cutoff = session.exec(
        select(id_col).order_by(id_col.asc()).offset(to_delete - 1).limit(1)
    ).first()
    if cutoff is None:
        return 0
    deleted = session.exec(delete(model).where(id_col <= cutoff)).rowcount
    session.commit()
    return int(deleted or 0)


def _purge_events_older_than(session: Session, cutoff: _dt.datetime) -> int:
    """Delete event rows strictly older than ``cutoff``; returns the row count."""
    deleted = session.exec(
        delete(EventLog).where(col(EventLog.timestamp) < cutoff)
    ).rowcount
    session.commit()
    return int(deleted or 0)


def enforce_size_cap(
    session: Session,
    max_bytes: int,
    *,
    headroom: float = 0.9,
    event_max_age_s: float | None = HISTORIAN_EVENT_MAX_AGE_S,
    now: _dt.datetime | None = None,
    logger: logging.Logger | None = None,
) -> RetentionResult:
    """Keep the historian inside its retention budgets by purging oldest rows.

    The cap is split in two and each half is enforced against its **own** table:

    * ``EventLog`` gets ``EVENTLOG_BUDGET_FRACTION`` of ``max_bytes`` plus an
      age-based pass (``event_max_age_s``). The age pass runs unconditionally —
      the event log previously had no automatic retention at all and grew for
      the life of the unit.
    * ``TagLog`` gets the remainder. Its bytes-per-row now comes from the tag
      historian's own footprint, not the whole file, so a fat event log can no
      longer inflate the estimate and delete the entire trend history sweep
      after sweep (issue #4027).

    Reclaiming the freed pages is deliberately *not* done here: the periodic
    sweep does it in bounded chunks via :func:`_reclaim_free_pages`.

    Args:
        session: An active SQLModel session.
        max_bytes: Target maximum on-disk size of the capture DB.
        headroom: Fraction of each budget to target after a purge
            (0 < headroom < 1), leaving room before the next sweep.
        event_max_age_s: Delete events older than this many seconds, or None to
            disable the age pass. Must be positive when given.
        now: Reference instant for the age pass (defaults to the current UTC
            time); injected for deterministic tests.
        logger: Where to report what was deleted and why.

    Returns:
        RetentionResult describing what it purged and the before/after size.

    Raises:
        TypeError: If any argument is of the wrong type.
        ValueError: If ``max_bytes`` <= 0, ``headroom`` not in (0, 1), or
            ``event_max_age_s`` is not a positive finite number.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
        raise TypeError(f"max_bytes must be an int, got {type(max_bytes).__name__}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if not 0.0 < headroom < 1.0:
        raise ValueError(f"headroom must be in (0, 1), got {headroom}")
    if event_max_age_s is not None:
        if isinstance(event_max_age_s, bool) or not isinstance(
            event_max_age_s, int | float
        ):
            raise TypeError(
                "event_max_age_s must be a number or None, got "
                f"{type(event_max_age_s).__name__}"
            )
        age = float(event_max_age_s)
        if not age > 0.0 or age != age or age == float("inf"):
            raise ValueError(
                f"event_max_age_s must be a finite value > 0, got {event_max_age_s}"
            )
    if now is not None and not isinstance(now, _dt.datetime):
        raise TypeError(f"now must be a datetime or None, got {type(now).__name__}")

    log = logger or logging.getLogger("dcs_backend.data_capture")
    bytes_before = _db_size_bytes()
    footprint = historian_footprint(session, total_bytes=bytes_before)
    over_cap = footprint.total_bytes > max_bytes

    # Two separately-tracked budgets; neither table can spend the other's share.
    event_budget = max(1, int(max_bytes * EVENTLOG_BUDGET_FRACTION))
    tag_budget = max(1, max_bytes - event_budget)

    # --- EventLog: age pass first (cheapest), then its own size budget ------- #
    event_deleted = 0
    if event_max_age_s is not None:
        cutoff = ensure_utc(now or utcnow()) - _dt.timedelta(seconds=event_max_age_s)
        aged_out = _purge_events_older_than(session, cutoff)
        if aged_out:
            log.info(
                "Retention: purged %d event rows older than %s.",
                aged_out,
                _iso_utc(cutoff),
            )
        event_deleted += aged_out

    event_rows = int(
        session.exec(select(func.count()).select_from(EventLog)).one() or 0
    )
    over_budget = _purge_oldest_by_id(
        session,
        EventLog,
        rows=event_rows,
        current_bytes=footprint.eventlog_bytes,
        budget_bytes=event_budget,
        headroom=headroom,
    )
    if over_budget:
        log.warning(
            "Retention: event log over budget (%d > %d bytes); purged %d oldest rows.",
            footprint.eventlog_bytes,
            event_budget,
            over_budget,
        )
    event_deleted += over_budget

    # --- TagLog: charged only for its own pages ----------------------------- #
    tag_rows = int(session.exec(select(func.count()).select_from(TagLog)).one() or 0)
    tag_deleted = _purge_oldest_by_id(
        session,
        TagLog,
        rows=tag_rows,
        current_bytes=footprint.taglog_bytes,
        budget_bytes=tag_budget,
        headroom=headroom,
    )
    if tag_deleted:
        log.warning(
            "Retention: tag historian over budget (%d > %d bytes of a %d-byte "
            "cap, footprint %s); purged %d oldest rows.",
            footprint.taglog_bytes,
            tag_budget,
            max_bytes,
            "measured" if footprint.measured else "estimated",
            tag_deleted,
        )
    elif over_cap:
        # Neither table is over its own budget yet the file is over the cap: the
        # excess lives somewhere else (other tables, WAL, free pages). Say so
        # instead of silently shredding trend history to chase it.
        log.warning(
            "Retention: DB over cap (%d > %d bytes) but taglog (%d B) and "
            "eventlog (%d B) are both inside their budgets; reclaiming pages only.",
            footprint.total_bytes,
            max_bytes,
            footprint.taglog_bytes,
            footprint.eventlog_bytes,
        )

    return RetentionResult(
        over_cap=over_cap,
        rows_deleted=tag_deleted,
        event_rows_deleted=event_deleted,
        db_bytes_before=bytes_before,
        db_bytes_after=_db_size_bytes(),
        taglog_bytes_before=footprint.taglog_bytes,
        eventlog_bytes_before=footprint.eventlog_bytes,
    )


def utcnow() -> _dt.datetime:
    """Timezone-aware current UTC instant (wrapper for easy test patching)."""
    return _dt.datetime.now(UTC)
