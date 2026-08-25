"""Time-series historian write path and background batching for the poll loop.

One responsibility: persist a scan's worth of tag samples and events as cheaply
and safely as possible. A single bulk INSERT replaces per-tag inserts, and
:class:`HistorianWriter` moves SQLite writes off the asyncio event loop into a
batched, retrying worker thread so a ``VACUUM`` lock or slow flash write cannot
stall the control loop or drop alarm transitions (issue #4023).

It also defines:
* :class:`HistorianRecord` — immutable scan payload of tags, events, and quality.
* :class:`ThrottledHistorianSink` — strips un-due samples before enqueueing.
* :func:`log_scan` — bulk insert helper for active sessions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from models import DataSource, TagLog
from sqlalchemy import insert
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

logger = logging.getLogger("dcs_backend.historian")

__all__ = [
    "DEFAULT_HISTORIAN_BATCH_MAX",
    "DEFAULT_HISTORIAN_QUEUE_SIZE",
    "HistorianRecord",
    "HistorianWriter",
    "ThrottledHistorianSink",
    "_WriterCounters",
    "log_scan",
]

#: Default bound on the historian queue. At 10 Hz this is ~25 s of buffer —
#: long enough to ride out a VACUUM, short enough to bound memory on the Pi.
DEFAULT_HISTORIAN_QUEUE_SIZE = 256
#: Records folded into a single transaction by the writer task.
DEFAULT_HISTORIAN_BATCH_MAX = 32


@dataclass(frozen=True)
class HistorianRecord:
    """One scan's worth of persistable work.

    ``tags`` is None whenever no sample is due (or none may be trusted), so the
    writer can skip opening a session entirely. ``events`` are alarm/quality
    transitions, which are NOT resamplable and are therefore retried.
    """

    tags: Mapping[str, float] | None
    events: tuple[Any, ...] = ()
    quality: str = DataSource.LIVE.value

    def __post_init__(self) -> None:
        if self.tags is not None and not isinstance(self.tags, Mapping):
            raise TypeError(f"tags must be a mapping or None, got {type(self.tags)}")
        if not isinstance(self.events, tuple):
            raise TypeError(f"events must be a tuple, got {type(self.events).__name__}")
        if not isinstance(self.quality, str):
            raise TypeError(f"quality must be a str, got {type(self.quality).__name__}")
        if not self.quality.strip():
            raise ValueError("quality must be a non-empty string")

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to persist (no session need be opened)."""
        return not self.tags and not self.events


@dataclass
class _WriterCounters:
    """Mutable diagnostics for :class:`HistorianWriter` (surfaced on the API)."""

    write_failures: int = 0
    dropped_samples: int = 0
    rows_written: int = 0
    events_written: int = 0
    retries: int = 0


def log_scan(
    session: Session,
    tags: dict[str, float],
    *,
    timestamp: datetime | None = None,
    quality: str = DataSource.LIVE.value,
) -> int:
    """Bulk-insert one scan's tag samples; return the number of rows written.

    Args:
        session: An active SQLModel session. The caller owns the transaction
            (this does not commit) so tag and alarm writes share one commit.
        tags: Mapping of tag name -> value for this scan.
        timestamp: Sample time for every row; defaults to now (UTC). One shared
            timestamp keeps a scan atomic in time and avoids 32 clock reads.
        quality: Provenance stamped on every row (see ``models.DataSource``).
            The caller must not persist held or faulted scans at all — a gap is
            the truthful record of an outage (issue #4004).

    Returns:
        Number of rows inserted (0 for an empty mapping).

    Raises:
        TypeError: If ``session`` is not a Session, ``tags`` is not a dict,
            ``timestamp`` is not a datetime/None, or ``quality`` is not a str.
        ValueError: If ``quality`` is blank or any tag value is not
            finite/convertible to float.
    """
    if not isinstance(session, Session):
        raise TypeError(f"session must be a Session, got {type(session).__name__}")
    if not isinstance(tags, dict):
        raise TypeError(f"tags must be a dict, got {type(tags).__name__}")
    if timestamp is not None and not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be a datetime or None, got {type(timestamp)}")
    if not isinstance(quality, str):
        raise TypeError(f"quality must be a str, got {type(quality).__name__}")
    if not quality.strip():
        raise ValueError("quality must be a non-empty string")

    if not tags:
        return 0

    ts = timestamp if timestamp is not None else datetime.now(UTC)

    rows = []
    for name, value in tags.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tag {name!r} has non-numeric value {value!r}") from exc
        rows.append(
            {
                "tag_name": str(name),
                "value": numeric,
                "timestamp": ts,
                "quality": str(quality),
            }
        )

    session.execute(insert(TagLog), rows)
    return len(rows)


class HistorianWriter:
    """Batched, off-loop historian writer with alarm-preserving backpressure.

    The scan loop calls :meth:`submit` (non-blocking) and moves on. A dedicated
    task drains the bounded queue, folds several scans into ONE transaction and
    executes it via ``asyncio.to_thread`` so blocking SQLite never runs on the
    event loop (issue #4023).

    Backpressure policy mirrors the physics: tag samples are resamplable and the
    oldest is dropped when the queue fills, but the dropped record's alarm and
    data-quality events are carried forward onto the newcomer — a transition
    happens once and cannot be recovered by waiting for the next scan.
    """

    def __init__(
        self,
        session_factory: Callable[[], Iterator[Session]],
        *,
        queue_size: int = DEFAULT_HISTORIAN_QUEUE_SIZE,
        batch_max: int = DEFAULT_HISTORIAN_BATCH_MAX,
        retry_attempts: int = 3,
        retry_delay_s: float = 0.2,
        poll_timeout_s: float = 0.5,
        log_scan: Callable[..., int] = log_scan,
        to_thread: Callable[..., Any] = asyncio.to_thread,
        sleep: Callable[[float], Any] = time.sleep,
        async_sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not callable(log_scan):
            raise TypeError("log_scan must be callable")
        for name, value in (
            ("queue_size", queue_size),
            ("batch_max", batch_max),
            ("retry_attempts", retry_attempts),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        self._session_factory = session_factory
        self._queue: asyncio.Queue[HistorianRecord] = asyncio.Queue(maxsize=queue_size)
        self._batch_max = batch_max
        self._retry_attempts = retry_attempts
        self._retry_delay_s = float(retry_delay_s)
        self._poll_timeout_s = float(poll_timeout_s)
        self._log_scan = log_scan
        self._to_thread = to_thread
        self._sleep = sleep
        # Bound at construction so a test that monkeypatches ``asyncio.sleep``
        # for the scan loop cannot also hijack the writer's idle poll.
        self._async_sleep = async_sleep
        self._counters = _WriterCounters()

    # -- diagnostics ------------------------------------------------------
    @property
    def write_failures(self) -> int:
        """Batches abandoned after exhausting their retries."""
        return self._counters.write_failures

    @property
    def dropped_samples(self) -> int:
        """Tag samples discarded because the queue was full."""
        return self._counters.dropped_samples

    @property
    def rows_written(self) -> int:
        return self._counters.rows_written

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def rebind(self) -> None:
        """Recreate the outbox for the loop that is about to drain it.

        ``asyncio.Queue`` binds to the first event loop that touches it, so a
        writer constructed at import time (or left over from a previous run)
        must be re-armed before a new loop starts polling. Counters survive —
        they are process-lifetime diagnostics, not per-run state.
        """
        self._queue = asyncio.Queue(maxsize=self._queue.maxsize)

    # -- producer side ----------------------------------------------------
    def submit(self, record: HistorianRecord) -> bool:
        """Enqueue a record without blocking; return False if it was coalesced.

        Raises:
            TypeError: if ``record`` is not a :class:`HistorianRecord`.
        """
        if not isinstance(record, HistorianRecord):
            raise TypeError(
                f"record must be a HistorianRecord, got {type(record).__name__}"
            )
        if record.is_empty:
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except asyncio.QueueFull:
            pass
        # Full: evict the oldest sample but rescue its (non-resamplable) events.
        rescued: tuple[Any, ...] = ()
        try:
            evicted = self._queue.get_nowait()
            rescued = evicted.events
            self._counters.dropped_samples += 1
        except asyncio.QueueEmpty:  # pragma: no cover - racy drain
            pass
        merged = (
            record
            if not rescued
            else HistorianRecord(
                tags=record.tags,
                events=rescued + record.events,
                quality=record.quality,
            )
        )
        try:
            self._queue.put_nowait(merged)
        except asyncio.QueueFull:  # pragma: no cover - racy producer
            logger.error("Historian queue still full; dropped a scan record.")
            return False
        return False

    # -- consumer side ----------------------------------------------------
    async def drain_once(self) -> int:
        """Write one batch; return how many records it contained (0 if idle)."""
        batch: list[HistorianRecord] = []
        try:
            batch.append(self._queue.get_nowait())
        except asyncio.QueueEmpty:
            return 0
        while len(batch) < self._batch_max:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._to_thread(self._write_batch, tuple(batch))
        return len(batch)

    async def run(self, stop_event: asyncio.Event) -> None:
        """Drain until ``stop_event`` is set, then flush what is left.

        Raises:
            TypeError: if ``stop_event`` is not an asyncio.Event.
        """
        if not isinstance(stop_event, asyncio.Event):
            raise TypeError(
                f"stop_event must be an asyncio.Event, got {type(stop_event).__name__}"
            )
        logger.info("Historian writer task started.")
        while not stop_event.is_set():
            if await self.drain_once() == 0:
                # Poll rather than await stop_event.wait(): an Event binds to the
                # first loop that waits on it, and this module-level writer is
                # reused across runs (and across test event loops).
                await self._async_sleep(self._poll_timeout_s)
        # Best-effort flush so a clean shutdown does not discard buffered rows.
        while await self.drain_once():
            pass
        logger.info("Historian writer task stopped.")

    # -- worker-thread body ----------------------------------------------
    def _write_batch(self, batch: Sequence[HistorianRecord]) -> None:
        """Commit a batch in ONE transaction, retrying a locked database.

        Runs in a worker thread. Never raises: a historian failure must not be
        able to take down the control loop, but it is always counted and logged
        so it cannot pass silently (the #4023 defect).
        """
        samples = [r for r in batch if r.tags]
        events = [ev for r in batch for ev in r.events]
        if not samples and not events:
            return
        for attempt in range(1, self._retry_attempts + 1):
            try:
                self._commit_once(samples, events)
                return
            except OperationalError as exc:
                last_attempt = attempt >= self._retry_attempts
                if last_attempt:
                    self._counters.write_failures += 1
                    logger.error(
                        "Historian batch abandoned after %d attempts "
                        "(%d samples, %d events): %s",
                        attempt,
                        len(samples),
                        len(events),
                        exc,
                    )
                    self._log_unpersisted_events(events)
                    return
                self._counters.retries += 1
                logger.warning(
                    "Historian database busy (attempt %d/%d); retrying: %s",
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                self._sleep(self._retry_delay_s)
            except Exception as exc:  # noqa: BLE001 - never kill the writer task
                self._counters.write_failures += 1
                logger.error("Historian batch failed permanently: %s", exc)
                self._log_unpersisted_events(events)
                return

    @staticmethod
    def _log_unpersisted_events(events: Sequence[Any]) -> None:
        """Emit any alarm/quality row that could not be stored, to the app log.

        A transition happens once. If the database refuses it, the process log
        is the last place it can still be recovered from during an incident
        review — losing it silently is exactly the #4023 defect.
        """
        for event in events:
            logger.error(
                "UNPERSISTED %s event: %s",
                getattr(event, "event_type", "?"),
                getattr(event, "description", event),
            )

    def _commit_once(
        self, samples: Sequence[HistorianRecord], events: Sequence[Any]
    ) -> None:
        """Open a session, write the batch and commit; rollback and re-raise."""
        gen = self._session_factory()
        session = next(iter(gen))
        try:
            written = 0
            for record in samples:
                written += self._log_scan(
                    session, dict(record.tags or {}), quality=record.quality
                )
            for event in events:
                session.add(event)
            session.commit()
            self._counters.rows_written += written
            self._counters.events_written += len(events)
        except Exception:
            try:
                session.rollback()
            except Exception as rollback_err:  # pragma: no cover - defensive
                logger.error("Historian rollback failed: %s", rollback_err)
            raise
        finally:
            self._close(gen, session)

    @staticmethod
    def _close(gen: Any, session: Any) -> None:
        """Release the session, preferring the factory generator's own teardown."""
        closer = getattr(gen, "close", None)
        if callable(closer):
            closer()
            return
        session_close = getattr(session, "close", None)
        if callable(session_close):
            session_close()


class ThrottledHistorianSink:
    """Applies a sampling throttle before a record reaches the writer queue.

    Throttling HERE means a scan with nothing due never enters the queue and
    never opens a session — the #4023 defect was opening and tearing one down
    every scan even when the capture throttle suppressed the write. Alarm and
    data-quality events are never throttled: they are transitions, not samples.
    """

    def __init__(self, writer: Any, is_sample_due: Callable[[], bool]) -> None:
        if not callable(getattr(writer, "submit", None)):
            raise TypeError("writer must expose a callable submit(record)")
        if not callable(is_sample_due):
            raise TypeError("is_sample_due must be callable")
        self._writer = writer
        self._is_sample_due = is_sample_due

    def submit(self, record: HistorianRecord) -> None:
        """Forward ``record``, stripping its tag sample when none is due.

        Raises:
            TypeError: if ``record`` is not a :class:`HistorianRecord`.
        """
        if not isinstance(record, HistorianRecord):
            raise TypeError(
                f"record must be a HistorianRecord, got {type(record).__name__}"
            )
        if record.tags and not self._is_sample_due():
            record = HistorianRecord(
                tags=None, events=record.events, quality=record.quality
            )
        if record.is_empty:
            return
        self._writer.submit(record)
