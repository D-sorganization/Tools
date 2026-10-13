"""TimescaleDB and FanOut dual-write historian sinks (issue #5543, H4 of #4046).

Architecture: Level 3 Plant Historian write backend.

Key design contracts:
1. Non-authoritative / Best Effort: TimescaleDB writes must never block, delay,
   or fail the control poll loop or SQLite primary writes.
2. Lazy Driver Import: ``asyncpg`` is imported lazily inside connection helpers
   so the base environment runs without requiring PostgreSQL drivers installed.
3. FanOut Composition: ``FanOutSink`` splits incoming scans across the authoritative
   primary (SQLite) and best-effort secondary (TimescaleDB). Any failure in the
   secondary is logged, increments a failure counter, and is otherwise suppressed.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections import deque
from collections.abc import Mapping
from typing import Any

from historian import HistorianRecord
from historian_sink import HistorianSink

logger = logging.getLogger("dcs_backend.timescale_sink")

__all__ = [
    "ALLOWED_HISTORIAN_BACKENDS",
    "DEFAULT_TIMESCALE_QUEUE_SIZE",
    "FanOutSink",
    "TimescaleConfig",
    "TimescaleSink",
    "create_historian_sink",
]

ALLOWED_HISTORIAN_BACKENDS: frozenset[str] = frozenset({"sqlite", "dual"})
DEFAULT_TIMESCALE_QUEUE_SIZE: int = 1024


class TimescaleConfig:
    """Connection parameters for TimescaleDB / PostgreSQL."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        sslmode: str | None = None,
    ) -> None:
        self.host = host or os.getenv("HISTORIAN_HOST", "localhost")
        self.port = port or int(os.getenv("HISTORIAN_PORT", "5432"))
        self.database = database or os.getenv("HISTORIAN_DB", "historian")
        self.user = user or os.getenv("HISTORIAN_USER", "historian_writer")
        self.password = password or os.getenv("HISTORIAN_PASSWORD", "")
        self.sslmode = sslmode or os.getenv("HISTORIAN_SSLMODE", "disable")


class TimescaleSink:
    """Asynchronous write sink for TimescaleDB hypertable ingestion.

    Implements :class:`HistorianSink`. Operates off-loop with an in-memory queue.
    Lazy imports the async database driver.
    """

    def __init__(
        self,
        config: TimescaleConfig | None = None,
        queue_size: int = DEFAULT_TIMESCALE_QUEUE_SIZE,
    ) -> None:
        self._config = config or TimescaleConfig()
        self._queue_size = queue_size
        self._queue: deque[HistorianRecord] = deque(maxlen=queue_size)
        self._pool: Any = None
        self._written_samples: int = 0
        self._dropped_samples: int = 0
        self._write_failures: int = 0
        self._closed: bool = False

    @property
    def is_connected(self) -> bool:
        """Whether a connection pool has been lazily established."""
        return self._pool is not None

    def submit(self, record: HistorianRecord) -> bool:
        """Submit a scan record to the internal queue without blocking."""
        if not isinstance(record, HistorianRecord):
            raise TypeError(f"record must be HistorianRecord, got {type(record)}")
        if self._closed:
            return False

        if len(self._queue) >= self._queue_size:
            self._dropped_samples += 1
            # Evict oldest to preserve fresh samples
            self._queue.popleft()

        self._queue.append(record)
        return True

    async def drain_once(self) -> int:
        """Drain queued records and batch insert into TimescaleDB."""
        if not self._queue or self._closed:
            return 0

        # Lazy import of driver
        try:
            import asyncpg  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "asyncpg driver not installed; dropping %d records",
                len(self._queue),
            )
            count = len(self._queue)
            self._dropped_samples += count
            self._write_failures += 1
            self._queue.clear()
            return 0

        batch: list[HistorianRecord] = []
        while self._queue and len(batch) < 100:
            batch.append(self._queue.popleft())

        if not batch:
            return 0

        try:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    host=self._config.host,
                    port=self._config.port,
                    database=self._config.database,
                    user=self._config.user,
                    password=self._config.password,
                    ssl=self._config.sslmode != "disable",
                    min_size=1,
                    max_size=5,
                )

            # Insert batch into tag_sample
            rows = []
            for rec in batch:
                if rec.tags:
                    ts = rec.timestamp if hasattr(rec, "timestamp") else None
                    for tag_name, val in rec.tags.items():
                        rows.append((ts, tag_name, float(val), 192))

            if rows and self._pool:
                async with self._pool.acquire() as conn:
                    # Ingest rows
                    await conn.copy_records_to_table(
                        "tag_sample",
                        records=rows,
                        columns=["ts", "tag_id", "value", "quality"],
                    )

            self._written_samples += len(rows)
            return len(batch)
        except Exception as exc:
            logger.warning("TimescaleDB batch insert failed: %s", exc)
            self._write_failures += 1
            self._dropped_samples += len(batch)
            return 0

    def stats(self) -> Mapping[str, Any]:
        """Diagnostic counters for TimescaleSink."""
        return {
            "queue_depth": len(self._queue),
            "written_samples": self._written_samples,
            "dropped_samples": self._dropped_samples,
            "write_failures": self._write_failures,
            "is_connected": self.is_connected,
        }

    def close(self) -> None:
        """Close connection pool and release resources."""
        self._closed = True
        if self._pool is not None:
            try:
                # Sync close hook if needed
                self._pool = None
            except Exception:
                pass


class FanOutSink:
    """Composite HistorianSink writing to primary and secondary sinks."""

    def __init__(self, primary: HistorianSink, secondary: HistorianSink) -> None:
        self._primary = primary
        self._secondary = secondary
        self._secondary_failures: int = 0

    @property
    def secondary_failures(self) -> int:
        """Count of errors caught and suppressed on the secondary sink."""
        return self._secondary_failures

    def submit(self, record: HistorianRecord) -> bool:
        """Submit to primary; forward to secondary with fault containment."""
        accepted = self._primary.submit(record)
        try:
            self._secondary.submit(record)
        except Exception as exc:
            logger.warning("Secondary historian sink submit failed: %s", exc)
            self._secondary_failures += 1
        return accepted

    async def drain_once(self) -> int:
        """Drain primary; drain secondary with fault containment."""
        drained_primary = self._primary.drain_once()
        count = (
            await drained_primary
            if inspect.isawaitable(drained_primary)
            else drained_primary
        )

        try:
            drained_secondary = self._secondary.drain_once()
            if inspect.isawaitable(drained_secondary):
                await drained_secondary
        except Exception as exc:
            logger.warning("Secondary historian sink drain failed: %s", exc)
            self._secondary_failures += 1

        return count

    def stats(self) -> Mapping[str, Any]:
        """Diagnostic counters merging primary and secondary status."""
        p_stats = dict(self._primary.stats())
        s_stats = dict(self._secondary.stats())
        return {
            **p_stats,
            "secondary_failures": self._secondary_failures,
            "secondary_stats": s_stats,
        }

    def close(self) -> None:
        """Close both sinks safely."""
        try:
            self._primary.close()
        finally:
            try:
                self._secondary.close()
            except Exception as exc:
                logger.warning("Secondary sink close error: %s", exc)


def create_historian_sink(
    backend: str,
    primary: HistorianSink,
    timescale_sink: HistorianSink | None = None,
) -> HistorianSink:
    """Factory creating the appropriate HistorianSink according to the backend strategy.

    Args:
        backend: 'sqlite' or 'dual'.
        primary: Authoritative SQLite HistorianSink (e.g. HistorianWriter).
        timescale_sink: Optional explicit TimescaleSink instance.

    Returns:
        The configured HistorianSink.

    Raises:
        ValueError: If backend is not 'sqlite' or 'dual'.
    """
    clean = backend.strip().lower()
    if clean == "sqlite":
        return primary
    if clean == "dual":
        secondary = timescale_sink if timescale_sink is not None else TimescaleSink()
        return FanOutSink(primary=primary, secondary=secondary)

    raise ValueError(
        f"Invalid historian backend '{backend}'. Allowed values: 'sqlite', 'dual'"
    )
