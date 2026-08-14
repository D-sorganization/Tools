"""Store-and-forward shipping of historian samples to a remote plant historian.

One responsibility: get samples off the control node without ever letting the
remote destination influence the control node's timing.

Why a thread and not a coroutine
--------------------------------
``_poll_once`` calls the historian write path synchronously from inside the
async scan. Doing remote I/O there — even awaited — puts network latency on the
scan budget. At 10 Hz a single 2 s TCP timeout costs 20 scans and stalls the HMI
broadcast, alarm evaluation, and the E-stop re-engage path. That is a safety
regression, not a performance one.

So the producer (the scan loop) only ever does a bounded, non-blocking
``put_nowait`` onto an in-memory queue, and a daemon thread owns every socket
operation. The scan loop cannot block on the network by construction.

Delivery guarantees
-------------------
**At-most-once, and deliberately so.** The queue is in memory only; a process
restart discards whatever had not shipped. This is acceptable because SQLite
remains the authoritative local store — a restart loses *forwarding*, never
*data*. Backfilling the remote from SQLite is a separate concern and is not
attempted here. Do not build anything on an assumption of exactly-once.

Under sustained backpressure the queue drops the **oldest** samples. For process
history the newest data is the operationally useful data, and an unbounded queue
on a Pi with a gigabyte free is an out-of-memory crash of the control node —
which is a far worse outcome than a gap in a trend.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

__all__ = [
    "RemoteHistorianWriter",
    "Sample",
    "ShipperStats",
    "StoreAndForwardSink",
]

logger = logging.getLogger("dcs_backend.historian_shipper")

# A single measurement: when, which tag, what value.
Sample = tuple[datetime, str, float]

# Ceiling on reconnect backoff. Long enough that a historian down overnight
# costs ~120 reconnect attempts rather than ~29000, short enough that recovery
# after a transient blip is felt within a scan-or-two of operator patience.
_MAX_BACKOFF_S = 30.0
_INITIAL_BACKOFF_S = 0.5


@runtime_checkable
class RemoteHistorianWriter(Protocol):
    """The network-facing half of the shipper, owned entirely by the worker.

    Implementations are only ever touched from the shipper's worker thread, so
    they do not need to be thread-safe.
    """

    def connect(self) -> None:
        """Establish the connection. May raise; the shipper will back off."""
        ...

    def write_batch(self, samples: Sequence[Sample]) -> int:
        """Persist a batch. May raise; the shipper will reconnect and retry."""
        ...

    def close(self) -> None:
        """Release resources. Must be idempotent and must not raise."""
        ...


@dataclass(frozen=True)
class ShipperStats:
    """Point-in-time snapshot of shipper health.

    Exposed so a gap in a Grafana trend can be diagnosed as a *forwarding* gap
    rather than misread as a real process measurement — a flat line that is
    actually missing data is a genuine hazard for anyone reading a trend.
    """

    enabled: bool
    connected: bool
    queue_depth: int
    queue_max: int
    shipped_total: int
    dropped_total: int
    consecutive_failures: int
    last_success_ts: datetime | None = None
    lag_s: float | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable form for the health endpoint."""
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "queue_depth": self.queue_depth,
            "queue_max": self.queue_max,
            "shipped_total": self.shipped_total,
            "dropped_total": self.dropped_total,
            "consecutive_failures": self.consecutive_failures,
            "last_success_ts": (
                self.last_success_ts.isoformat() if self.last_success_ts else None
            ),
            "lag_s": self.lag_s,
            "last_error": self.last_error,
        }


@dataclass
class _Counters:
    """Mutable counters with a single writer thread each.

    ``dropped`` is written only by the producer (scan loop); ``shipped``,
    ``last_success``, ``failures``, ``connected`` and ``last_error`` only by the
    worker. Single-writer means ``+=`` needs no lock, and a reader tolerating a
    momentarily stale value is exactly what a health endpoint wants. This keeps
    the 10 Hz enqueue path free of lock contention.
    """

    dropped: int = 0
    shipped: int = 0
    failures: int = 0
    connected: bool = False
    last_success: datetime | None = None
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class StoreAndForwardSink:
    """A :class:`~historian_sink.HistorianSink` that forwards over the network.

    Satisfies the sink contract: :meth:`write_scan` never blocks on I/O, never
    raises in steady state, and may drop under backpressure.
    """

    def __init__(
        self,
        writer: RemoteHistorianWriter,
        *,
        queue_max: int = 100_000,
        batch_size: int = 1_000,
        flush_interval_s: float = 1.0,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        """Build a shipper. Call :meth:`start` to run it.

        Args:
            writer: The remote destination. Owned by the worker thread.
            queue_max: Bounded queue depth. Overflow drops oldest.
            batch_size: Maximum samples per remote round-trip.
            flush_interval_s: Maximum time a partial batch waits before shipping.
            jitter: Returns a value in [0, 1) used to spread reconnect attempts.

        Raises:
            TypeError: If ``writer`` does not implement
                :class:`RemoteHistorianWriter`, or a numeric argument is not
                numeric.
            ValueError: If ``queue_max`` or ``batch_size`` is < 1, or
                ``flush_interval_s`` is not positive and finite.
        """
        if not isinstance(writer, RemoteHistorianWriter):
            raise TypeError(
                "writer must implement RemoteHistorianWriter, "
                f"got {type(writer).__name__}"
            )
        queue_max = _positive_int("queue_max", queue_max)
        batch_size = _positive_int("batch_size", batch_size)
        flush_interval_s = _positive_float("flush_interval_s", flush_interval_s)
        if not callable(jitter):
            raise TypeError(f"jitter must be callable, got {type(jitter).__name__}")

        self._writer = writer
        self._queue: queue.Queue[Sample] = queue.Queue(maxsize=queue_max)
        self._queue_max = queue_max
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._jitter = jitter

        self._counters = _Counters()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the worker thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="historian-shipper",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Historian shipper started (queue_max=%d, batch_size=%d)",
            self._queue_max,
            self._batch_size,
        )

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Stop the worker and release the remote connection.

        Bounded by ``timeout_s`` so application shutdown can never hang on an
        unreachable historian. Idempotent; never raises.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                logger.warning(
                    "Historian shipper did not stop within %.1fs; "
                    "abandoning %d queued samples",
                    timeout_s,
                    self._queue.qsize(),
                )
        self._thread = None
        try:
            self._writer.close()
        except Exception:  # noqa: BLE001 - shutdown must not fail on the remote
            logger.debug("Remote historian close failed", exc_info=True)

    # ------------------------------------------------------------- sink surface

    def write_scan(self, tags: Mapping[str, float], timestamp: datetime) -> int:
        """Enqueue one scan's samples. Non-blocking; drops oldest when full.

        Args:
            tags: Mapping of tag name -> value.
            timestamp: Shared sample time for the scan.

        Returns:
            Number of samples enqueued (may be less than ``len(tags)`` only if
            a value was non-finite and skipped).
        """
        accepted = 0
        for name, value in tags.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                # A non-numeric tag is a local-historian problem and is already
                # rejected there with a hard error. Forwarding just skips it
                # rather than taking down the scan a second time.
                continue
            if not self._enqueue((timestamp, str(name), numeric)):
                continue
            accepted += 1
        return accepted

    def _enqueue(self, sample: Sample) -> bool:
        """Put with drop-oldest overflow. Never blocks, never raises."""
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            pass

        # Full: evict the oldest to make room. The get/put pair is not atomic,
        # but the only other consumer is the worker, which can only make more
        # room. Worst case the put still fails and we count a drop.
        try:
            self._queue.get_nowait()
            self._counters.dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            self._counters.dropped += 1
            return False

    # ------------------------------------------------------------------- worker

    def _run(self) -> None:
        """Worker loop: connect, drain, ship, back off on failure."""
        backoff = _INITIAL_BACKOFF_S
        while not self._stop.is_set():
            if not self._counters.connected:
                if not self._try_connect():
                    # Sleep on the stop event so shutdown is immediate rather
                    # than waiting out a 30 s backoff.
                    self._stop.wait(backoff * (0.5 + self._jitter()))
                    backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
                    continue
                backoff = _INITIAL_BACKOFF_S

            batch = self._collect_batch()
            if not batch:
                continue
            if not self._ship(batch):
                self._stop.wait(backoff * (0.5 + self._jitter()))
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)

        # Final best-effort flush of whatever is already queued.
        if self._counters.connected:
            final = self._collect_batch(blocking=False)
            if final:
                self._ship(final)

    def _try_connect(self) -> bool:
        try:
            self._writer.connect()
        except Exception as exc:  # noqa: BLE001 - any failure is a retry
            self._counters.failures += 1
            self._counters.last_error = f"{type(exc).__name__}: {exc}"
            # Rate-limited: only the first failure of an outage and then every
            # 10th, so a historian down overnight does not fill the Pi's disk
            # with identical log lines at 10 Hz.
            if self._counters.failures == 1 or self._counters.failures % 10 == 0:
                logger.warning(
                    "Historian shipper cannot connect (attempt %d): %s",
                    self._counters.failures,
                    exc,
                )
            return False
        self._counters.connected = True
        logger.info("Historian shipper connected")
        return True

    def _collect_batch(self, *, blocking: bool = True) -> list[Sample]:
        """Gather up to ``batch_size`` samples, waiting at most one interval."""
        batch: list[Sample] = []
        if blocking:
            try:
                batch.append(self._queue.get(timeout=self._flush_interval_s))
            except queue.Empty:
                return batch
        while len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _ship(self, batch: list[Sample]) -> bool:
        """Write one batch. On failure, mark disconnected and report."""
        try:
            self._writer.write_batch(batch)
        except Exception as exc:  # noqa: BLE001 - any failure is a reconnect
            self._counters.connected = False
            self._counters.failures += 1
            self._counters.last_error = f"{type(exc).__name__}: {exc}"
            if self._counters.failures == 1 or self._counters.failures % 10 == 0:
                logger.warning(
                    "Historian shipper failed to write %d samples "
                    "(failure %d); dropping batch: %s",
                    len(batch),
                    self._counters.failures,
                    exc,
                )
            # The batch is discarded rather than retried. Retrying in place
            # would stall the drain and let the queue overflow into dropping
            # *newer* data to preserve data we already know we cannot deliver.
            self._counters.dropped += len(batch)
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                logger.debug("Remote close during error recovery failed", exc_info=True)
            return False

        self._counters.shipped += len(batch)
        self._counters.last_success = datetime.now(UTC)
        self._counters.failures = 0
        self._counters.last_error = None
        return True

    # -------------------------------------------------------------- diagnostics

    def stats(self) -> ShipperStats:
        """Snapshot shipper health. Safe to call from any thread."""
        last = self._counters.last_success
        lag = (datetime.now(UTC) - last).total_seconds() if last else None
        return ShipperStats(
            enabled=True,
            connected=self._counters.connected,
            queue_depth=self._queue.qsize(),
            queue_max=self._queue_max,
            shipped_total=self._counters.shipped,
            dropped_total=self._counters.dropped,
            consecutive_failures=self._counters.failures,
            last_success_ts=last,
            lag_s=lag,
            last_error=self._counters.last_error,
        )


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _positive_float(name: str, value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    v = float(value)
    if v <= 0.0 or v != v or v == float("inf"):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v
