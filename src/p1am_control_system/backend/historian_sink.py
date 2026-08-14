"""Pluggable historian write backends.

One responsibility: define the seam between "a scan happened" and "somewhere
durable learned about it", so the local SQLite historian and a remote plant
historian (TimescaleDB) can both be fed without either knowing about the other.

Design constraint that shapes this module
-----------------------------------------
``poll_runtime._poll_once`` writes historian rows *and* alarm-event rows on one
SQLAlchemy session and commits them together. That shared commit is deliberate:
it makes a scan atomic in the local database, so a crash can never leave an
alarm event without the sample that triggered it. A sink that owned its own
session would silently break that atomicity.

So the split is:

* The **local** write stays exactly where it is, on the caller's session, via
  :func:`historian.log_scan`. It is the source of truth and cannot be skipped.
* A :class:`HistorianSink` is a **forwarding** interface only. It receives a
  copy of the scan and is free to be remote, queued, lossy, or absent. It is
  never permitted to affect the local write or the poll loop.

:class:`HistorianWriter` composes the two behind the exact callable shape
``_poll_once`` already expects, so the control path is untouched.

LOD: this module imports only ``historian``, the ``signal_quality`` value type,
and stdlib — nothing from FastAPI, the PLC clients, or the database engine — so
it unit-tests against a plain session double. (``historian`` itself imports
``signal_quality``, so this adds no new dependency edge to the package.)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import historian

try:
    from datetime import UTC
except ImportError:  # Python 3.10 — repo supports 3.10+
    UTC = timezone.utc  # noqa: UP017

from signal_quality import SignalFrame
from sqlmodel import Session

__all__ = [
    "HistorianSink",
    "HistorianWriter",
    "NullHistorianSink",
]

logger = logging.getLogger("dcs_backend.historian_sink")


@runtime_checkable
class HistorianSink(Protocol):
    """A best-effort destination for forwarded scan samples.

    Implementations MUST treat every method as non-throwing from the caller's
    perspective in steady state, and MUST NOT block for longer than a scan
    period. A sink that needs to do network I/O is expected to enqueue and
    return, not to perform the I/O inline (see
    :mod:`historian_shipper`).

    Implementations MAY drop samples under backpressure. Loss of *forwarded*
    data is acceptable; loss of *local* data is not, and the local write is not
    routed through this interface.
    """

    def write_scan(self, tags: Mapping[str, float], timestamp: datetime) -> int:
        """Accept one scan's samples for forwarding.

        Args:
            tags: Mapping of tag name -> value for this scan.
            timestamp: The single sample time shared by every tag in the scan.

        Returns:
            Number of samples accepted. May be 0 if the sink dropped them.
        """
        ...

    def close(self) -> None:
        """Release resources. Must be idempotent and must not raise."""
        ...


class NullHistorianSink:
    """The default sink: accepts everything, does nothing, never fails.

    Used when remote forwarding is disabled so the write path has no branch and
    no ``None`` check in the hot loop.
    """

    __slots__ = ()

    def write_scan(self, tags: Mapping[str, float], timestamp: datetime) -> int:
        """Discard the scan. Returns 0 — nothing was forwarded anywhere."""
        return 0

    def close(self) -> None:
        """No-op."""
        return None


class HistorianWriter:
    """Throttled local historian write plus best-effort remote forwarding.

    Exposes :meth:`write`, which matches the
    ``Callable[[Session, dict[str, float]], int]`` shape that
    ``poll_runtime._poll_once`` already accepts, so wiring this in requires no
    change to the control path.

    Ordering guarantee: the local write happens first and its result is what is
    returned. Forwarding happens after, and any failure there is swallowed. A
    broken remote historian can therefore never reduce local durability or
    surface an error into the scan loop.

    Both destinations receive the *same* timestamp for a given scan, so a sample
    can be correlated across the two stores exactly rather than approximately.

    The throttle is consulted exactly once per :meth:`write` call. Local and
    remote are written in lockstep — a scan is either captured to both or to
    neither — which keeps the two stores directly comparable and keeps the
    remote volume predictable from the operator-facing capture interval.
    """

    def __init__(
        self,
        *,
        due: Callable[[], bool],
        sink: HistorianSink | None = None,
        log_scan: Callable[..., int] = historian.log_scan,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Build a writer.

        Args:
            due: Predicate consulted once per scan to decide whether to persist.
                Typically ``CaptureThrottle.due``. Calling it is expected to
                have the side effect of consuming the throttle window, so it is
                called at most once per :meth:`write`.
            sink: Forwarding destination. ``None`` means no forwarding.
            log_scan: The local bulk-insert primitive. Injected for tests.
            clock: Returns the aware-UTC sample time for a scan. Injected for
                tests.

        Raises:
            TypeError: If ``due``, ``log_scan``, or ``clock`` is not callable,
                or ``sink`` is neither ``None`` nor a ``HistorianSink``.
        """
        if not callable(due):
            raise TypeError(f"due must be callable, got {type(due).__name__}")
        if not callable(log_scan):
            raise TypeError(f"log_scan must be callable, got {type(log_scan).__name__}")
        if not callable(clock):
            raise TypeError(f"clock must be callable, got {type(clock).__name__}")
        if sink is not None and not isinstance(sink, HistorianSink):
            raise TypeError(
                f"sink must implement HistorianSink, got {type(sink).__name__}"
            )

        self._due = due
        self._sink: HistorianSink = sink if sink is not None else NullHistorianSink()
        self._log_scan = log_scan
        self._clock = clock

    @property
    def sink(self) -> HistorianSink:
        """The configured forwarding sink (never ``None``)."""
        return self._sink

    def write(
        self,
        session: Session,
        tags: dict[str, float],
        *,
        signal_frame: SignalFrame | None = None,
    ) -> int:
        """Persist a scan locally when due, then forward it best-effort.

        Args:
            session: Active session owned by the caller. Not committed here —
                the caller commits historian and alarm rows together.
            tags: Mapping of tag name -> value for this scan.
            signal_frame: Per-scan signal-quality metadata. Forwarded verbatim
                to the local write so quality is persisted with the sample.
                Deliberately NOT passed to the remote sink: the sink contract is
                a plain ``{tag: value}`` + timestamp forward, and widening it
                would make every sink implementation depend on the quality
                model.

        Returns:
            Number of rows written to the **local** historian. 0 when the
            throttle declined the scan. The forwarding result is deliberately
            not reflected here: callers must not be able to confuse "the plant
            historian is unreachable" with "nothing was recorded".
        """
        if not self._due():
            return 0

        ts = self._clock()
        written = self._log_scan(session, tags, timestamp=ts, signal_frame=signal_frame)

        # Forwarding is best-effort by contract. A remote historian that is
        # down, slow, or misconfigured must never propagate into the scan loop,
        # so every exception stops here. Sinks are additionally expected to
        # rate-limit their own logging; this guard is the last resort.
        try:
            self._sink.write_scan(tags, ts)
        except Exception:  # noqa: BLE001 - deliberate isolation boundary
            logger.debug("Historian forwarding failed", exc_info=True)

        return written

    def close(self) -> None:
        """Close the forwarding sink. Never raises."""
        try:
            self._sink.close()
        except Exception:  # noqa: BLE001 - shutdown must not fail on the sink
            logger.debug("Historian sink close failed", exc_info=True)
