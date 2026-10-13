"""Pluggable historian write backend protocol (issue #5540, H1 of #4046).

One responsibility: define the seam between the control scan loop and durable
time-series storage backends.

By design:
* :meth:`submit` is non-blocking and safe to call directly from the scan loop;
  backends queue or coalesce samples and never block the poll cycle or delay
  de-energizing safety actions.
* :meth:`drain_once` drains or writes one batch of queued records.
* :meth:`stats` surfaces diagnostic counters (e.g. written rows, dropped samples).
* :meth:`close` releases backend resources idempotently without raising.
"""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Protocol, runtime_checkable

from historian import HistorianRecord

__all__ = ["HistorianSink"]


@runtime_checkable
class HistorianSink(Protocol):
    """Protocol for pluggable historian write backends.

    Implementations MUST ensure :meth:`submit` is non-blocking and safe to call
    from the poll loop. Backpressure policy is implementation-defined (e.g.
    dropping resamplable tag values while preserving discrete alarm events).
    """

    def submit(self, record: HistorianRecord) -> bool:
        """Submit a scan record to the sink without blocking.

        Args:
            record: The scan payload of tags, events, and quality provenance.

        Returns:
            True if the record was accepted directly into the queue/buffer;
            False if it was dropped, coalesced, or rejected.

        Raises:
            TypeError: If record is not a :class:`HistorianRecord`.
        """
        ...

    def drain_once(self) -> Awaitable[int] | int:
        """Drain or write one batch; return count of drained records (0 if idle)."""
        ...

    def stats(self) -> Mapping[str, int]:
        """Return a mapping of diagnostic counters and health metrics."""
        ...

    def close(self) -> None:
        """Release sink resources. Must be idempotent and never raise."""
        ...
