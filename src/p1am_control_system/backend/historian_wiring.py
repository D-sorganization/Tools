"""Assembly of the historian write path from settings.

One responsibility: decide, from configuration alone, what the scan loop's
historian writer should be — and make that decision testable without standing up
FastAPI, a PLC, or a database.

Keeping this out of ``main.py`` means the wiring can be unit-tested directly;
``main`` only calls :func:`build_historian_writer` and holds the result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from historian_shipper import ShipperStats, StoreAndForwardSink
from historian_sink import HistorianWriter
from settings import P1AMSettings, get_settings

__all__ = ["build_historian_writer", "shipper_stats"]

logger = logging.getLogger("dcs_backend.historian_wiring")

# Reported when forwarding is switched off, so the health surface always answers
# the same shape and a dashboard does not need a null branch.
_DISABLED_STATS = ShipperStats(
    enabled=False,
    connected=False,
    queue_depth=0,
    queue_max=0,
    shipped_total=0,
    dropped_total=0,
    consecutive_failures=0,
)


def build_historian_writer(
    due: Callable[[], bool],
    settings: P1AMSettings | None = None,
) -> tuple[HistorianWriter, StoreAndForwardSink | None]:
    """Build the scan-loop historian writer and, if enabled, the shipper.

    Args:
        due: Throttle predicate consulted once per scan — normally
            ``CaptureThrottle.due``.
        settings: Configuration. Defaults to the process settings.

    Returns:
        ``(writer, shipper)``. ``shipper`` is ``None`` when remote forwarding is
        disabled, in which case nothing is imported, no thread is started, and
        no socket is opened.

    Raises:
        TypeError: If ``due`` is not callable.
    """
    if not callable(due):
        raise TypeError(f"due must be callable, got {type(due).__name__}")

    resolved = settings if settings is not None else get_settings()

    if not resolved.timescale_enabled:
        logger.info("Remote plant historian forwarding disabled (SQLite only)")
        return HistorianWriter(due=due), None

    # Imported here rather than at module scope so a bench Pi without a
    # Postgres driver installed never pays for it — and never fails to boot
    # because of it.
    from timescale_writer import TimescaleWriter  # noqa: PLC0415

    remote = TimescaleWriter(
        resolved.timescale_dsn,
        connect_timeout_s=resolved.timescale_connect_timeout_s,
    )
    shipper = StoreAndForwardSink(
        remote,
        queue_max=resolved.timescale_queue_max,
        batch_size=resolved.timescale_batch_size,
        flush_interval_s=resolved.timescale_flush_interval_s,
    )
    shipper.start()
    logger.info("Remote plant historian forwarding enabled -> %s", remote.safe_dsn)
    return HistorianWriter(due=due, sink=shipper), shipper


def shipper_stats(shipper: StoreAndForwardSink | None) -> ShipperStats:
    """Return shipper health, or a disabled snapshot when not forwarding."""
    return shipper.stats() if shipper is not None else _DISABLED_STATS
