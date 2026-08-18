"""Single-attempt runtime helpers for the P1AM backend loops.

The FastAPI shell owns lifecycle and cadence. This module owns one PLC
connection attempt and one PLC polling scan so safety and persistence contracts
can be tested without sleeping in infinite loops.

It also owns the two seams that keep a scan honest and cheap:

* :class:`DataQualityTracker` — turns a change of tag provenance into an
  auditable event, so an outage appears in the log instead of as a silent gap.
* :class:`HistorianWriter` — moves the SQLite write off the event loop into a
  batched, retrying worker so a ``VACUUM`` lock can no longer lose an alarm
  transition or stall the control loop (issue #4023).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alarm_processing import process_alarm_events
from historian import log_scan as _default_log_scan
from models import DATA_SOURCE_SEVERITY, DataSource, EventLog, RoutingConfig
from power_supply_passthrough import ensure_power_supply_passthrough
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

logger = logging.getLogger("dcs_backend.poll_runtime")

__all__ = [
    "DataQualityTracker",
    "HistorianRecord",
    "HistorianWriter",
    "POLL_FAILURE_ESCALATION_THRESHOLD",
    "POLL_FAILURE_MAX_BACKOFF_S",
    "ThrottledHistorianSink",
    "log_poll_failure",
    "loop_diagnostics",
]

#: Default bound on the historian queue. At 10 Hz this is ~25 s of buffer —
#: long enough to ride out a VACUUM, short enough to bound memory on the Pi.
DEFAULT_HISTORIAN_QUEUE_SIZE = 256
#: Records folded into a single transaction by the writer task.
DEFAULT_HISTORIAN_BATCH_MAX = 32


def _resolve_source(
    *,
    plc_connected: bool,
    tags_read: bool,
    have_held: bool,
    simulated: bool,
) -> DataSource:
    """Classify this scan's data provenance.

    On a simulator driver every value is, by construction, simulated — the
    operator chose that. On real hardware a value that was not read this scan is
    NOT a measurement: it is either the last good reading being held for display
    or nothing at all, and either way the control path must be told so.
    """
    if tags_read:
        return DataSource.SIMULATED if simulated else DataSource.LIVE
    if simulated:
        return DataSource.SIMULATED if have_held else DataSource.FAULT
    if plc_connected and have_held:
        return DataSource.HELD
    return DataSource.FAULT


class DataQualityTracker:
    """Emits an EventLog row whenever the poll loop's data provenance changes.

    Stateful across scans (the loop owns one instance) and deliberately silent
    while the source is unchanged, so a sustained outage costs one row, not ten
    per second.
    """

    def __init__(self) -> None:
        self._source: DataSource | None = None

    @property
    def source(self) -> DataSource | None:
        """The most recently observed source, or None before the first scan."""
        return self._source

    def observe(self, source: str) -> EventLog | None:
        """Record ``source``; return an EventLog row only on a transition.

        Raises:
            TypeError: if ``source`` is not a string.
            ValueError: if ``source`` is not a known DataSource value.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be a str, got {type(source).__name__}")
        try:
            resolved = DataSource(source)
        except ValueError as exc:
            raise ValueError(f"unknown data source {source!r}") from exc
        if resolved == self._source:
            return None
        previous = self._source
        self._source = resolved
        severity = DATA_SOURCE_SEVERITY.get(resolved.value, 0)
        description = (
            f"PLC data source changed from {previous.value if previous else 'unknown'}"
            f" to {resolved.value}."
        )
        if resolved == DataSource.FAULT:
            description += (
                " Control laws, alarm evaluation and historian sampling are"
                " suspended until a live reading returns."
            )
        elif resolved == DataSource.HELD:
            description += " Last good values are displayed but not controlled on."
        return EventLog(
            event_type="DATA_QUALITY",
            description=description,
            severity=severity,
        )


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
        log_scan: Callable[..., int] = _default_log_scan,
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


def _reengage_service_estop(service: Any) -> None:
    """Re-latch a service controller's E-stop if it exposes the seam.

    Defensive (LOD): the poll/connect helpers only depend on the small public
    ``engage_estop`` / ``set_estop_active`` seam, and tolerate doubles that omit
    it. Both are idempotent, so re-calling while already latched is safe.
    """
    engage = getattr(service, "engage_estop", None)
    if callable(engage):
        engage()
    _set_service_estop_flag(service, True)


def _set_service_estop_flag(service: Any, active: bool) -> None:
    """Set only a service's low-level write-seam interlock flag, if present.

    Does NOT touch the controller's own E-stop latch (that is one-way and only
    an operator reset clears it). Tolerates doubles without the seam (LOD).
    """
    set_flag = getattr(service, "set_estop_active", None)
    if callable(set_flag):
        set_flag(active)


#: Consecutive failed scans before the loop logs a single degraded warning and
#: publishes a degraded snapshot; further failures drop to debug.
POLL_FAILURE_ESCALATION_THRESHOLD = 3
#: Ceiling on the exponential failure backoff, in seconds.
POLL_FAILURE_MAX_BACKOFF_S = 5.0


def log_poll_failure(failures: int, retry_delay_s: float, err: Exception) -> None:
    """Log a failed scan, escalating once and then falling silent.

    One message template (DRY): early failures are errors, the threshold
    crossing is a single warning, and a sustained outage drops to debug so a
    10 Hz loop cannot flood the journal.
    """
    if failures < POLL_FAILURE_ESCALATION_THRESHOLD:
        level, phase = logging.ERROR, "PLC polling loop error"
    elif failures == POLL_FAILURE_ESCALATION_THRESHOLD:
        level, phase = logging.WARNING, "PLC polling loop degraded"
    else:
        level, phase = logging.DEBUG, "PLC polling loop still degraded"
    logger.log(
        level,
        "%s after %d consecutive failures; retrying in %.3fs: %s",
        phase,
        failures,
        retry_delay_s,
        err,
    )


def loop_diagnostics(
    *, scheduler: Any, perf: Any, writer: Any, ws: Any
) -> dict[str, Any]:
    """Loop-health counters published on every frame and on /api/performance.

    Deliberately duck-typed and side-effect free, so the shape of what an
    operator sees is asserted by a unit test rather than by running the app.
    """
    return {
        "scan_interval_s": scheduler.period_s,
        "scan_overruns": scheduler.overrun_count,
        "last_overrun_s": scheduler.last_overrun_s,
        "broadcast_every_n": perf.broadcast_every_n,
        "historian_write_failures": writer.write_failures,
        "historian_samples_dropped": writer.dropped_samples,
        "ws_frames_dropped": ws.frames_dropped,
    }


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


async def _connect_once(
    *,
    plc: Any,
    power_supply: Any,
    apply_config: Callable[[RoutingConfig], None],
    estop_active: bool,
    temperature: Any = None,
    ensure_passthrough: Callable[..., Any] = ensure_power_supply_passthrough,
) -> RoutingConfig | None:
    """Attempt one background PLC connection and routing sync.

    When ``estop_active`` is set, the hardware E-stop is re-asserted AND the
    process-local controller latches are re-engaged on reconnect: a fresh PLC
    connection must not let the next poll re-command an output before the
    controllers are re-latched. Re-latching is idempotent and best-effort.
    """
    if plc.connected:
        return None

    connected = await plc.connect()
    if not connected:
        return None

    logger.info("Connected to PLC successfully in background.")
    if estop_active:
        # Re-latch the process-local controllers and low-level write-seam
        # interlock BEFORE anything can command an output, so the reconnect
        # cannot transiently re-energize a heater relay / AO setpoint.
        _reengage_service_estop(power_supply)
        if temperature is not None:
            _reengage_service_estop(temperature)
        set_plc_flag = getattr(plc, "set_estop_active", None)
        if callable(set_plc_flag):
            set_plc_flag(True)
        try:
            await plc.trigger_estop()
            logger.warning("Re-asserted hardware E-stop on reconnect.")
        except Exception as estop_err:
            logger.error(f"Failed to re-assert E-stop: {estop_err}")

    try:
        plc_config = await plc.read_routing()
        if plc_config is None:
            return None
        plc_config = await ensure_passthrough(
            plc,
            plc_config,
            command_tag=power_supply.controller.config.command_tag,
            logger=logger,
        )
        apply_config(plc_config)
        logger.info("Synced routing and alarm limits from PLC.")
        return plc_config
    except Exception as sync_err:
        logger.warning(f"Could not sync routing from PLC: {sync_err}")
        return None


async def _beat_watchdog(plc: Any) -> None:
    """Stroke the firmware host-alive watchdog, if the client exposes the seam.

    PR #4044 made holding register 560 a heartbeat: the firmware drives its
    outputs safe after 2 s without a *change* to it. Only a scan that actually
    read live data counts as proof of life. Guarded with ``getattr`` so this
    lands cleanly whether or not the client-side seam has merged yet (LOD).
    """
    beat = getattr(plc, "write_heartbeat", None)
    if not callable(beat):
        return
    try:
        result = beat()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # noqa: BLE001 - a missed beat must not abort a scan
        logger.warning("Firmware heartbeat write failed: %s", exc)


async def _read_scan_tags(
    plc: Any, backup: Any, *, simulated: bool
) -> dict[str, float] | None:
    """Read this scan's tags from the primary client, else the wired simulator.

    The backup is consulted only on a simulator driver — and the caller is
    expected not to wire one otherwise. Both conditions are checked (defence in
    depth): fabricated data must never reach a real-hardware scan, whatever a
    future caller passes (issue #4004).
    """
    tags: dict[str, float] | None = None
    if plc.connected:
        tags = await plc.read_tags()
    if tags is None and simulated and backup is not None and not plc.connected:
        tags = await backup.read_tags()
    if tags is not None and not isinstance(tags, dict):
        raise TypeError(f"poll tags must be a dict or None, got {type(tags).__name__}")
    return tags


def _apply_write_seam_interlocks(
    *,
    plc: Any,
    power_supply: Any,
    temperature: Any,
    estop_active: bool,
    data_trusted: bool,
) -> None:
    """Arm or lower the low-level write-seam interlocks for this scan.

    Armed when the E-stop is latched (H4: before any service poll can command an
    output) AND whenever this scan has no trustworthy measurement — without a
    reading, nothing may be turned into an *energizing* command. The seams never
    block the de-energizing direction, and the controllers' own one-way E-stop
    latch is only engaged for a real E-stop, never for a data gap.
    """
    if estop_active:
        _reengage_service_estop(power_supply)
        if temperature is not None:
            _reengage_service_estop(temperature)
    else:
        # Lower/raise only the low-level interlock so it tracks the shared flag.
        # The controllers' own E-stop latch is intentionally NOT cleared here —
        # that requires an explicit operator reset (clear_estop).
        _set_service_estop_flag(power_supply, not data_trusted)
        if temperature is not None:
            _set_service_estop_flag(temperature, not data_trusted)
    set_plc_flag = getattr(plc, "set_estop_active", None)
    if callable(set_plc_flag):
        set_plc_flag(estop_active or not data_trusted)


async def _poll_once(
    *,
    plc: Any,
    latest_tag_values: dict[str, float],
    ws: Any,
    alicats: Any,
    power_supply: Any,
    alarm_engine: Any,
    active_alarm_map: dict[str, dict[str, Any]],
    estop_active: bool,
    backup: Any = None,
    simulated: bool = False,
    temperature: Any = None,
    historian: Any = None,
    quality_tracker: Any = None,
    broadcast: bool = True,
    diagnostics: Mapping[str, Any] | None = None,
    process_events: Callable[
        [Any, dict[str, float], dict[str, dict[str, Any]]],
        list[Any],
    ] = process_alarm_events,
) -> dict[str, Any]:
    """Run one PLC scan, publish it, and queue the historian/alarm rows.

    The scan's data provenance decides what it is allowed to touch (#4004):

    * ``live`` / ``simulated`` — a real measurement (or a deliberately selected
      bench simulation). Drives the control laws, the alarm engine and the
      historian; a live scan also strokes the firmware watchdog.
    * ``held`` — the link is up but this read failed. The last good values are
      broadcast so the HMI does not flicker, but they are NOT presented to the
      control laws, the alarm engine or the historian.
    * ``fault`` — no data at all. Same suppression, plus the write-seam
      interlocks are armed so no output can be energized on stale information.

    Args:
        backup: A simulator client, wired ONLY when the configured driver is
            itself a simulator. None on real hardware — a dropped link must
            surface as a fault, never as invented continuity.
        simulated: Whether the active driver is a simulator.
        historian: Optional :class:`HistorianWriter`-like sink exposing
            ``submit(HistorianRecord)``. The scan never touches SQLite itself.
        quality_tracker: Optional :class:`DataQualityTracker` recording
            provenance transitions as auditable events.
        broadcast: When False the frame is built and returned but not pushed to
            the WebSocket clients (performance-mode decimation, #4008). The scan
            itself always runs.
        diagnostics: Loop-health counters merged into the frame.

    Raises:
        TypeError: If ``latest_tag_values`` / ``active_alarm_map`` are not
            dicts, or the client returns a non-dict tag set.
    """
    if not isinstance(latest_tag_values, dict):
        raise TypeError(
            f"latest_tag_values must be a dict, got {type(latest_tag_values).__name__}"
        )
    if not isinstance(active_alarm_map, dict):
        raise TypeError(
            f"active_alarm_map must be a dict, got {type(active_alarm_map).__name__}"
        )

    plc_connected = bool(plc.connected)
    fresh_tags = await _read_scan_tags(plc, backup, simulated=simulated)
    source = _resolve_source(
        plc_connected=plc_connected,
        tags_read=fresh_tags is not None,
        have_held=bool(latest_tag_values),
        simulated=simulated,
    )
    if fresh_tags is not None:
        latest_tag_values.update(fresh_tags)

    # Values the control path may act on. A held/faulted scan yields None so the
    # services see "no measurement" and fail safe (the thermocouple deglitch
    # trips TC_FAULT, the AO command is clamped by the armed write seam) instead
    # of acting on a stale or fabricated number.
    trusted_tags = fresh_tags if source.is_measurement else None
    # Values the HMI may *display*: held readings are still shown, but the frame
    # says plainly where they came from.
    display_tags = fresh_tags if fresh_tags is not None else dict(latest_tag_values)
    tag_list = [display_tags.get(f"TAG_{i}", 0.0) for i in range(32)]

    _apply_write_seam_interlocks(
        plc=plc,
        power_supply=power_supply,
        temperature=temperature,
        estop_active=estop_active,
        data_trusted=source.is_measurement,
    )

    ps_status = await power_supply.poll(trusted_tags)
    temp_status = (
        await temperature.poll(trusted_tags) if temperature is not None else None
    )
    if estop_active and plc_connected:
        await plc.trigger_estop()
    elif source == DataSource.LIVE:
        await _beat_watchdog(plc)

    payload: dict[str, Any] = {
        "tags": tag_list,
        "tags_dict": display_tags,
        "alicats": alicats.get_devices_data(),
        "active_alarms": active_alarm_map,
        "e_stop_active": estop_active,
        "power_supply": ps_status.model_dump(),
        "data_source": source.value,
        "plc_connected": plc_connected,
        "simulated": bool(simulated),
    }
    if temp_status is not None:
        payload["temperature"] = temp_status.model_dump()
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    if broadcast:
        await ws.broadcast(payload)

    events: list[Any] = []
    if quality_tracker is not None:
        transition = quality_tracker.observe(source.value)
        if transition is not None:
            events.append(transition)
    if trusted_tags is not None:
        # Only a real measurement may move the alarm state machine. Feeding it a
        # held/simulated scan is what silently cleared an active HiHi to Normal
        # when the link flapped (issue #4004).
        events.extend(process_events(alarm_engine, trusted_tags, active_alarm_map))

    if historian is not None:
        historian.submit(
            HistorianRecord(
                tags=trusted_tags,
                events=tuple(events),
                quality=source.value,
            )
        )

    return payload
