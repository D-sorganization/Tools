"""Single-attempt runtime helpers for the P1AM backend loops.

The FastAPI shell owns lifecycle and cadence. This module owns one PLC
connection attempt and one PLC polling scan so safety and persistence contracts
can be tested without sleeping in infinite loops.

It also re-exports the two seams that keep a scan honest and cheap:

* :class:`DataQualityTracker` — turns a change of tag provenance into an
  auditable event, so an outage appears in the log instead of as a silent gap.
* :class:`HistorianWriter` — moves the SQLite write off the event loop into a
  batched, retrying worker so a ``VACUUM`` lock can no longer lose an alarm
  transition or stall the control loop (issue #4023).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

from alarm_processing import process_alarm_events
from data_quality import DataQualityTracker
from historian import (
    DEFAULT_HISTORIAN_BATCH_MAX,
    DEFAULT_HISTORIAN_QUEUE_SIZE,
    HistorianRecord,
    HistorianWriter,
    ThrottledHistorianSink,
    _WriterCounters,
)
from models import DataSource, RoutingConfig
from power_supply_passthrough import ensure_power_supply_passthrough

logger = logging.getLogger("dcs_backend.poll_runtime")

__all__ = [
    "DEFAULT_HISTORIAN_BATCH_MAX",
    "DEFAULT_HISTORIAN_QUEUE_SIZE",
    "DataQualityTracker",
    "HistorianRecord",
    "HistorianWriter",
    "POLL_FAILURE_ESCALATION_THRESHOLD",
    "POLL_FAILURE_MAX_BACKOFF_S",
    "ThrottledHistorianSink",
    "_WriterCounters",
    "_apply_write_seam_interlocks",
    "_beat_watchdog",
    "connect_once",
    "poll_once",
    "_read_scan_tags",
    "_reengage_service_estop",
    "_resolve_source",
    "_set_service_estop_flag",
    "log_poll_failure",
    "loop_diagnostics",
]


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


async def connect_once(
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


async def poll_once(
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
        plc: PLC client interface.
        latest_tag_values: Cache of most recent tag readings.
        ws: WebSocket connection manager.
        alicats: Alicat MFC manager.
        power_supply: Power supply service controller.
        alarm_engine: SCADA alarm engine.
        active_alarm_map: Mapping of currently active alarms.
        estop_active: Whether emergency stop is latched.
        backup: A simulator client, wired ONLY when the configured driver is
            itself a simulator. None on real hardware — a dropped link must
            surface as a fault, never as invented continuity.
        simulated: Whether the active driver is a simulator.
        temperature: Optional temperature service controller.
        historian: Optional :class:`HistorianWriter`-like sink exposing
            ``submit(HistorianRecord)``. The scan never touches SQLite itself.
        quality_tracker: Optional :class:`DataQualityTracker` recording
            provenance transitions as auditable events.
        broadcast: When False the frame is built and returned but not pushed to
            the WebSocket clients (performance-mode decimation, #4008). The scan
            itself always runs.
        diagnostics: Loop-health counters merged into the frame.
        process_events: Event processor callback for alarm transitions.

    Returns:
        The published scan frame dictionary.

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
