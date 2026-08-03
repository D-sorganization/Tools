"""Single-attempt runtime helpers for the P1AM backend loops.

The FastAPI shell owns lifecycle and cadence. This module owns one PLC
connection attempt and one PLC polling scan so safety and persistence contracts
can be tested without sleeping in infinite loops.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import historian
from alarm_processing import process_alarm_events
from models import RoutingConfig
from power_supply_passthrough import ensure_power_supply_passthrough
from signal_quality import SignalFrame, SignalFrameFactory, SignalQuality
from sqlmodel import Session

logger = logging.getLogger("dcs_backend.poll_runtime")
_default_signal_frames = SignalFrameFactory()


def _health_payload(frame: SignalFrame | None) -> dict[str, object]:
    if frame is None:
        return {
            "quality": SignalQuality.BAD.value,
            "diagnostic_reason": "no_data",
            "sequence": None,
            "server_timestamp": None,
            "source": "unavailable",
        }
    sample = next(iter(frame.samples.values()))
    return {
        "quality": sample.quality.value,
        "diagnostic_reason": sample.diagnostic_reason,
        "sequence": frame.sequence,
        "server_timestamp": frame.server_timestamp.isoformat(),
        "source": sample.source,
    }


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


async def _poll_once(
    *,
    plc: Any,
    backup: Any,
    latest_tag_values: dict[str, float],
    ws: Any,
    alicats: Any,
    power_supply: Any,
    temperature: Any = None,
    alarm_engine: Any,
    active_alarm_map: dict[str, dict[str, Any]],
    session_factory: Callable[[], Iterator[Session]],
    estop_active: bool,
    log_scan: Callable[[Session, dict[str, float]], int] = historian.log_scan,
    process_events: Callable[
        [Any, dict[str, float], dict[str, dict[str, Any]]],
        list[Any],
    ] = process_alarm_events,
    signal_frames: SignalFrameFactory | None = None,
) -> dict[str, Any]:
    """Run one PLC scan, broadcast it, and persist historian/alarm rows."""
    if not isinstance(latest_tag_values, dict):
        raise TypeError(
            f"latest_tag_values must be a dict, got {type(latest_tag_values).__name__}"
        )
    if not isinstance(active_alarm_map, dict):
        raise TypeError(
            f"active_alarm_map must be a dict, got {type(active_alarm_map).__name__}"
        )

    frame_factory = signal_frames or _default_signal_frames
    frame: SignalFrame | None = None
    tags = None
    if plc.connected:
        tags = await plc.read_tags()
        if tags is None:
            # Connected but the read hiccuped (common when the Pi is under CPU
            # load and the Modbus read trips its timeout). HOLD the last good
            # values instead of substituting the offline simulator's fabricated
            # readings — otherwise a momentary comms miss shows as a spurious
            # drop to ~0 and feeds the control law a false "cold", which would
            # command the heater ON (a runaway contributor). The next scan
            # retries; a *prolonged* loss is surfaced by the degraded-poll path
            # and by the connection dropping (which routes to the sim below).
            if latest_tag_values:
                tags = dict(latest_tag_values)
                frame = frame_factory.stale(
                    tags,
                    source="plc.driver",
                    reason="read_timeout",
                )
        else:
            frame = frame_factory.good(tags, source="plc.driver")
    if tags is None and not plc.connected:
        # No live PLC (offline / dev, or the connection has dropped) — the
        # simulator drives the plant so the HMI still animates. On real hardware
        # the background connect loop is reconnecting in parallel.
        tags = await backup.read_tags()
        if tags is not None:
            frame = frame_factory.simulated(tags, source="synthetic.simulator")
    if tags is not None and not isinstance(tags, dict):
        raise TypeError(f"poll tags must be a dict or None, got {type(tags).__name__}")

    if tags is not None:
        latest_tag_values.update(tags)

    tag_list = (
        [tags.get(f"TAG_{i}", 0.0) for i in range(32)] if tags is not None else []
    )
    # H4: when E-stopped, re-latch the controllers and arm the write-seam
    # interlocks BEFORE the service polls (which can command outputs), so a scan
    # can never re-energize an output between a reconnect and the hardware
    # re-assert. Idempotent + best-effort; the not-estopped path is unchanged.
    if estop_active:
        _reengage_service_estop(power_supply)
        if temperature is not None:
            _reengage_service_estop(temperature)
        set_plc_flag = getattr(plc, "set_estop_active", None)
        if callable(set_plc_flag):
            set_plc_flag(True)
    else:
        # Lower only the low-level write-seam interlock so it tracks the shared
        # flag. The controllers' own E-stop latch is intentionally NOT cleared
        # here — that requires an explicit operator reset (clear_estop).
        _set_service_estop_flag(power_supply, False)
        if temperature is not None:
            _set_service_estop_flag(temperature, False)
        set_plc_flag = getattr(plc, "set_estop_active", None)
        if callable(set_plc_flag):
            set_plc_flag(False)
    ps_status = await power_supply.poll(tags)
    temp_status = await temperature.poll(tags) if temperature is not None else None
    if estop_active and plc.connected:
        await plc.trigger_estop()

    payload = {
        "tags": tag_list,
        "tags_dict": tags if tags is not None else {},
        "tag_samples": frame.to_payload() if frame is not None else {},
        "comms_health": _health_payload(frame),
        "alicats": alicats.get_devices_data(),
        "active_alarms": active_alarm_map,
        "e_stop_active": estop_active,
        "power_supply": ps_status.model_dump(),
    }
    if temp_status is not None:
        payload["temperature"] = temp_status.model_dump()
    await ws.broadcast(payload)

    if tags is not None:
        db_session = None
        try:
            db_session = next(session_factory())
            log_scan(db_session, tags, signal_frame=frame)
            if frame is not None and frame.alarm_eligible:
                for event_log in process_events(alarm_engine, tags, active_alarm_map):
                    db_session.add(event_log)
            db_session.commit()
        except Exception as db_err:
            if db_session:
                db_session.rollback()
            logger.error(f"Error logging tags/alarms: {db_err}")
        finally:
            if db_session:
                db_session.close()

    return payload
