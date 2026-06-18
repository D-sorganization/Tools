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
from sqlmodel import Session

logger = logging.getLogger("dcs_backend.poll_runtime")


async def _connect_once(
    *,
    plc: Any,
    power_supply: Any,
    apply_config: Callable[[RoutingConfig], None],
    estop_active: bool,
    ensure_passthrough: Callable[..., Any] = ensure_power_supply_passthrough,
) -> RoutingConfig | None:
    """Attempt one background PLC connection and routing sync."""
    if plc.connected:
        return None

    connected = await plc.connect()
    if not connected:
        return None

    logger.info("Connected to PLC successfully in background.")
    if estop_active:
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

    tags = None
    if plc.connected:
        tags = await plc.read_tags()
    if tags is None:
        tags = await backup.read_tags()
    if tags is not None and not isinstance(tags, dict):
        raise TypeError(f"poll tags must be a dict or None, got {type(tags).__name__}")

    if tags is not None:
        latest_tag_values.update(tags)

    tag_list = (
        [tags.get(f"TAG_{i}", 0.0) for i in range(32)] if tags is not None else []
    )
    ps_status = await power_supply.poll(tags)
    temp_status = await temperature.poll(tags) if temperature is not None else None
    if estop_active and plc.connected:
        await plc.trigger_estop()

    payload = {
        "tags": tag_list,
        "tags_dict": tags if tags is not None else {},
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
            log_scan(db_session, tags)
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
