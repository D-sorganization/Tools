"""Power-supply PID pass-through repair helpers."""

from __future__ import annotations

import logging
from typing import Any

from defaults import ensure_pid_passthrough
from models import RoutingConfig


async def ensure_power_supply_passthrough(
    client: Any,
    plc_config: RoutingConfig,
    power_supply_service: Any,
    logger: logging.Logger,
    pid_index: int = 0,
) -> RoutingConfig:
    """Verify and auto-repair the power-supply PID pass-through on connect."""
    command_tag: str = power_supply_service.controller.config.command_tag
    try:
        repaired_config, needs_repair = ensure_pid_passthrough(
            plc_config, pid_index, command_tag
        )
    except ValueError as exc:
        logger.warning("Power-supply pass-through check skipped: %s", exc)
        return plc_config

    if not needs_repair:
        logger.info(
            "Power-supply PID%d already a pass-through to %s.", pid_index, command_tag
        )
        return plc_config

    logger.warning(
        "Power-supply PID%d is NOT routed to AO %s (output will not respond) - "
        "auto-repairing to a pass-through.",
        pid_index,
        command_tag,
    )
    try:
        wrote = await client.write_routing(repaired_config)
        if wrote:
            await client.save_to_flash()
            logger.warning(
                "Repaired power-supply PID%d pass-through and saved to NVRAM.",
                pid_index,
            )
            return repaired_config
        logger.error(
            "Failed to write power-supply pass-through repair - AO %s remains "
            "misrouted; output will not respond until corrected.",
            command_tag,
        )
    except Exception as exc:
        logger.error("Error repairing power-supply pass-through: %s", exc)
    return plc_config
