"""FastAPI and PLC integration for the P1AM temperature controller.

Wires the pure :class:`TemperatureController` to the live PLC: it reads the
thermocouple tag on every scan, ticks the on/off control law, and drives the
heater relay through the client's public ``write_coil`` seam (24 V DO -> relay
-> 110 V resistive heater). Mirrors ``power_supply_integration.py`` so the two
control subsystems share one shape (LOD: this layer talks only to the
controller and the PLC client's public seam, never their internals).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import hardware
from auth_config import require_admin_key
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from temperature_controller import TemperatureController
from temperature_models import TemperatureConfig, TemperatureStatus


class TemperatureService:
    """Owns the controller and applies its relay command to the PLC coil."""

    def __init__(self, plc_client: Any, logger: logging.Logger) -> None:
        self.controller = TemperatureController(TemperatureConfig())
        self._plc_client = plc_client
        self._logger = logger

    async def poll(self, tags: dict[str, float] | None) -> TemperatureStatus:
        """Feed the measured thermocouple into the controller, drive the relay.

        Returns the controller status either way; the relay write is best-effort
        so a momentary PLC hiccup never aborts the scan loop. Note the heater is
        only ever *commanded on* — when the write fails the relay is whatever the
        PLC last latched, and the controller's HH cutoff / E-stop still force the
        commanded value to False, so a failed write can never *energize* it.
        """
        temp_c = self._temp_from_tags(tags)
        # Pass a monotonic clock so the controller can enforce the
        # anti-short-cycle min on/off dwell across scans.
        relay_on = self.controller.tick(measured_temp_c=temp_c, now=time.monotonic())
        await self._write_relay(relay_on)
        return self.controller.status()

    def engage_estop(self) -> None:
        """Latch the controller's E-stop (forces the heater relay off)."""
        self.controller.engage_estop()

    def clear_estop(self) -> None:
        """Release the controller's E-stop latch (operator must re-arm)."""
        self.controller.clear_estop()

    def _temp_from_tags(self, tags: dict[str, float] | None) -> float:
        """Scale the thermocouple tag (percent of full scale) into deg C.

        The firmware publishes every analog channel as 0-100% of its range, so
        the controlled temperature is ``tag_percent * temp_full_scale_c / 100``
        (e.g. a 1400 deg C type-K range -> 50% reads as 700 deg C).
        """
        if not tags:
            return 0.0
        cfg = self.controller.config
        temp_pct = float(tags.get(cfg.temp_tag, 0.0))
        return float(temp_pct * cfg.temp_full_scale_c / 100.0)

    async def _write_relay(self, on: bool) -> bool:
        """Command the heater relay via the client's public coil seam."""
        try:
            return bool(
                await self._plc_client.write_coil(hardware.HEATER_RELAY_COIL, on)
            )
        except Exception as exc:  # best-effort: never abort the scan loop
            self._logger.error("heater relay write failed: %s", exc)
            return False


class TemperatureSetpointRequest(BaseModel):
    """Operator-facing temperature setpoint command (deg C)."""

    value_c: float


class TemperaturePermissiveRequest(BaseModel):
    enabled: bool


def create_temperature_router(service: TemperatureService) -> APIRouter:
    router = APIRouter(prefix="/api/temperature", tags=["temperature"])
    controller = service.controller

    @router.get("/config", response_model=TemperatureConfig)
    async def get_temperature_config() -> TemperatureConfig:
        return controller.config

    @router.put(
        "/config",
        response_model=TemperatureConfig,
        dependencies=[Depends(require_admin_key)],
    )
    async def update_temperature_config(
        new_config: TemperatureConfig,
    ) -> TemperatureConfig:
        controller.update_config(new_config)
        return controller.config

    @router.get("/status", response_model=TemperatureStatus)
    async def get_temperature_status() -> TemperatureStatus:
        return controller.status()

    @router.post("/setpoint", dependencies=[Depends(require_admin_key)])
    async def apply_temperature_setpoint(
        req: TemperatureSetpointRequest,
    ) -> dict[str, Any]:
        try:
            applied = controller.set_setpoint_c(req.value_c)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"applied_c": applied}

    @router.post("/permissive", dependencies=[Depends(require_admin_key)])
    async def set_temperature_permissive(
        req: TemperaturePermissiveRequest,
    ) -> TemperatureStatus:
        controller.set_permissive(req.enabled)
        return controller.status()

    @router.post("/acknowledge_trip", dependencies=[Depends(require_admin_key)])
    async def acknowledge_temperature_trip() -> TemperatureStatus:
        controller.acknowledge_trip()
        return controller.status()

    return router
