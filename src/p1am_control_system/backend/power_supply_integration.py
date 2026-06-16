"""FastAPI and PLC integration for the P1AM power-supply controller."""

from __future__ import annotations

import logging
import struct
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from auth_config import require_admin_key
from power_supply import (
    PowerSupplyConfig,
    PowerSupplyController,
    PowerSupplyMode,
    PowerSupplyStatus,
)
from pydantic import BaseModel


class PowerSupplyService:
    """Owns the controller and applies its command to the PLC PID pass-through."""

    def __init__(self, plc_client: Any, logger: logging.Logger) -> None:
        self.controller = PowerSupplyController(PowerSupplyConfig())
        self._plc_client = plc_client
        self._logger = logger

    async def poll(self, tags: dict[str, float] | None) -> PowerSupplyStatus:
        """Feed measured tags into the controller and write the AO command."""
        current_a, voltage_v, temp_c = self._inputs_from_tags(tags)
        command_percent = self.controller.tick(
            measured_current_a=current_a,
            measured_voltage_v=voltage_v,
            measured_temp_c=temp_c,
        )
        await self._write_pid_setpoint(0, command_percent)
        return self.controller.status()

    def _inputs_from_tags(
        self,
        tags: dict[str, float] | None,
    ) -> tuple[float, float, float]:
        if not tags:
            return 0.0, 0.0, 0.0

        cfg = self.controller.config
        current_pct = float(tags.get(cfg.current_feedback_tag, 0.0))
        voltage_pct = float(tags.get(cfg.voltage_feedback_tag, 0.0))
        temp_c = float(tags.get(cfg.temp_tag, 0.0))
        current_a = current_pct * cfg.current_full_scale_a / 100.0
        voltage_v = voltage_pct * cfg.voltage_full_scale_v / 100.0
        return current_a, voltage_v, temp_c

    async def _write_pid_setpoint(self, pid_index: int, value: float) -> bool:
        """Write a PID setpoint to the PLC register pair used for AO pass-through."""
        if not self._plc_client.connected:
            return False
        if pid_index < 0 or pid_index >= 4:
            self._logger.warning("PID index %d out of range", pid_index)
            return False

        lo, hi = struct.unpack("<HH", struct.pack("<f", float(value)))
        base = 200 + pid_index * 10
        try:
            async with self._plc_client.lock:
                resp = await self._plc_client._get_client().write_registers(
                    address=base + 2, values=[lo, hi]
                )
            if resp.isError():
                self._logger.error(
                    "write_pid_setpoint(%d, %f) failed: %s", pid_index, value, resp
                )
                return False
            return True
        except Exception as exc:
            self._logger.error(
                "write_pid_setpoint(%d, %f) exception: %s", pid_index, value, exc
            )
            return False


class PowerSupplySetpointRequest(BaseModel):
    """Operator-facing setpoint command."""

    mode: PowerSupplyMode
    value_a: float | None = None
    value_w: float | None = None


class PowerSupplyPermissiveRequest(BaseModel):
    enabled: bool


def create_power_supply_router(service: PowerSupplyService) -> APIRouter:
    router = APIRouter(prefix="/api/power_supply", tags=["power_supply"])
    controller = service.controller

    @router.get("/config", response_model=PowerSupplyConfig)
    async def get_power_supply_config() -> PowerSupplyConfig:
        return controller.config

    @router.put("/config", response_model=PowerSupplyConfig, dependencies=[Depends(require_admin_key)])
    async def update_power_supply_config(
        new_config: PowerSupplyConfig,
    ) -> PowerSupplyConfig:
        controller.update_config(new_config)
        return controller.config

    @router.get("/status", response_model=PowerSupplyStatus)
    async def get_power_supply_status() -> PowerSupplyStatus:
        return controller.status()

    @router.post("/setpoint", dependencies=[Depends(require_admin_key)])
    async def apply_power_supply_setpoint(
        req: PowerSupplySetpointRequest,
    ) -> dict[str, Any]:
        if req.mode == PowerSupplyMode.CURRENT:
            if req.value_a is None:
                raise HTTPException(
                    status_code=400,
                    detail="value_a is required when mode='current'",
                )
            try:
                applied = controller.set_current_setpoint(req.value_a)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"mode": "current", "applied_a": applied}

        if req.value_w is None:
            raise HTTPException(
                status_code=400,
                detail="value_w is required when mode='power'",
            )
        try:
            achievable = controller.set_power_setpoint(req.value_w)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"mode": "power", "achievable_w": achievable}

    @router.post("/permissive", dependencies=[Depends(require_admin_key)])
    async def set_power_supply_permissive(
        req: PowerSupplyPermissiveRequest,
    ) -> PowerSupplyStatus:
        controller.set_permissive(req.enabled)
        return controller.status()

    @router.post("/acknowledge_trip", dependencies=[Depends(require_admin_key)])
    async def acknowledge_power_supply_trip() -> PowerSupplyStatus:
        controller.acknowledge_trip()
        return controller.status()

    return router
