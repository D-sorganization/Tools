"""FastAPI and PLC integration for the P1AM power-supply controller."""

from __future__ import annotations

import logging
from typing import Any

from auth_config import require_admin_key
from fastapi import APIRouter, Depends, HTTPException
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
        # Defense-in-depth E-stop interlock, independent of the controller's own
        # latch: when set, ``_write_pid_setpoint`` forces the AO command to 0 and
        # surfaces a failed de-energize as a comms failure. Set by the poll loop.
        self._estop_active = False
        # Latched True when a commanded de-energize (0) write failed even after a
        # retry, so the caller can escalate a comms alarm.
        self._deenergize_comms_failed = False

    @property
    def deenergize_comms_failed(self) -> bool:
        """True when the last commanded AO-zero write failed after retry."""
        return self._deenergize_comms_failed

    def set_estop_active(self, active: bool) -> None:
        """Set the service-level E-stop write-seam interlock.

        Raises:
            TypeError: if ``active`` is not a bool.
        """
        if not isinstance(active, bool):
            raise TypeError(f"active must be a bool, got {type(active).__name__}")
        self._estop_active = active

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

    def engage_estop(self) -> None:
        """Latch the controller's E-stop (software half of the kill switch)."""
        self.controller.engage_estop()

    def clear_estop(self) -> None:
        """Release the controller's E-stop latch (operator must re-arm)."""
        self.controller.clear_estop()

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
        """Command a PID setpoint via the client's public seam.

        Delegates to ``plc_client.write_pid_setpoint`` — no longer reaches into
        the client's private connection/lock or hand-rolls the register encoding
        (which had diverged from modbus_client.float_to_registers). Works for the
        real Modbus client and the simulator alike.

        Defense-in-depth interlock: when the service E-stop flag is set, an
        energizing (non-zero) command is forced to 0 here regardless of what the
        controller returned, independent of the controller's own latch. The
        de-energizing direction (0) is never blocked. A failed de-energize write
        is retried once; if it still fails ``_deenergize_comms_failed`` is
        latched so the caller can escalate a comms alarm. A successful write
        clears it.
        """
        if self._estop_active and value != 0.0:
            self._logger.warning("PID setpoint energize forced to 0 — E-stop interlock")
            value = 0.0
        deenergize = value == 0.0
        attempts = 2 if deenergize else 1
        for attempt in range(attempts):
            try:
                ok = bool(await self._plc_client.write_pid_setpoint(pid_index, value))
            except Exception as exc:  # best-effort: never abort the scan loop
                self._logger.error("PID setpoint write failed: %s", exc)
                ok = False
            if ok:
                if deenergize:
                    self._deenergize_comms_failed = False
                return True
            if deenergize and attempt + 1 < attempts:
                self._logger.error("PID setpoint de-energize FAILED — retrying")
        if deenergize:
            self._deenergize_comms_failed = True
            self._logger.error(
                "PID setpoint de-energize FAILED after retry — comms alarm"
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

    @router.put(
        "/config",
        response_model=PowerSupplyConfig,
        dependencies=[Depends(require_admin_key)],
    )
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
