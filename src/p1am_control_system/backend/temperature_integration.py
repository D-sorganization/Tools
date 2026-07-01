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
from collections.abc import Callable
from typing import Any, cast

import hardware
from auth_config import require_admin_key
from config_store import load_config, load_model, save_config, save_model
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from temperature_controller import TemperatureController
from temperature_models import (
    TcType,
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
)

__all__ = [
    "TcTypeRequest",
    "TemperaturePermissiveRequest",
    "TemperatureService",
    "TemperatureSetpointRequest",
    "create_temperature_router",
]

# config_store keys for the persisted temperature settings.
_CONFIG_KEY = "temperature_config"
_SETPOINT_KEY = "temperature_setpoint"


class TemperatureService:
    """Owns the controller and applies its relay command to the PLC coil."""

    def __init__(
        self,
        plc_client: Any,
        logger: logging.Logger,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.controller = TemperatureController(TemperatureConfig())
        self._plc_client = plc_client
        self._logger = logger
        # Optional zero-arg callable returning a context-managed Session. When
        # None (as in the existing tests) the service skips persistence entirely,
        # so behaviour is unchanged. When set, operator changes are written to
        # the durable config store and recalled at boot via restore_persisted.
        self._session_factory = session_factory
        # Last operator setpoint recalled from persisted settings (deg C). Surfaced
        # in status() so the HMI can pre-fill the target after a restart. Recalling
        # it NEVER arms/energizes the heater — the controller stays IDLE.
        self._last_setpoint_c: float | None = None
        # Defense-in-depth E-stop interlock, independent of the controller's own
        # latch: when set, ``_write_relay`` forces the heater relay OFF and
        # surfaces a failed de-energize as a comms failure. Set by the poll loop.
        self._estop_active = False
        # Latched True when a commanded de-energize (relay OFF) write failed even
        # after a retry, so the caller can escalate a comms alarm.
        self._deenergize_comms_failed = False

    @property
    def deenergize_comms_failed(self) -> bool:
        """True when the last commanded relay-OFF write failed after retry."""
        return self._deenergize_comms_failed

    def set_estop_active(self, active: bool) -> None:
        """Set the service-level E-stop write-seam interlock.

        Raises:
            TypeError: if ``active`` is not a bool.
        """
        if not isinstance(active, bool):
            raise TypeError(f"active must be a bool, got {type(active).__name__}")
        self._estop_active = active

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
        return self.status()

    def status(self) -> TemperatureStatus:
        """Controller status augmented with the recalled last setpoint.

        Returns the controller snapshot with ``last_setpoint_c`` set to the value
        recalled from persisted settings (``None`` when nothing was recalled), so
        the HMI can pre-fill the target field without the service leaking any
        controller internals (LOD).
        """
        status: TemperatureStatus = self.controller.status()
        status.last_setpoint_c = self._last_setpoint_c
        return status

    def update_config(self, new_config: TemperatureConfig) -> TemperatureConfig:
        """Apply a new controller config and persist it (best-effort).

        Delegates to the controller, then persists the *resulting* controller
        config (which may have re-clamped the setpoint) under ``_CONFIG_KEY`` when
        a session factory is configured.

        Raises:
            TypeError: if new_config is not a TemperatureConfig (from controller).
        """
        self.controller.update_config(new_config)
        self._persist_config()
        config: TemperatureConfig = self.controller.config
        return config

    def set_setpoint(self, value_c: float) -> float:
        """Apply an operator setpoint and persist it when it actually took effect.

        Returns the clamped value the controller applied. The setpoint is persisted
        (and remembered as the recalled ``last_setpoint_c``) only when the controller
        was armed/running so the change actually applied — a stopped no-op is never
        stored, so a later restore does not resurrect a value the operator never ran.

        Raises:
            TypeError: if value_c is not numeric (from controller).
            ValueError: if value_c is not finite (from controller).
        """
        applied: float = self.controller.set_setpoint_c(value_c)
        if self.controller.state in (
            TemperatureState.ARMED,
            TemperatureState.RUNNING,
        ):
            self._last_setpoint_c = applied
            self._persist_setpoint(applied)
        return applied

    def set_active_tc_type(self, tc_type: TcType) -> None:
        """Switch the controlling thermocouple and persist the config.

        Raises:
            TypeError: if tc_type is not a TcType (from controller).
            ValueError: if the resulting config is invalid (from controller).
        """
        self.controller.set_active_tc_type(tc_type)
        self._persist_config()

    def restore_persisted(self, session: Any) -> None:
        """Recall persisted settings into the controller (SAFE: stays IDLE).

        Applies the persisted config (limits/labels) and remembers the last
        setpoint for the HMI to pre-fill. It NEVER arms, energizes, or changes the
        controller's state machine — the heater always comes back IDLE and the
        operator must press Start to resume. Best-effort: a corrupt/incompatible
        blob is logged and skipped so it can never block boot.

        Args:
            session: An active SQLModel session bound to the config DB.
        """
        try:
            cfg = load_model(session, _CONFIG_KEY, TemperatureConfig)
            if cfg is not None:
                # update_config only re-tunes limits/setpoint clamp; it does not
                # arm the controller, so the state machine stays IDLE.
                self.controller.update_config(cfg)
            data = load_config(session, _SETPOINT_KEY)
            if data is not None and "value_c" in data:
                self._last_setpoint_c = float(cast(float, data["value_c"]))
        except Exception as exc:  # noqa: BLE001 - a bad blob must not block boot
            self._logger.warning("temperature restore skipped: %s", exc)

    def _persist_config(self) -> None:
        """Persist the current controller config (best-effort, no-op if disabled)."""
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as s:
                save_model(s, _CONFIG_KEY, self.controller.config)
        except Exception as exc:  # noqa: BLE001 - persistence must not break control
            self._logger.warning("temperature config persist failed: %s", exc)

    def _persist_setpoint(self, value_c: float) -> None:
        """Persist the last applied setpoint (best-effort, no-op if disabled)."""
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as s:
                save_config(s, _SETPOINT_KEY, {"value_c": value_c})
        except Exception as exc:  # noqa: BLE001 - persistence must not break control
            self._logger.warning("temperature setpoint persist failed: %s", exc)

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
        """Command the heater relay via the client's public coil seam.

        Defense-in-depth interlock: when the service E-stop flag is set, an
        energizing command (``on`` True) is forced OFF here regardless of what
        the controller returned, independent of the controller's own latch. The
        de-energizing direction is never blocked. A failed de-energize write is
        retried once; if it still fails ``_deenergize_comms_failed`` is latched
        so the caller can escalate a comms alarm. A successful write clears it.
        """
        if self._estop_active and on:
            self._logger.warning("heater relay energize forced OFF — E-stop interlock")
            on = False
        deenergize = not on
        attempts = 2 if deenergize else 1
        for attempt in range(attempts):
            try:
                ok = bool(
                    await self._plc_client.write_coil(hardware.HEATER_RELAY_COIL, on)
                )
            except Exception as exc:  # best-effort: never abort the scan loop
                self._logger.error("heater relay write failed: %s", exc)
                ok = False
            if ok:
                if deenergize:
                    self._deenergize_comms_failed = False
                return True
            if deenergize and attempt + 1 < attempts:
                self._logger.error("heater relay de-energize FAILED — retrying")
        if deenergize:
            self._deenergize_comms_failed = True
            self._logger.error(
                "heater relay de-energize FAILED after retry — comms alarm"
            )
        return False


class TemperatureSetpointRequest(BaseModel):
    """Operator-facing temperature setpoint command (deg C)."""

    value_c: float


class TemperaturePermissiveRequest(BaseModel):
    enabled: bool


class TcTypeRequest(BaseModel):
    """Operator selection of which thermocouple (K or R) drives the heater."""

    active_tc_type: TcType


def create_temperature_router(service: TemperatureService) -> APIRouter:
    router = APIRouter(prefix="/api/temperature", tags=["temperature"])
    controller = service.controller

    @router.get("/config", response_model=TemperatureConfig)
    async def get_temperature_config() -> TemperatureConfig:
        config: TemperatureConfig = controller.config
        return config

    @router.put(
        "/config",
        response_model=TemperatureConfig,
        dependencies=[Depends(require_admin_key)],
    )
    async def update_temperature_config(
        new_config: TemperatureConfig,
    ) -> TemperatureConfig:
        return service.update_config(new_config)

    @router.get("/status", response_model=TemperatureStatus)
    async def get_temperature_status() -> TemperatureStatus:
        return service.status()

    @router.post("/setpoint", dependencies=[Depends(require_admin_key)])
    async def apply_temperature_setpoint(
        req: TemperatureSetpointRequest,
    ) -> dict[str, Any]:
        try:
            applied = service.set_setpoint(req.value_c)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"applied_c": applied}

    @router.post("/permissive", dependencies=[Depends(require_admin_key)])
    async def set_temperature_permissive(
        req: TemperaturePermissiveRequest,
    ) -> TemperatureStatus:
        controller.set_permissive(req.enabled)
        status: TemperatureStatus = controller.status()
        return status

    @router.post("/tc_type", dependencies=[Depends(require_admin_key)])
    async def set_active_tc_type(req: TcTypeRequest) -> TemperatureStatus:
        """Switch the heater's controlling thermocouple between type K and R."""
        try:
            service.set_active_tc_type(req.active_tc_type)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return service.status()

    @router.post("/acknowledge_trip", dependencies=[Depends(require_admin_key)])
    async def acknowledge_temperature_trip() -> TemperatureStatus:
        controller.acknowledge_trip()
        status: TemperatureStatus = controller.status()
        return status

    return router
