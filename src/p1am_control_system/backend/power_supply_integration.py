"""FastAPI and PLC integration for the P1AM power-supply controller.

Wires the pure :class:`PowerSupplyController` to the live PLC and, when a
``session_factory`` is supplied, persists the operator's control *settings*
(config + last setpoint) to the durable config store so the supply comes back
with the last session's configuration after a restart. Mirrors
``temperature_integration.py`` so the two control subsystems share one shape
(LOD: this layer talks only to the controller, the PLC client's public seam, and
the config-store public functions — never their internals).

Safety scope: persistence stores **settings only**. Restoring on boot applies
the config/limits and stashes the last setpoint for HMI pre-fill; it NEVER arms
or energizes the output — the controller stays IDLE until the operator re-enables
permissive and re-commands a setpoint.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import hardware
from auth_config import require_admin_key
from config_store import load_config, load_model, save_model
from fastapi import APIRouter, Depends, HTTPException
from power_supply import (
    PowerSupplyConfig,
    PowerSupplyController,
    PowerSupplyMode,
    PowerSupplyStatus,
)
from power_supply_models import PowerSupplyLastSetpoint
from pydantic import BaseModel

__all__ = [
    "PowerSupplyPermissiveRequest",
    "PowerSupplyService",
    "PowerSupplySetpointRequest",
    "create_power_supply_router",
]

# Durable-store keys for the power-supply operator settings. Kept as module
# constants (DRY) so the service and any future reader agree on the exact names.
_CONFIG_KEY = "power_config"
_SETPOINT_KEY = "power_setpoint"


class SensorFeedbackError(RuntimeError):
    """Raised when a scan's power-supply feedback is absent or unusable.

    Distinguishes "no reading" from "a reading of zero" so the caller can trip
    rather than act on fabricated data (issue #4016).
    """


class PowerSupplyService:
    """Owns the controller and applies its command to the PLC PID pass-through."""

    def __init__(
        self,
        plc_client: Any,
        logger: logging.Logger,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Create the service.

        Args:
            plc_client: PLC client exposing the public ``write_pid_setpoint`` seam.
            logger: Logger for best-effort write/persist diagnostics.
            session_factory: Optional zero-arg callable returning a
                context-managed SQLModel ``Session`` (e.g. ``lambda: Session(engine)``).
                When ``None`` the service skips all persistence — so tests and
                deployments without a DB behave exactly as before.

        Raises:
            TypeError: if ``session_factory`` is provided but is not callable.
        """
        if session_factory is not None and not callable(session_factory):
            raise TypeError(
                "session_factory must be callable or None, got "
                f"{type(session_factory).__name__}"
            )
        self.controller = PowerSupplyController(PowerSupplyConfig())
        self._plc_client = plc_client
        self._logger = logger
        self._session_factory = session_factory
        # Operator's last commanded setpoint, recalled on boot for HMI pre-fill.
        # Purely informational: it never arms or energizes the output.
        self._last_setpoint: PowerSupplyLastSetpoint | None = None
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
        """Feed measured tags into the controller and write the AO command.

        A scan with unusable feedback latches SENSOR_FAULT and drives the
        output to its safe state rather than substituting zeros, which would
        leave the supply energized against interlocks that can no longer fire.
        """
        try:
            current_a, voltage_v, temp_c = self._inputs_from_tags(tags)
        except SensorFeedbackError as err:
            self.controller.signal_sensor_fault(str(err))
            await self._write_pid_setpoint(0, 0.0)
            return self.controller.status()
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

    def status(self) -> PowerSupplyStatus:
        """Return the controller status with the recalled last setpoint attached.

        The controller owns the live snapshot; the service overlays the
        operator's last commanded setpoint (recalled from durable storage on
        boot, or updated on each ``set_*_setpoint`` call) so the HMI can pre-fill.
        Purely informational — never arms or energizes the output.
        """
        snapshot = self.controller.status()
        snapshot.last_setpoint = self._last_setpoint
        return snapshot

    @property
    def last_setpoint(self) -> PowerSupplyLastSetpoint | None:
        """The operator's last commanded setpoint recalled/updated for HMI use."""
        return self._last_setpoint

    def update_config(self, new_config: PowerSupplyConfig) -> PowerSupplyConfig:
        """Apply an operator config change and persist it (best-effort).

        Delegates to the controller (which validates and re-clamps) and then, when
        a ``session_factory`` is set, durably stores the new config under
        ``"power_config"`` so it survives a restart.

        Args:
            new_config: A fully-validated :class:`PowerSupplyConfig`.

        Returns:
            The controller's active config after the update.

        Raises:
            TypeError: if ``new_config`` is not a ``PowerSupplyConfig`` (raised by
                the controller).
        """
        self.controller.update_config(new_config)
        applied = self.controller.config
        self._persist_model(_CONFIG_KEY, applied)
        return applied

    def set_current_setpoint(self, value_a: float) -> float:
        """Apply a current setpoint and persist it (best-effort).

        Delegates to the controller (which validates, clamps, and may reject the
        change depending on state) and then persists the operator's intent under
        ``"power_setpoint"`` so the HMI can recall it after a restart.

        Args:
            value_a: Desired current setpoint in amps.

        Returns:
            The current setpoint now in effect. On a rejected request (IDLE or
            TRIPPED) that is the unchanged existing setpoint, not the request.

        Raises:
            TypeError: if ``value_a`` is not numeric.
            ValueError: if ``value_a`` is NaN or infinite.
        """
        applied = float(self.controller.set_current_setpoint(value_a))
        # Persist only what actually took effect. Recording the raw request
        # meant a setpoint the controller rejected was written to durable
        # storage and pre-filled into the HMI after the next restart, so the
        # operator was shown a value the plant never ran at (issue #4017).
        if applied == float(value_a):
            self._record_setpoint(
                PowerSupplyLastSetpoint(mode=PowerSupplyMode.CURRENT, value_a=applied)
            )
        return applied

    def set_power_setpoint(self, value_w: float) -> float:
        """Apply a power setpoint and persist it (best-effort).

        Delegates to the controller and persists the operator's intent under
        ``"power_setpoint"`` for HMI recall.

        Args:
            value_w: Desired power setpoint in watts.

        Returns:
            The achievable power given clamping.

        Raises:
            TypeError: if ``value_w`` is not numeric.
            ValueError: if ``value_w`` is NaN, infinite, or negative.
        """
        achievable = float(self.controller.set_power_setpoint(value_w))
        self._record_setpoint(
            PowerSupplyLastSetpoint(mode=PowerSupplyMode.POWER, value_w=value_w)
        )
        return achievable

    def set_permissive(self, enabled: bool) -> PowerSupplyStatus:
        """Toggle permissive on the controller and return the service status.

        Permissive is a live operator control (arming intent), not a persisted
        setting — restoring on boot must never re-arm — so it is deliberately not
        written to the durable store.

        Raises:
            TypeError: if ``enabled`` is not exactly a bool.
        """
        self.controller.set_permissive(enabled)
        return self.status()

    def restore_persisted(self, session: Any) -> None:
        """Load and apply persisted operator settings (best-effort, boot-only).

        Applies the stored config to the controller and stashes the last
        setpoint for HMI pre-fill. SAFETY: this NEVER arms or energizes the
        output — it does not call ``set_permissive`` nor push a setpoint into the
        controller, so the controller stays IDLE. Any error is swallowed so a
        corrupt/legacy blob can only fall back to defaults, never block startup.

        Args:
            session: An active SQLModel session bound to the config DB.
        """
        try:
            cfg = load_model(session, _CONFIG_KEY, PowerSupplyConfig)
            if cfg is not None:
                self.controller.update_config(cfg)
        except Exception as exc:  # noqa: BLE001 - never block boot on restore
            self._logger.warning("power config restore skipped: %s", exc)
        try:
            data = load_config(session, _SETPOINT_KEY)
            if data is not None:
                self._last_setpoint = PowerSupplyLastSetpoint(**data)
        except Exception as exc:  # noqa: BLE001 - never block boot on restore
            self._logger.warning("power setpoint restore skipped: %s", exc)

    def _record_setpoint(self, setpoint: PowerSupplyLastSetpoint) -> None:
        """Update the in-memory last setpoint and persist it (best-effort)."""
        self._last_setpoint = setpoint
        self._persist_model(_SETPOINT_KEY, setpoint)

    def _persist_model(self, key: str, model: BaseModel) -> None:
        """Persist ``model`` under ``key`` when a session factory is configured.

        No-op when ``session_factory`` is ``None``. Best-effort: a persistence
        failure is logged, never raised, so an operator command always succeeds
        even if the durable store is momentarily unavailable.
        """
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as session:
                save_model(session, key, model)
        except Exception as exc:  # noqa: BLE001 - persistence must not break control
            self._logger.error("persisting %r failed: %s", key, exc)

    def _inputs_from_tags(
        self,
        tags: dict[str, float] | None,
    ) -> tuple[float, float, float]:
        cfg = self.controller.config
        missing = self._missing_feedback_tags(tags)
        if missing:
            # Fabricating 0.0 here made both HH trips permanently un-trippable
            # and reported a confident, cold-looking supply while the output
            # stayed energized. Missing data is a fault, not a measurement
            # (issue #4016). Raising keeps the safe path in one place -- poll()
            # catches it and latches SENSOR_FAULT.
            raise SensorFeedbackError(
                "power supply feedback unavailable: " + ", ".join(missing)
            )

        # An empty `missing` proves every required tag is present and finite;
        # restate it so the type checker sees the narrowing too.
        assert tags is not None
        current_pct = float(tags[cfg.current_feedback_tag])
        voltage_pct = float(tags[cfg.voltage_feedback_tag])
        temp_pct = float(tags[cfg.temp_tag])
        current_a = current_pct * cfg.current_full_scale_a / 100.0
        voltage_v = voltage_pct * cfg.voltage_full_scale_v / 100.0
        # The firmware publishes thermocouples as PERCENT of full scale, not
        # degrees C. Passing the raw tag through as though it were already degC
        # made the HH_TEMP trip -- a degC threshold -- unreachable by any
        # physically possible reading (issue #4003). Converted through the one
        # shared helper so this cannot drift from the temperature service.
        temp_c = hardware.percent_to_celsius(temp_pct, cfg.temp_full_scale_c)
        return current_a, voltage_v, temp_c

    def _missing_feedback_tags(self, tags: dict[str, float] | None) -> list[str]:
        """Names of feedback tags that are absent or unusable this scan.

        A tag that is present but non-finite is just as unusable as one that is
        absent -- a NaN compares False against every threshold, so it would
        silently defeat the trips rather than raise them.
        """
        cfg = self.controller.config
        required = (
            cfg.current_feedback_tag,
            cfg.voltage_feedback_tag,
            cfg.temp_tag,
        )
        if not tags:
            return list(required)
        missing: list[str] = []
        for name in required:
            if name not in tags:
                missing.append(f"{name} (absent)")
                continue
            value = tags[name]
            if not isinstance(value, int | float) or isinstance(value, bool):
                missing.append(f"{name} (non-numeric)")
            elif not math.isfinite(float(value)):
                missing.append(f"{name} (non-finite)")
        return missing

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
        return service.update_config(new_config)

    @router.get("/status", response_model=PowerSupplyStatus)
    async def get_power_supply_status() -> PowerSupplyStatus:
        return service.status()

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
                applied = service.set_current_setpoint(req.value_a)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"mode": "current", "applied_a": applied}

        if req.value_w is None:
            raise HTTPException(
                status_code=400,
                detail="value_w is required when mode='power'",
            )
        try:
            achievable = service.set_power_setpoint(req.value_w)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"mode": "power", "achievable_w": achievable}

    @router.post("/permissive", dependencies=[Depends(require_admin_key)])
    async def set_power_supply_permissive(
        req: PowerSupplyPermissiveRequest,
    ) -> PowerSupplyStatus:
        return service.set_permissive(req.enabled)

    @router.post("/acknowledge_trip", dependencies=[Depends(require_admin_key)])
    async def acknowledge_power_supply_trip() -> PowerSupplyStatus:
        controller.acknowledge_trip()
        return controller.status()

    return router
