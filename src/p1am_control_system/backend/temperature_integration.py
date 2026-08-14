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
import math
import time
from collections.abc import Callable
from typing import Any, cast

import hardware
from auth_config import require_admin_key, require_read_auth
from config_store import load_config, load_model, save_config, save_model
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from temperature_controller import TemperatureController
from temperature_models import (
    TcType,
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
    ThermocoupleChannel,
)
from thermocouple_filter import (
    DEFAULT_MAX_STEP_C,
    FilterSample,
    ThermocoupleDeglitchFilter,
)

__all__ = [
    "BurnoutModeRequest",
    "TcTypeRequest",
    "TemperaturePermissiveRequest",
    "TemperatureService",
    "TemperatureSetpointRequest",
    "create_temperature_router",
]

# config_store keys for the persisted temperature settings.
_CONFIG_KEY = "temperature_config"
_SETPOINT_KEY = "temperature_setpoint"
_BURNOUT_KEY = "temperature_burnout"


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
        # Latest reading of each thermocouple in deg C (None until first poll),
        # computed from tags every scan regardless of which TC is controlling.
        # Surfaced in status() so the HMI can display/plot both channels at once,
        # and the non-controlling one is fed to the controller as the cross-check
        # / HH-on-either-TC reference.
        self._last_type_k_c: float | None = None
        self._last_type_r_c: float | None = None
        # Per-channel deglitch filters. The P1-04THM drives an open thermocouple
        # to 0 C (low-side burnout); a raw burnout-zero fed into the control law
        # reads as "cold" and would call for heat (runaway). Each channel's filter
        # rejects an implausible drop to ~0, holds the last-good value through the
        # glitch, and on a *persistent* fault emits fault=True so we trip the
        # heater (fail-safe). The controlling channel's filtered value is what the
        # control law and the HMI see; both channels are always filtered so a
        # switch lands on an already-warm filter.
        # Each filter must be told its own channel's full scale. Constructing
        # them bare pinned both to the module default (1400 C) regardless of
        # what the channel was configured for, so on a shorter-range channel
        # the high-side burnout rail sat above any reachable reading and an
        # open thermocouple was accepted as genuine (issue #4035).
        self._k_filter = self._build_filter(self.controller.config.type_k)
        self._r_filter = self._build_filter(self.controller.config.type_r)
        # True when the CONTROLLING channel's reading is currently being held
        # (a live glitch is being ridden out). Surfaced in status() so the HMI can
        # warn the operator that the control sensor is acting up.
        self._control_sensor_holding = False
        # Latched True once a control-sensor fault has been logged, so the
        # fail-safe trip is logged once (not every scan) until it recovers.
        self._control_sensor_faulted = False
        # P1-04THM open-circuit (burnout) fail direction. True = HIGH-side (an open
        # thermocouple reads full scale -> heater shuts off, fail-safe); False =
        # LOW-side (an open reads 0 C -> looks cold). Persisted + recalled; defaults
        # to the fail-safe direction. Re-asserted to the PLC (coil) every scan so it
        # survives a PLC reboot (the firmware boots low-side).
        self._burnout_high_side = True
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
        cfg = self.controller.config
        now = time.monotonic()
        # Scale BOTH thermocouples every scan, independent of which one controls,
        # then deglitch each so a burnout-zero / dropout can never reach the
        # control law or the HMI as a spurious "cold". The FILTERED per-channel
        # values are what we publish and control on.
        k_sample = self._k_filter.update(self._temp_from_channel(tags, cfg.type_k), now)
        r_sample = self._r_filter.update(self._temp_from_channel(tags, cfg.type_r), now)
        self._last_type_k_c = k_sample.value_c
        self._last_type_r_c = r_sample.value_c

        active_is_k = cfg.active_tc_type == TcType.TYPE_K
        active_sample = k_sample if active_is_k else r_sample
        other_sample = r_sample if active_is_k else k_sample
        self._note_control_sensor_health(active_sample)

        # Pass a monotonic clock so the controller can enforce the
        # anti-short-cycle min on/off dwell across scans. On a persistent
        # control-sensor fault we feed a non-finite value, which trips the
        # controller's TC_FAULT (fail-safe) rather than heating on a stale hold.
        relay_on = self.controller.tick(
            measured_temp_c=self._control_value(active_sample),
            now=now,
            other_temp_c=other_sample.value_c if not other_sample.fault else None,
        )
        await self._write_relay(relay_on)
        await self._assert_burnout_coil()
        return self.status()

    def _rebuild_filters(self) -> None:
        """Re-create both deglitch filters from the controller's live config."""
        config = self.controller.config
        self._k_filter = self._build_filter(config.type_k)
        self._r_filter = self._build_filter(config.type_r)
        self._control_sensor_holding = False

    @staticmethod
    def _build_filter(channel: ThermocoupleChannel) -> ThermocoupleDeglitchFilter:
        """Construct a deglitch filter matched to one channel's range.

        Constructing the filters bare pinned both to the module default full
        scale regardless of what the channel was configured for. On a
        shorter-range channel the high-side burnout rail then sat above any
        reachable reading, so an open thermocouple was accepted as a genuine
        measurement -- the exact condition the filter exists to catch
        (issue #4035).

        ``max_step_c`` is scaled with the range too, so "non-physical
        single-scan step" means the same fraction of span on every channel.
        """
        span_ratio = channel.full_scale_c / hardware.THERMOCOUPLE_FULL_SCALE_C
        return ThermocoupleDeglitchFilter(
            full_scale_c=channel.full_scale_c,
            max_step_c=DEFAULT_MAX_STEP_C * span_ratio,
        )

    @staticmethod
    def _control_value(sample: FilterSample) -> float:
        """The value to feed the control law from a filtered sample.

        A persistent fault becomes NaN so the controller's existing non-finite
        ``TC_FAULT`` check trips the heater (fail-safe); a not-yet-seen reading
        becomes a benign 0.0 (the controller is IDLE at startup so this cannot
        energize); otherwise the held/accepted value is used.
        """
        if sample.fault:
            return math.nan
        if sample.value_c is None:
            return 0.0
        return float(sample.value_c)

    def _note_control_sensor_health(self, active: FilterSample) -> None:
        """Track + log the controlling sensor's filter state on each scan.

        Logs once on each transition (into holding, into fault, and back to OK)
        so the journal shows when the control sensor started glitching without
        spamming a line every scan.
        """
        was_holding = self._control_sensor_holding
        self._control_sensor_holding = active.holding
        if active.fault and not self._control_sensor_faulted:
            self._control_sensor_faulted = True
            self._logger.error(
                "control thermocouple FAULT — reading bad past hold timeout; "
                "tripping heater (fail-safe)"
            )
        elif active.holding and not was_holding:
            self._logger.warning(
                "control thermocouple glitch — holding last-good %.1f C",
                active.value_c if active.value_c is not None else float("nan"),
            )
        elif not active.holding and was_holding:
            self._control_sensor_faulted = False
            self._logger.info("control thermocouple recovered")

    def status(self) -> TemperatureStatus:
        """Controller status augmented with service-owned readings.

        Returns the controller snapshot with ``last_setpoint_c`` set to the value
        recalled from persisted settings (``None`` when nothing was recalled) and
        ``type_k_temp_c`` / ``type_r_temp_c`` set to the latest per-channel
        readings from the most recent poll (``None`` before the first scan), so
        the HMI can pre-fill the target field and show/plot both thermocouples
        without the service leaking any controller internals (LOD).
        """
        status: TemperatureStatus = self.controller.status()
        status.last_setpoint_c = self._last_setpoint_c
        status.type_k_temp_c = self._last_type_k_c
        status.type_r_temp_c = self._last_type_r_c
        status.control_sensor_holding = self._control_sensor_holding
        status.burnout_high_side = self._burnout_high_side
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
        # Rebuild the filters against the new ranges. A filter carrying state
        # from the previous scaling would see the rescaled reading as a
        # non-physical step and hold, then trip TC_FAULT mid-run for a change
        # that only touched a conversion factor (issue #4035).
        self._rebuild_filters()
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

    def set_burnout_high_side(self, high_side: bool) -> bool:
        """Select the P1-04THM open-circuit (burnout) fail direction and persist.

        HIGH-side (True) makes an open thermocouple read full scale, so the heater
        shuts off (fail-safe); LOW-side (False) makes an open read 0 C (looks cold).
        The new direction is written to the PLC on the next scan (and re-asserted
        every scan thereafter). Returns the applied value.

        Raises:
            TypeError: if ``high_side`` is not a bool.
        """
        if not isinstance(high_side, bool):
            raise TypeError(f"high_side must be a bool, got {type(high_side).__name__}")
        self._burnout_high_side = high_side
        self._persist_burnout(high_side)
        return high_side

    @property
    def burnout_high_side(self) -> bool:
        """Current commanded burnout fail direction (True = high-side/fail-safe)."""
        return self._burnout_high_side

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
                value = float(cast(float, data["value_c"]))
                self._last_setpoint_c = value
                # Seed the controller's held setpoint so status().setpoint_c
                # reports the recalled target at boot (matching last_setpoint_c
                # and the pre-filled HMI box) instead of 0. SAFE: preload only
                # applies in IDLE and the relay is force-held off until the
                # operator arms and runs — restoring never energizes the heater.
                self.controller.preload_setpoint_c(value)
            burnout = load_config(session, _BURNOUT_KEY)
            if burnout is not None and "high_side" in burnout:
                # Recall the burnout direction; it is re-asserted to the PLC each
                # scan. Defaults to fail-safe (high-side) when nothing was stored.
                self._burnout_high_side = bool(burnout["high_side"])
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

    def _persist_burnout(self, high_side: bool) -> None:
        """Persist the burnout fail direction (best-effort, no-op if disabled)."""
        if self._session_factory is None:
            return
        try:
            with self._session_factory() as s:
                save_config(s, _BURNOUT_KEY, {"high_side": high_side})
        except Exception as exc:  # noqa: BLE001 - persistence must not break control
            self._logger.warning("temperature burnout persist failed: %s", exc)

    async def _assert_burnout_coil(self) -> None:
        """Re-assert the burnout-direction coil to the PLC (best-effort).

        Written every scan so the firmware always matches the commanded direction
        even after a PLC reboot (the firmware boots low-side). The firmware only
        reconfigures the module when the coil actually changes, so re-writing the
        same value each scan is cheap and idempotent. Never raises: a failed write
        must not abort the scan loop.
        """
        try:
            await self._plc_client.write_coil(
                hardware.THM_BURNOUT_COIL, self._burnout_high_side
            )
        except Exception as exc:  # noqa: BLE001 - best-effort, never abort the scan
            self._logger.debug("burnout coil write failed: %s", exc)

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

    @staticmethod
    def _temp_from_channel(tags: dict[str, float] | None, channel: Any) -> float | None:
        """Scale one thermocouple channel's tag (percent) into deg C, or None.

        Returns None when there is no scan data yet (``tags`` is falsy) so the HMI
        can distinguish "not read" from a genuine 0 deg C. Uses the SAME
        percent-of-full-scale conversion as ``_temp_from_tags`` (DRY), applied to
        the given channel rather than the active one, so both K and R are computed
        identically regardless of which is controlling.
        """
        if not tags:
            return None
        pct = float(tags.get(channel.tag, 0.0))
        return float(pct * channel.full_scale_c / 100.0)

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


class BurnoutModeRequest(BaseModel):
    """Operator selection of the P1-04THM open-circuit fail direction.

    high_side True = an open thermocouple reads full scale (heater shuts off,
    fail-safe); False = an open reads 0 C (looks cold, fail-dangerous).
    """

    high_side: bool


def create_temperature_router(service: TemperatureService) -> APIRouter:
    router = APIRouter(prefix="/api/temperature", tags=["temperature"])
    controller = service.controller

    @router.get(
        "/config",
        response_model=TemperatureConfig,
        dependencies=[Depends(require_read_auth)],
    )
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

    @router.get(
        "/status",
        response_model=TemperatureStatus,
        dependencies=[Depends(require_read_auth)],
    )
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

    @router.post("/burnout_mode", dependencies=[Depends(require_admin_key)])
    async def set_burnout_mode(req: BurnoutModeRequest) -> TemperatureStatus:
        """Select the P1-04THM open-circuit fail direction (high-side/low-side)."""
        try:
            service.set_burnout_high_side(req.high_side)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return service.status()

    @router.post("/acknowledge_trip", dependencies=[Depends(require_admin_key)])
    async def acknowledge_temperature_trip() -> TemperatureStatus:
        controller.acknowledge_trip()
        status: TemperatureStatus = controller.status()
        return status

    return router
