"""Temperature state machine, safety interlocks, and on/off control law.

Controls a resistive heater through a single on/off relay using thermocouple
temperature feedback. The controller is fully testable without a PLC: feed
measured temperatures into ``tick()`` and inspect state plus the commanded
relay. The relay is driven by an on/off hysteresis band around the setpoint.

Trips latch until ``acknowledge_trip()``; E-stop latches until
``clear_estop()``. A high-high (HH) cutoff latches the controller TRIPPED and
forces the relay off the moment the measured temperature reaches the limit.
"""

from __future__ import annotations

import logging
import math

from safety_state_machine import SafetyStateMachine
from temperature_models import (
    TcPath,
    TcType,
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
    ThermocoupleChannel,
)

__all__ = [
    "TcPath",
    "TcType",
    "TemperatureConfig",
    "TemperatureController",
    "TemperatureState",
    "TemperatureStatus",
    "ThermocoupleChannel",
]

logger = logging.getLogger("dcs_backend.temperature")

# --------------------------------------------------------------------------
# Cross-check ("controlling sensor stuck") trip thresholds.
#
# The type-R incident showed the worst failure this system can have: the
# controlling thermocouple died and read a fixed ~34 C ("cold") while the vessel
# was really >790 C, so the on/off law called for heat indefinitely with no
# over-temperature protection until HH. When BOTH thermocouples are available we
# can catch this fast: if the CONTROLLING sensor reads essentially cold while the
# other reads clearly hot AND the heater is running, the controlling sensor is
# lying and we trip. The band is deliberately wide (cold vs. hot, not a tight
# delta) so a legitimate inter-probe gradient never false-trips, and it is
# debounced over several scans so a single transient read can never trip it.
_CROSS_FAULT_ACTIVE_MAX_C = 100.0  # controlling TC must read below this ("cold")
_CROSS_FAULT_OTHER_MIN_C = 200.0  # other TC must read at/above this ("hot")
_CROSS_FAULT_DEBOUNCE_SCANS = 5  # consecutive disagreeing scans before latching


class TemperatureController(SafetyStateMachine[TemperatureState]):
    """State machine + on/off control law for the resistive heater.

    In non-running or unsafe states (IDLE / ARMED / TRIPPED / no-permissive /
    E-stop / any trip), the relay is always forced off. In RUNNING, the relay
    follows an on/off hysteresis band around the setpoint: it switches ON at
    ``setpoint - deadband`` and OFF at ``setpoint + deadband``, holding its
    previous state inside the band.
    """

    # Concrete state type for the checker. The SafetyStateMachine base is generic
    # (``_state: StateT``); naming the resolved type here also keeps it typed
    # under CI's mypy --follow-imports=skip, where the base module is elided.
    _state: TemperatureState

    def __init__(self, config: TemperatureConfig) -> None:
        if not isinstance(config, TemperatureConfig):
            raise TypeError(
                f"config must be TemperatureConfig, got {type(config).__name__}"
            )
        super().__init__(
            idle=TemperatureState.IDLE,
            armed=TemperatureState.ARMED,
            running=TemperatureState.RUNNING,
            tripped=TemperatureState.TRIPPED,
            logger=logger,
        )
        self._config = config
        self._setpoint_c = 0.0
        self._last_t = 0.0
        # Most recent reading of the OTHER (non-controlling) thermocouple in
        # deg C, or None when it wasn't supplied / wasn't finite. Used only for
        # the HH-on-either-TC backstop and the cross-check trip below; it never
        # drives the control law (that always follows the active TC via _last_t).
        self._last_other_t: float | None = None
        # Consecutive scans the cross-check disagreement has held. Debounces the
        # stuck-controlling-sensor trip so a single transient can never latch it.
        self._cross_fault_scans = 0
        # Commanded relay state from the most recent tick. Held across ticks so
        # the hysteresis band can keep the relay in its prior position while the
        # measured temperature sits inside the deadband.
        self._relay_on = False
        # Monotonic-seconds timestamp of the last relay state change, used to
        # enforce the anti-short-cycle min on/off dwell. None until first switch.
        self._last_switch_t: float | None = None

    @property
    def config(self) -> TemperatureConfig:
        return self._config

    def _clamp_setpoint(self, value_c: float) -> float:
        """Clamp a setpoint to the configured ``[min, max]`` band (DRY).

        Single source of truth for the setpoint bound check shared by
        ``set_setpoint_c``, ``preload_setpoint_c``, ``update_config`` and
        ``set_active_tc_type`` (the last two re-clamp against the NEW config).
        """
        return float(
            max(self._config.setpoint_min_c, min(value_c, self._config.setpoint_max_c))
        )

    def update_config(self, new_config: TemperatureConfig) -> None:
        """Replace operator-configurable parameters.

        Precondition: new_config is a fully-validated TemperatureConfig.
        Postcondition: subsequent commands and trip checks use new_config; the
        current setpoint is re-clamped to the new bounds without changing state.

        Raises:
            TypeError: if new_config is not a TemperatureConfig instance.
        """
        if not isinstance(new_config, TemperatureConfig):
            raise TypeError(
                f"new_config must be TemperatureConfig, got {type(new_config).__name__}"
            )
        if self._estopped:
            # A latched E-stop is one-way: safety limits must not be re-tuned
            # while it is engaged (consistent with set_setpoint_c / set_permissive).
            logger.warning("temperature config change ignored — E-stop is latched")
            return
        self._config = new_config
        # Re-clamp current setpoint to new bounds without changing state.
        self._setpoint_c = self._clamp_setpoint(self._setpoint_c)

    def set_active_source(self, tc_type: TcType, tc_path: TcPath) -> None:
        """Select the controlling thermocouple SOURCE (type + acquisition path).

        A source is one of the 2x2 combinations of type (K/R) and path (TC card
        vs analog conditioner). Switches the active channel and re-clamps the
        setpoint / safety limits to the newly-active channel's full scale, so
        every resulting state stays valid and the rest of the system keeps
        reading the controlled temperature through ``config.temp_tag`` /
        ``config.temp_full_scale_c`` (DRY/LOD). The state machine
        (IDLE/ARMED/RUNNING/TRIPPED) is unchanged.

        Precondition: tc_type is a TcType and tc_path is a TcPath.
        Postcondition: config.active_tc_type == tc_type and
        config.active_tc_path == tc_path; setpoint within the new active band.

        Raises:
            TypeError: if tc_type is not a TcType or tc_path is not a TcPath.
            ValueError: if the resulting config is invalid (e.g. the target
                channel's full scale is below setpoint_min_c).
        """
        if not isinstance(tc_type, TcType):
            raise TypeError(f"tc_type must be a TcType, got {type(tc_type).__name__}")
        if not isinstance(tc_path, TcPath):
            raise TypeError(f"tc_path must be a TcPath, got {type(tc_path).__name__}")
        if self._estopped:
            # Don't let a source switch ratchet the safety limits while E-stopped.
            logger.warning("TC-source change ignored — E-stop is latched")
            return
        full_scale = self._config.channel_for(tc_type, tc_path).full_scale_c
        data = self._config.model_dump(
            exclude={"temp_tag", "temp_full_scale_c", "active_tc_label"}
        )
        data["active_tc_type"] = tc_type
        data["active_tc_path"] = tc_path
        data["setpoint_max_c"] = min(self._config.setpoint_max_c, full_scale)
        data["hh_limit_c"] = min(self._config.hh_limit_c, full_scale)
        # Reconstruct through the constructor so the cross-field invariants are
        # re-checked (raises ValueError on a degenerate channel configuration).
        self._config = TemperatureConfig(**data)
        self._setpoint_c = self._clamp_setpoint(self._setpoint_c)

    def set_active_tc_type(self, tc_type: TcType) -> None:
        """Select the thermocouple type (K/R), keeping the current path.

        Backward-compatible shim over :meth:`set_active_source`: callers that
        only care about type (the original two-way K/R toggle) keep working while
        the acquisition path is preserved. See ``set_active_source`` for the full
        contract, preconditions, and raised exceptions.
        """
        self.set_active_source(tc_type, self._config.active_tc_path)

    def set_permissive(self, on: bool) -> None:
        """Toggle permissive. A trip latch is not cleared by a permissive change.

        Precondition: on is a bool (no truthy coercion to catch caller bugs).
        Postcondition:
            - permissive == on
            - If on goes False and state was RUNNING / ARMED, state -> IDLE,
              setpoint -> 0
            - If on goes True and state was IDLE, state -> ARMED
            - If state was TRIPPED, state remains TRIPPED
            - If E-stop is latched, on=True is ignored (stays disarmed).

        Raises:
            TypeError: if on is not exactly bool.
        """
        self._apply_permissive(on)

    def _on_disarm(self) -> None:
        self._setpoint_c = 0.0

    def set_setpoint_c(self, value_c: float) -> float:
        """Apply a temperature setpoint (deg C).

        The value is clamped to [setpoint_min_c, setpoint_max_c] before being
        applied. Setpoint application is rejected in IDLE and TRIPPED states
        (returns clamped value but does not change internal setpoint).

        Precondition: value_c is finite numeric.
        Postcondition:
            - If state was ARMED / RUNNING: internal setpoint_c == clamped value
            - If clamped > 0 and state was ARMED: state -> RUNNING
            - Returns the clamped value that would be applied (0.0 if estopped).

        Raises:
            TypeError: if value_c is not numeric.
            ValueError: if value_c is NaN or infinite.
        """
        if not isinstance(value_c, int | float) or isinstance(value_c, bool):
            raise TypeError(f"value_c must be numeric, got {type(value_c).__name__}")
        v = float(value_c)
        if not math.isfinite(v):
            raise ValueError(f"value_c must be finite, got {v}")

        if self._estopped:
            logger.warning("temperature setpoint ignored — E-stop is latched")
            return 0.0

        clamped = self._clamp_setpoint(v)

        if self._state in (TemperatureState.ARMED, TemperatureState.RUNNING):
            self._setpoint_c = clamped
            if self._state == TemperatureState.ARMED and clamped > 0.0:
                self._state = TemperatureState.RUNNING
        elif self._state == TemperatureState.IDLE:
            logger.warning(
                "temperature setpoint %.1f C ignored — controller in IDLE; "
                "enable permissive first",
                clamped,
            )
        elif self._state == TemperatureState.TRIPPED:
            logger.warning(
                "temperature setpoint %.1f C ignored — controller TRIPPED; "
                "acknowledge first",
                clamped,
            )
        return clamped

    def preload_setpoint_c(self, value_c: float) -> float:
        """Seed the held setpoint at boot WITHOUT arming or energizing.

        Recalls a persisted target so ``status().setpoint_c`` reports what the
        controller will heat to, matching the pre-filled HMI entry box, instead
        of reading 0 until the operator presses Start. The state machine stays
        IDLE and the relay stays force-held off (``_should_force_actuator_off``
        is True in every non-RUNNING state), so seeding can never energize the
        heater. Applied ONLY in IDLE and never while E-stopped, so it can never
        resurrect a target into an armed/running/tripped controller. Clamped to
        ``[setpoint_min_c, setpoint_max_c]``.

        Precondition: value_c is finite numeric.
        Postcondition:
            - In IDLE (not E-stopped): _setpoint_c == clamped value, state
              unchanged (IDLE); returns the clamped value.
            - In any other state (ARMED/RUNNING/TRIPPED) or while E-stopped:
              nothing changes and 0.0 is returned.

        Raises:
            TypeError: if value_c is not numeric.
            ValueError: if value_c is NaN or infinite.
        """
        if not isinstance(value_c, int | float) or isinstance(value_c, bool):
            raise TypeError(f"value_c must be numeric, got {type(value_c).__name__}")
        v = float(value_c)
        if not math.isfinite(v):
            raise ValueError(f"value_c must be finite, got {v}")
        if self._estopped or self._state != TemperatureState.IDLE:
            return 0.0
        clamped = self._clamp_setpoint(v)
        self._setpoint_c = clamped
        return clamped

    def _estop_log_message(self) -> str:
        return "E-STOP engaged — heater relay latched off"

    def _on_estop_engaged(self) -> None:
        # engage_estop (inherited) forces IDLE / disarms; here we drop the
        # heater-specific setpoint and latch the commanded relay off.
        self._setpoint_c = 0.0
        self._relay_on = False

    def _on_estop_cleared(self) -> None:
        self._setpoint_c = 0.0

    def _on_trip_acknowledged(self) -> None:
        self._setpoint_c = 0.0

    @staticmethod
    def _finite_or_none(value: float | None) -> float | None:
        """Return ``value`` as a finite float, or None if missing/garbage.

        The other-TC reading is advisory safety data, not control feedback, so an
        absent or non-finite value simply means "no cross-check data this scan"
        rather than a fault — unlike the controlling TC, whose garbage reading
        trips ``_evaluate_sensor_fault``.
        """
        if value is None or isinstance(value, bool):
            return None
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            return None
        return float(value)

    def _evaluate_sensor_fault(self, raw_temp: float) -> None:
        """Latch a TC_FAULT trip on a non-finite/garbage feedback while RUNNING.

        An open or failed thermocouple can read non-finite (NaN/inf) or a
        non-numeric junk value. ``_safe_finite`` would coerce that to 0 C
        ("cold"), so the on/off law would call for heat indefinitely — the
        classic heater-runaway failure mode. Treat a bad reading as a sensor
        fault and trip (fail-safe). Scoped to RUNNING because that is the only
        state in which the control law can energize the relay; in every other
        state the relay is already force-held off, so a transient junk reading
        during bring-up doesn't latch a spurious trip.
        """
        bad = (
            isinstance(raw_temp, bool)
            or not isinstance(raw_temp, int | float)
            or not math.isfinite(float(raw_temp))
        )
        if bad and self._state == TemperatureState.RUNNING:
            self._trips.add("TC_FAULT")

    def _evaluate_trips(self) -> None:
        """Latch every temperature trip for this scan, then flip to TRIPPED.

        Adds any breached trip to ``self._trips`` first and latches once, so a
        trip discovered by the cross-check (evaluated after HH) still flips the
        state in the SAME tick rather than lagging a scan.
        """
        self._evaluate_hh_cutoff()
        self._evaluate_cross_fault()
        self._latch_trips(log_context=self._trip_log_context())

    def _evaluate_hh_cutoff(self) -> None:
        """Latch HH when EITHER thermocouple reaches the high-high limit.

        Checking both channels means a stuck/dead *controlling* sensor cannot
        mask a real over-temperature: if the vessel is genuinely hot, the healthy
        channel still trips the cutoff even while the controller reads the dead
        one at a false "cold". The other channel only counts when it was supplied
        (``_last_other_t`` is not None), so single-TC callers are unaffected.
        """
        hottest = self._last_t
        if self._last_other_t is not None:
            hottest = max(hottest, self._last_other_t)
        if hottest >= self._config.hh_limit_c:
            self._trips.add("HH_TEMP")

    def _evaluate_cross_fault(self) -> None:
        """Debounced trip when the CONTROLLING thermocouple is stuck cold.

        Fires only when, for ``_CROSS_FAULT_DEBOUNCE_SCANS`` consecutive scans,
        the controller is RUNNING, the controlling TC reads essentially cold
        (< ``_CROSS_FAULT_ACTIVE_MAX_C``), and the other TC reads clearly hot
        (>= ``_CROSS_FAULT_OTHER_MIN_C``). That combination — heating while the
        sensor we steer on says "cold" but its neighbour says "hot" — is the
        signature of a dead/disconnected controlling thermocouple (the type-R
        incident), which the on/off law would otherwise answer with unbounded
        heat. Any disagreeing streak that breaks resets the debounce, so only a
        sustained fault latches ``TC_DISAGREE``.
        """
        other = self._last_other_t
        disagreeing = (
            self._state == TemperatureState.RUNNING
            and other is not None
            and self._last_t < _CROSS_FAULT_ACTIVE_MAX_C
            and other >= _CROSS_FAULT_OTHER_MIN_C
        )
        if not disagreeing:
            self._cross_fault_scans = 0
            return
        self._cross_fault_scans += 1
        if self._cross_fault_scans >= _CROSS_FAULT_DEBOUNCE_SCANS:
            self._trips.add("TC_DISAGREE")

    def _trip_log_context(self) -> str:
        """Human-readable temperature context for the trip log line."""
        if self._last_other_t is None:
            return f"T={self._last_t:.1f} C"
        return f"T={self._last_t:.1f} C, other={self._last_other_t:.1f} C"

    def _should_force_relay_off(self) -> bool:
        """All "kill the heater now" conditions in one place."""
        return bool(self._should_force_actuator_off())

    def _set_relay(self, on: bool, now: float | None) -> None:
        """Apply a commanded relay state, recording the switch time so the
        anti-short-cycle dwell can be measured. A no-op when unchanged."""
        if on != self._relay_on:
            self._relay_on = on
            if now is not None:
                self._last_switch_t = now

    def _dwell_blocks_switch(self, now: float | None) -> bool:
        """True when the anti-short-cycle dwell has not yet elapsed since the
        last switch, so an opposite relay demand must wait.

        Disabled (returns False) when no clock is supplied, no switch has
        happened yet, or the relevant min on/off time is 0.
        """
        if now is None or self._last_switch_t is None:
            return False
        min_dwell = (
            self._config.min_on_time_s
            if self._relay_on
            else self._config.min_off_time_s
        )
        if min_dwell <= 0.0:
            return False
        return bool((now - self._last_switch_t) < min_dwell)

    def tick(
        self,
        measured_temp_c: float,
        now: float | None = None,
        *,
        other_temp_c: float | None = None,
    ) -> bool:
        """Advance the controller one cycle and return the commanded relay state.

        Reads the temperature feedback, evaluates the HH cutoff (latching it on
        breach), and applies the on/off hysteresis band when RUNNING.

        On/off behavior (per `deadband_c` in the config):
          - Relay turns ON when measured <= setpoint - deadband.
          - Relay turns OFF when measured >= setpoint + deadband.
          - Inside the band the relay HOLDS its previous state, so it doesn't
            chatter around the setpoint.
          - Anti-short-cycle: after a switch the relay is held for at least
            `min_on_time_s` / `min_off_time_s` before an opposite demand is
            honored, capping how often the heater cycles. Enforced only when a
            clock is supplied via `now` (the live scan loop passes one).
          - Any path that forces the relay off (E-stop / IDLE / ARMED /
            TRIPPED / permissive off / any trip) bypasses the band AND the
            dwell, so shutdowns are always one tick.

        Args:
            measured_temp_c: controlling thermocouple temperature in deg C.
            now: Monotonic timestamp in seconds, used to enforce the min on/off
                dwell. When None the dwell is not enforced (the control law
                stays deterministic for unit tests that pass explicit times).
            other_temp_c: reading of the OTHER (non-controlling) thermocouple in
                deg C, or None when unavailable. Used only by the HH-on-either-TC
                backstop and the cross-check trip — never by the control law. A
                non-finite value is treated as None (no safety data this scan).

        Precondition: measured_temp_c is a finite float. Non-finite inputs are
        treated as 0 (safe).
        Postcondition: returns the commanded relay state; HH trip is latched if
        its condition is met; the relay state is stored for the next tick.

        Returns:
            True if the heater relay should be energized, else False.
        """
        # Detect a sensor fault from the RAW reading before it is coerced to a
        # finite value, so an open thermocouple trips instead of reading "cold".
        self._evaluate_sensor_fault(measured_temp_c)
        self._last_t = self._safe_finite(measured_temp_c)
        self._last_other_t = self._finite_or_none(other_temp_c)

        self._evaluate_trips()

        if self._should_force_relay_off():
            # Safety always wins and bypasses the anti-short-cycle dwell.
            self._set_relay(False, now)
            return False

        # RUNNING on/off hysteresis around the setpoint.
        on_threshold = self._setpoint_c - self._config.deadband_c
        off_threshold = self._setpoint_c + self._config.deadband_c
        if self._last_t <= on_threshold:
            desired = True
        elif self._last_t >= off_threshold:
            desired = False
        else:
            desired = self._relay_on  # inside the band -> hold

        # Anti-short-cycle: make an opposite demand wait out the min dwell.
        if desired != self._relay_on and self._dwell_blocks_switch(now):
            desired = self._relay_on

        self._set_relay(desired, now)
        return self._relay_on

    def status(self) -> TemperatureStatus:
        """Return a snapshot of controller state for serialization."""
        return TemperatureStatus(
            state=self._state,
            permissive=self._permissive,
            setpoint_c=self._setpoint_c,
            measured_temp_c=self._last_t,
            relay_on=self._relay_on,
            trips=sorted(self._trips),
            hh_limit_c=self._config.hh_limit_c,
            deadband_c=self._config.deadband_c,
            min_on_time_s=self._config.min_on_time_s,
            min_off_time_s=self._config.min_off_time_s,
            active_tc_type=self._config.active_tc_type,
            active_tc_path=self._config.active_tc_path,
            active_tc_label=self._config.active_tc_label,
            estopped=self._estopped,
        )
