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

from temperature_models import (
    TemperatureConfig,
    TemperatureState,
    TemperatureStatus,
)

__all__ = [
    "TemperatureConfig",
    "TemperatureController",
    "TemperatureState",
    "TemperatureStatus",
]

logger = logging.getLogger("dcs_backend.temperature")


class TemperatureController:
    """State machine + on/off control law for the resistive heater.

    In non-running or unsafe states (IDLE / ARMED / TRIPPED / no-permissive /
    E-stop / any trip), the relay is always forced off. In RUNNING, the relay
    follows an on/off hysteresis band around the setpoint: it switches ON at
    ``setpoint - deadband`` and OFF at ``setpoint + deadband``, holding its
    previous state inside the band.
    """

    def __init__(self, config: TemperatureConfig) -> None:
        if not isinstance(config, TemperatureConfig):
            raise TypeError(
                f"config must be TemperatureConfig, got {type(config).__name__}"
            )
        self._config = config
        self._state = TemperatureState.IDLE
        self._permissive = False
        self._setpoint_c = 0.0
        self._trips: set[str] = set()
        self._last_t = 0.0
        # Commanded relay state from the most recent tick. Held across ticks so
        # the hysteresis band can keep the relay in its prior position while the
        # measured temperature sits inside the deadband.
        self._relay_on = False
        # Monotonic-seconds timestamp of the last relay state change, used to
        # enforce the anti-short-cycle min on/off dwell. None until first switch.
        self._last_switch_t: float | None = None
        # E-stop latch. While set, the controller forces the relay off, refuses
        # to arm, and rejects setpoints until clear_estop() is called. This is a
        # one-way kill — it must be explicitly cleared by an operator.
        self._estopped = False

    @property
    def state(self) -> TemperatureState:
        return self._state

    @property
    def permissive(self) -> bool:
        return self._permissive

    @property
    def config(self) -> TemperatureConfig:
        return self._config

    @property
    def trips(self) -> list[str]:
        return sorted(self._trips)

    @property
    def estopped(self) -> bool:
        return self._estopped

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
        self._config = new_config
        # Re-clamp current setpoint to new bounds without changing state.
        self._setpoint_c = max(
            new_config.setpoint_min_c,
            min(self._setpoint_c, new_config.setpoint_max_c),
        )

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
        if not isinstance(on, bool):
            raise TypeError(f"on must be bool, got {type(on).__name__}")
        if self._estopped:
            # A latched E-stop cannot be armed around. The relay stays off until
            # the operator explicitly clears the E-stop.
            if on:
                logger.warning("permissive ON ignored — E-stop is latched")
            self._permissive = False
            return
        self._permissive = on
        if self._state == TemperatureState.TRIPPED:
            return
        if on:
            if self._state == TemperatureState.IDLE:
                self._state = TemperatureState.ARMED
        else:
            self._setpoint_c = 0.0
            self._state = TemperatureState.IDLE

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

        clamped = float(
            max(
                self._config.setpoint_min_c,
                min(v, self._config.setpoint_max_c),
            )
        )

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

    def engage_estop(self) -> None:
        """Latch the emergency stop: force the relay off and disarm.

        This is the software half of the kill switch. It immediately drops the
        commanded relay to off (via the latch checked in `tick()`), clears the
        setpoint, turns permissive off, and returns the state machine to IDLE.
        The latch persists — `set_permissive(True)` and setpoint commands are
        rejected — until `clear_estop()` is called.

        Postcondition: estopped; permissive False; state IDLE; setpoint 0;
        relay off.
        """
        self._estopped = True
        self._permissive = False
        self._setpoint_c = 0.0
        self._state = TemperatureState.IDLE
        self._relay_on = False
        logger.error("E-STOP engaged — heater relay latched off")

    def clear_estop(self) -> None:
        """Release the emergency-stop latch.

        Leaves the controller IDLE with permissive off, so the operator must
        deliberately re-arm (permissive on) and re-enter a setpoint before the
        heater can fire again.

        Postcondition: not estopped; permissive False; state IDLE; setpoint 0.
        """
        if not self._estopped:
            return
        self._estopped = False
        self._permissive = False
        self._setpoint_c = 0.0
        self._state = TemperatureState.IDLE
        logger.warning("E-stop cleared — controller idle, re-arm required")

    def acknowledge_trip(self) -> bool:
        """Clear latched trips and return controller to safe idle/armed state.

        Postcondition:
            - All trips cleared.
            - Setpoint cleared to 0.
            - State -> ARMED if permissive is True, else IDLE.

        Returns:
            True if a trip was cleared. False if there were no trips.
        """
        if self._state != TemperatureState.TRIPPED:
            return False
        self._trips.clear()
        self._setpoint_c = 0.0
        self._state = (
            TemperatureState.ARMED if self._permissive else TemperatureState.IDLE
        )
        return True

    @staticmethod
    def _safe_finite(value: float) -> float:
        """Coerce a feedback input to a finite float; non-finite values map
        to 0 so a sensor failure can never accidentally hold the relay on or
        smuggle a NaN through the comparisons."""
        if not isinstance(value, int | float) or isinstance(value, bool):
            return 0.0
        v = float(value)
        return v if math.isfinite(v) else 0.0

    def _evaluate_trips(self) -> None:
        """Latch the HH cutoff based on the latest temperature; flips state to
        TRIPPED on first breach."""
        if self._last_t >= self._config.hh_limit_c:
            self._trips.add("HH_TEMP")
        if self._trips and self._state != TemperatureState.TRIPPED:
            logger.error(
                "trip latched: %s (T=%.1f C)",
                ",".join(sorted(self._trips)),
                self._last_t,
            )
            self._state = TemperatureState.TRIPPED

    def _should_force_relay_off(self) -> bool:
        """All "kill the heater now" conditions in one place."""
        return (
            self._estopped
            or self._state != TemperatureState.RUNNING
            or not self._permissive
            or bool(self._trips)
        )

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

    def tick(self, measured_temp_c: float, now: float | None = None) -> bool:
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
            measured_temp_c: thermocouple temperature in deg C.
            now: Monotonic timestamp in seconds, used to enforce the min on/off
                dwell. When None the dwell is not enforced (the control law
                stays deterministic for unit tests that pass explicit times).

        Precondition: measured_temp_c is a finite float. Non-finite inputs are
        treated as 0 (safe).
        Postcondition: returns the commanded relay state; HH trip is latched if
        its condition is met; the relay state is stored for the next tick.

        Returns:
            True if the heater relay should be energized, else False.
        """
        self._last_t = self._safe_finite(measured_temp_c)

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
            estopped=self._estopped,
        )
