"""Power-supply controller — state machine, safety interlocks, control law.

Provides the operator-facing model and control logic that the FastAPI layer
and the React tab wrap. Designed so it can be tested completely without a
PLC connection by feeding measured values into `tick()` and observing the
state changes and commanded output.

State machine:
    IDLE     -- permissive off; output forced to 0 %
    ARMED    -- permissive on, no setpoint applied; output 0 %
    RUNNING  -- commanding output at setpoint, no trip active
    TRIPPED  -- HH-temp or HH-power trip latched; output 0 % until ack

Tripping is one-way until `acknowledge_trip()` is called. A trip latches even
if the underlying signal returns to the safe band on the next tick.
"""

from __future__ import annotations

import logging
import math
import time

from power_supply_models import (
    PowerSupplyConfig,
    PowerSupplyMode,
    PowerSupplyState,
    PowerSupplyStatus,
)

__all__ = [
    "PowerSupplyConfig",
    "PowerSupplyController",
    "PowerSupplyMode",
    "PowerSupplyState",
    "PowerSupplyStatus",
]

logger = logging.getLogger("dcs_backend.power_supply")


class PowerSupplyController:
    """State machine + control law for the power supply.

    The controller is fed measured feedback values via `tick()`. Each tick:
        1. Updates internal feedback snapshot
        2. Computes measured power (V * I)
        3. Evaluates trip conditions (HH_POWER, HH_TEMP) and latches trips
        4. Recomputes the implied current setpoint when in POWER mode
        5. Returns the AO command percentage (0..100) to apply

    Invariants:
        - In IDLE / ARMED / TRIPPED state, returned command is 0.0.
        - When permissive is False, returned command is 0.0.
        - When any trip is latched, returned command is 0.0.
        - When state is RUNNING, returned command tracks setpoint clamped to
          [current_setpoint_min_a, current_setpoint_max_a], scaled to a percent
          of current_full_scale_a, then capped at config.output_clamp_percent.
    """

    def __init__(self, config: PowerSupplyConfig) -> None:
        if not isinstance(config, PowerSupplyConfig):
            raise TypeError(
                f"config must be PowerSupplyConfig, got {type(config).__name__}"
            )
        self._config = config
        self._state = PowerSupplyState.IDLE
        self._mode = PowerSupplyMode.CURRENT
        self._permissive = False
        self._setpoint_a = 0.0
        self._setpoint_w: float | None = None
        self._trips: set[str] = set()
        self._last_v = 0.0
        self._last_i = 0.0
        self._last_t = 0.0
        self._last_commanded_percent = 0.0
        # Slew-rate state: the actual percent we last sent to the AO (after
        # ramp limiting) and the monotonic timestamp of the last tick. Both
        # reset to zero whenever output is forced to zero (IDLE/ARMED/TRIPPED/
        # no-permissive) so the next RUNNING period starts the ramp from
        # zero rather than snapping up.
        self._slewed_percent = 0.0
        self._last_tick_monotonic: float | None = None
        # True when the output clamp is actively limiting the command (the
        # setpoint would otherwise drive the AO above output_clamp_percent).
        self._output_clamped = False

    @property
    def state(self) -> PowerSupplyState:
        return self._state

    @property
    def mode(self) -> PowerSupplyMode:
        return self._mode

    @property
    def permissive(self) -> bool:
        return self._permissive

    @property
    def config(self) -> PowerSupplyConfig:
        return self._config

    @property
    def trips(self) -> list[str]:
        return sorted(self._trips)

    def update_config(self, new_config: PowerSupplyConfig) -> None:
        """Replace operator-configurable parameters.

        Precondition: new_config is a fully-validated PowerSupplyConfig.
        Postcondition: subsequent commands and trip checks use new_config.

        Raises:
            TypeError: if new_config is not a PowerSupplyConfig instance.
        """
        if not isinstance(new_config, PowerSupplyConfig):
            raise TypeError(
                f"new_config must be PowerSupplyConfig, got {type(new_config).__name__}"
            )
        self._config = new_config
        # Re-clamp current setpoint to new bounds without changing state.
        self._setpoint_a = max(
            new_config.current_setpoint_min_a,
            min(self._setpoint_a, new_config.current_setpoint_max_a),
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

        Raises:
            TypeError: if on is not exactly bool.
        """
        if not isinstance(on, bool):
            raise TypeError(f"on must be bool, got {type(on).__name__}")
        self._permissive = on
        if self._state == PowerSupplyState.TRIPPED:
            return
        if on:
            if self._state == PowerSupplyState.IDLE:
                self._state = PowerSupplyState.ARMED
        else:
            self._setpoint_a = 0.0
            self._setpoint_w = None
            self._state = PowerSupplyState.IDLE

    def set_current_setpoint(self, value_a: float) -> float:
        """Apply a current setpoint (Amps).

        The value is clamped to [current_setpoint_min_a, current_setpoint_max_a]
        before being applied. Setpoint application is rejected in IDLE and
        TRIPPED states (returns clamped value but does not change internal
        setpoint).

        Precondition: value_a is finite numeric.
        Postcondition:
            - If state was ARMED / RUNNING: internal setpoint_a == clamped value;
              mode == CURRENT; setpoint_w == None
            - If clamped > 0 and state was ARMED: state -> RUNNING
            - Returns the clamped value that would be applied.

        Raises:
            TypeError: if value_a is not numeric.
            ValueError: if value_a is NaN or infinite.
        """
        if not isinstance(value_a, int | float) or isinstance(value_a, bool):
            raise TypeError(f"value_a must be numeric, got {type(value_a).__name__}")
        v = float(value_a)
        if not math.isfinite(v):
            raise ValueError(f"value_a must be finite, got {v}")

        clamped = float(
            max(
                self._config.current_setpoint_min_a,
                min(v, self._config.current_setpoint_max_a),
            )
        )

        if self._state in (PowerSupplyState.ARMED, PowerSupplyState.RUNNING):
            self._setpoint_a = clamped
            self._setpoint_w = None
            self._mode = PowerSupplyMode.CURRENT
            if self._state == PowerSupplyState.ARMED and clamped > 0.0:
                self._state = PowerSupplyState.RUNNING
        elif self._state == PowerSupplyState.IDLE:
            logger.warning(
                "current setpoint %.3f A ignored — controller in IDLE; "
                "enable permissive first",
                clamped,
            )
        elif self._state == PowerSupplyState.TRIPPED:
            logger.warning(
                "current setpoint %.3f A ignored — controller TRIPPED; "
                "acknowledge first",
                clamped,
            )
        return clamped

    def set_power_setpoint(self, value_w: float) -> float:
        """Apply a power setpoint (Watts).

        In POWER mode, the controller derives a target current from the
        most recent measured voltage: I_target = P_target / V_measured. The
        derived current is then clamped to the operator current bounds, and
        the resulting *achievable* power is returned.

        Precondition: value_w is finite numeric.
        Postcondition:
            - If state was ARMED / RUNNING and V_measured > 0: setpoint_w stored,
              mode == POWER, setpoint_a derived from V_measured
            - If state was ARMED and derived current > 0: state -> RUNNING
            - Returns the power that will actually be achieved given clamping.

        Raises:
            TypeError: if value_w is not numeric.
            ValueError: if value_w is NaN or infinite, or negative.
        """
        if not isinstance(value_w, int | float) or isinstance(value_w, bool):
            raise TypeError(f"value_w must be numeric, got {type(value_w).__name__}")
        w = float(value_w)
        if not math.isfinite(w):
            raise ValueError(f"value_w must be finite, got {w}")
        if w < 0.0:
            raise ValueError(f"value_w must be non-negative, got {w}")

        if self._last_v < 0.1:
            logger.warning(
                "power setpoint %.1f W ignored — measured voltage %.3f V too "
                "low for divide",
                w,
                self._last_v,
            )
            return 0.0

        i_raw = w / self._last_v
        clamped_i = float(
            max(
                self._config.current_setpoint_min_a,
                min(i_raw, self._config.current_setpoint_max_a),
            )
        )
        achievable_w = clamped_i * self._last_v

        if self._state in (PowerSupplyState.ARMED, PowerSupplyState.RUNNING):
            self._setpoint_a = clamped_i
            self._setpoint_w = w
            self._mode = PowerSupplyMode.POWER
            if self._state == PowerSupplyState.ARMED and clamped_i > 0.0:
                self._state = PowerSupplyState.RUNNING
        elif self._state == PowerSupplyState.IDLE:
            logger.warning(
                "power setpoint %.1f W ignored — controller IDLE; "
                "enable permissive first",
                w,
            )
        elif self._state == PowerSupplyState.TRIPPED:
            logger.warning(
                "power setpoint %.1f W ignored — controller TRIPPED; acknowledge first",
                w,
            )
        return achievable_w

    @staticmethod
    def _safe_finite(value: float) -> float:
        """Coerce a feedback input to a finite float; non-finite values map
        to 0 so a sensor failure can never accidentally trip the loop or
        smuggle a NaN through the math."""
        if not isinstance(value, int | float) or isinstance(value, bool):
            return 0.0
        v = float(value)
        return v if math.isfinite(v) else 0.0

    def _evaluate_trips(self, measured_power_w: float) -> None:
        """Latch trips based on the latest power and temperature; flips
        state to TRIPPED on first breach."""
        if measured_power_w > self._config.power_alarm_max_w:
            self._trips.add("HH_POWER")
        if self._last_t > self._config.temp_alarm_max_c:
            self._trips.add("HH_TEMP")
        if self._trips and self._state != PowerSupplyState.TRIPPED:
            logger.error(
                "trip latched: %s (P=%.1f W, T=%.1f C)",
                ",".join(sorted(self._trips)),
                measured_power_w,
                self._last_t,
            )
            self._state = PowerSupplyState.TRIPPED

    def _recompute_power_mode_setpoint(self) -> None:
        """In POWER mode, derive the current setpoint from the latest V."""
        if (
            self._mode != PowerSupplyMode.POWER
            or self._state != PowerSupplyState.RUNNING
            or self._setpoint_w is None
            or self._last_v < 0.1
        ):
            return
        i_raw = self._setpoint_w / self._last_v
        self._setpoint_a = max(
            self._config.current_setpoint_min_a,
            min(i_raw, self._config.current_setpoint_max_a),
        )

    def _should_force_output_zero(self) -> bool:
        """All three "kill the output now" conditions in one place."""
        return (
            self._state != PowerSupplyState.RUNNING
            or not self._permissive
            or bool(self._trips)
        )

    def _reset_slew_state(self) -> None:
        """Wipe slew tracker so the next RUNNING period starts at 0 % with
        a fresh dt baseline (no accumulated wall-clock catch-up)."""
        self._slewed_percent = 0.0
        self._last_commanded_percent = 0.0
        self._last_tick_monotonic = None

    def _apply_slew(self, target_percent: float, dt_s: float) -> float:
        """Slew-rate limit on increases only; decreases pass through."""
        if target_percent <= self._slewed_percent:
            self._slewed_percent = target_percent
        else:
            max_increase_pct = self._config.setpoint_ramp_rate_pct_per_s * dt_s
            self._slewed_percent = min(
                target_percent,
                self._slewed_percent + max_increase_pct,
            )
        return self._slewed_percent

    def tick(
        self,
        measured_current_a: float,
        measured_voltage_v: float,
        measured_temp_c: float,
        now: float | None = None,
    ) -> float:
        """Advance the controller one cycle and return the AO command percent.

        Reads feedback inputs, evaluates trip conditions (latching them on
        breach), recomputes current setpoint when in POWER mode, applies the
        slew-rate limiter on output increases, and returns the AO command
        percent (0..100) the caller should send to the PLC.

        Slew behavior (per `setpoint_ramp_rate_pct_per_s` in the config):
          - Increases are clamped to rate * dt_seconds.
          - Decreases are applied instantly (so a step-down in the operator
            setpoint takes effect on the very next tick).
          - Any path that forces output to zero (IDLE / ARMED / TRIPPED /
            permissive off) bypasses the ramp completely, so emergency
            shutdowns are always one tick.

        Args:
            measured_current_a: PS-side current feedback, engineering units.
            measured_voltage_v: PS-side voltage feedback, engineering units.
            measured_temp_c: HH-monitored thermocouple temperature in C.
            now: Optional monotonic timestamp in seconds. Production code
                leaves this as None so the controller reads
                time.monotonic() itself; tests inject explicit values so
                the slew behavior is deterministic.

        Precondition: all three measured inputs are finite floats. Non-finite
        inputs are treated as 0 (safe).
        Postcondition: returned percent is in [0, 100]; trips are latched if
        their conditions are met; internal slew tracker advanced.
        """
        self._last_i = self._safe_finite(measured_current_a)
        self._last_v = self._safe_finite(measured_voltage_v)
        self._last_t = self._safe_finite(measured_temp_c)

        measured_power_w = self._last_v * self._last_i
        self._evaluate_trips(measured_power_w)
        self._recompute_power_mode_setpoint()

        # dt for slew limiter — captured before the force-to-zero branch so
        # _last_tick_monotonic can be reset inside _reset_slew_state without
        # losing this tick's reading.
        tick_now = now if now is not None else time.monotonic()
        dt_s = (
            0.0
            if self._last_tick_monotonic is None
            else max(0.0, tick_now - self._last_tick_monotonic)
        )
        self._last_tick_monotonic = tick_now

        if self._should_force_output_zero():
            self._reset_slew_state()
            self._output_clamped = False
            return 0.0

        raw_percent = 100.0 * self._setpoint_a / self._config.current_full_scale_a
        raw_percent = max(0.0, min(raw_percent, 100.0))

        # Operator safety clamp: hard-cap the commanded output regardless of how
        # the setpoint scales. Applied before the slew limiter so the ramp
        # settles at the clamp instead of overshooting it.
        clamp_percent = self._config.output_clamp_percent
        target_percent = min(raw_percent, clamp_percent)
        self._output_clamped = raw_percent > clamp_percent

        commanded = self._apply_slew(target_percent, dt_s)
        self._last_commanded_percent = commanded
        return commanded

    def acknowledge_trip(self) -> bool:
        """Clear latched trips and return controller to safe idle/armed state.

        Postcondition:
            - All trips cleared.
            - Setpoint cleared to 0.
            - State -> ARMED if permissive is True, else IDLE.

        Returns:
            True if a trip was cleared. False if there were no trips.
        """
        if self._state != PowerSupplyState.TRIPPED:
            return False
        self._trips.clear()
        self._setpoint_a = 0.0
        self._setpoint_w = None
        self._state = (
            PowerSupplyState.ARMED if self._permissive else PowerSupplyState.IDLE
        )
        return True

    def status(self) -> PowerSupplyStatus:
        """Return a snapshot of controller state for serialization."""
        return PowerSupplyStatus(
            state=self._state,
            mode=self._mode,
            permissive=self._permissive,
            setpoint_a=self._setpoint_a,
            setpoint_w=self._setpoint_w,
            measured_current_a=self._last_i,
            measured_voltage_v=self._last_v,
            measured_power_w=self._last_v * self._last_i,
            measured_temp_c=self._last_t,
            commanded_output_percent=self._last_commanded_percent,
            trips=sorted(self._trips),
            output_clamp_percent=self._config.output_clamp_percent,
            output_clamped=self._output_clamped,
        )
