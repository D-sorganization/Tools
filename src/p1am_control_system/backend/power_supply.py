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
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class StrEnum(str, Enum):  # noqa: UP042
    pass


logger = logging.getLogger("dcs_backend.power_supply")


class PowerSupplyMode(StrEnum):
    """Operator-selected control mode."""

    CURRENT = "current"
    POWER = "power"


class PowerSupplyState(StrEnum):
    """Controller state machine state."""

    IDLE = "idle"
    ARMED = "armed"
    RUNNING = "running"
    TRIPPED = "tripped"


class PowerSupplyConfig(BaseModel):
    """Operator-configurable parameters for the power supply controller.

    All numeric fields are validated by Pydantic at construction (Pydantic's
    Field constraints raise ValidationError on bad input). The
    `_check_invariants` validator enforces the cross-field invariants:
        current_setpoint_min_a < current_setpoint_max_a
        current_setpoint_max_a <= current_full_scale_a
    """

    command_tag: str = Field(
        default="TAG_10",
        description="Modbus tag that drives the current-command AO (AO1).",
    )
    current_feedback_tag: str = Field(
        default="TAG_12",
        description="Modbus tag carrying measured I_out from the PS (AI1).",
    )
    voltage_feedback_tag: str = Field(
        default="TAG_13",
        description="Modbus tag carrying measured V_out from the PS (AI2).",
    )
    temp_tag: str = Field(
        default="TAG_0",
        description="Modbus tag carrying the HH-monitored thermocouple (TC1).",
    )

    current_full_scale_a: float = Field(
        default=100.0,
        gt=0.0,
        description="Amps at 100 % AO command (5 V on PS input after conditioner).",
    )
    voltage_full_scale_v: float = Field(
        default=50.0,
        gt=0.0,
        description="Volts at 100 % AI reading from PS V feedback.",
    )

    current_setpoint_min_a: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower clamp for operator current setpoint (A).",
    )
    current_setpoint_max_a: float = Field(
        default=50.0,
        gt=0.0,
        description="Upper clamp for operator current setpoint (A).",
    )

    power_alarm_max_w: float = Field(
        default=1000.0,
        gt=0.0,
        description="Power trip threshold in watts (HH_POWER).",
    )
    temp_alarm_max_c: float = Field(
        default=1200.0,
        gt=0.0,
        description="Temperature trip threshold in degrees Celsius (HH_TEMP).",
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> PowerSupplyConfig:
        if self.current_setpoint_min_a >= self.current_setpoint_max_a:
            raise ValueError(
                "current_setpoint_min_a "
                f"({self.current_setpoint_min_a}) must be less than "
                f"current_setpoint_max_a ({self.current_setpoint_max_a})"
            )
        if self.current_setpoint_max_a > self.current_full_scale_a:
            raise ValueError(
                f"current_setpoint_max_a ({self.current_setpoint_max_a}) must not "
                f"exceed current_full_scale_a ({self.current_full_scale_a})"
            )
        return self


class PowerSupplyStatus(BaseModel):
    """Snapshot of controller state for the UI / WebSocket stream."""

    state: PowerSupplyState
    mode: PowerSupplyMode
    permissive: bool

    setpoint_a: float
    setpoint_w: float | None

    measured_current_a: float
    measured_voltage_v: float
    measured_power_w: float
    measured_temp_c: float

    commanded_output_percent: float
    trips: list[str]


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
          [current_setpoint_min_a, current_setpoint_max_a] then scaled to
          a percent of current_full_scale_a.
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

        clamped = max(
            self._config.current_setpoint_min_a,
            min(v, self._config.current_setpoint_max_a),
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
        clamped_i = max(
            self._config.current_setpoint_min_a,
            min(i_raw, self._config.current_setpoint_max_a),
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

    def tick(
        self,
        measured_current_a: float,
        measured_voltage_v: float,
        measured_temp_c: float,
    ) -> float:
        """Advance the controller one cycle and return the AO command percent.

        Reads feedback inputs, evaluates trip conditions (latching them on
        breach), recomputes current setpoint when in POWER mode, and returns
        the AO command percent (0..100) the caller should send to the PLC.

        Precondition: all three measured inputs are finite floats. Non-finite
        inputs are treated as 0 (safe).
        Postcondition: returned percent is in [0, 100]; trips are latched if
        their conditions are met.
        """
        self._last_i = (
            float(measured_current_a)
            if isinstance(measured_current_a, int | float)
            and math.isfinite(float(measured_current_a))
            else 0.0
        )
        self._last_v = (
            float(measured_voltage_v)
            if isinstance(measured_voltage_v, int | float)
            and math.isfinite(float(measured_voltage_v))
            else 0.0
        )
        self._last_t = (
            float(measured_temp_c)
            if isinstance(measured_temp_c, int | float)
            and math.isfinite(float(measured_temp_c))
            else 0.0
        )

        measured_power_w = self._last_v * self._last_i

        # Trip evaluation (latching)
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

        # In POWER mode, recompute current target from latest V
        if (
            self._mode == PowerSupplyMode.POWER
            and self._state == PowerSupplyState.RUNNING
            and self._setpoint_w is not None
            and self._last_v >= 0.1
        ):
            i_raw = self._setpoint_w / self._last_v
            self._setpoint_a = max(
                self._config.current_setpoint_min_a,
                min(i_raw, self._config.current_setpoint_max_a),
            )

        if (
            self._state != PowerSupplyState.RUNNING
            or not self._permissive
            or self._trips
        ):
            self._last_commanded_percent = 0.0
            return 0.0

        percent = 100.0 * self._setpoint_a / self._config.current_full_scale_a
        clamped_percent = max(0.0, min(percent, 100.0))
        self._last_commanded_percent = clamped_percent
        return clamped_percent

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
        )
