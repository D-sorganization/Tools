"""Pydantic data models + enums for the power-supply subsystem.

Split out of `power_supply.py` so the controller module stays under the
per-file LOC budget. Kept as plain data: no business logic lives here so
this module is safely importable from anywhere (tests, FastAPI router,
controller) without circular-import risk.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class StrEnum(str, Enum):  # noqa: UP042
    """Compat shim for Python < 3.11 environments that don't ship StrEnum.

    Inherits from `str` so the enum members serialize as plain strings when
    passed through Pydantic / FastAPI / JSON.
    """


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
    setpoint_ramp_rate_pct_per_s: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Maximum rate at which the AO command percent may INCREASE, "
            "expressed as percent of full output per second. Decreases are "
            "applied instantly so the operator (or a trip) can always pull "
            "the loop down to zero immediately. Default 5 %/s gives a slow-"
            "start ramp (0 % -> 100 % takes 20 s)."
        ),
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
