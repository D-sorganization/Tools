"""Pydantic data models + enums for the temperature subsystem.

Split out of `temperature_controller.py` so the controller module stays clean
and these models stay plain data: no business logic lives here, so this module
is safely importable from anywhere (tests, FastAPI router, controller) without
circular-import risk. Mirrors the structure of `power_supply_models.py`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.python.compatibility import StrEnum


class TemperatureState(StrEnum):
    """Controller state machine state."""

    IDLE = "idle"
    ARMED = "armed"
    RUNNING = "running"
    TRIPPED = "tripped"


class TemperatureConfig(BaseModel):
    """Operator-configurable parameters for the temperature controller.

    All numeric fields are validated by Pydantic at construction (Pydantic's
    Field constraints raise ValidationError on bad input). The
    `_check_invariants` validator enforces the cross-field invariants:
        setpoint_min_c < setpoint_max_c
        setpoint_max_c <= temp_full_scale_c
        hh_limit_c <= temp_full_scale_c
    """

    temp_tag: str = Field(
        default="TAG_0",
        description="Modbus tag carrying the controlled thermocouple (TC0).",
    )

    temp_full_scale_c: float = Field(
        default=1400.0,
        gt=0.0,
        description="deg C at 100% of the thermocouple tag",
    )

    setpoint_min_c: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower clamp for operator temperature setpoint (deg C).",
    )
    setpoint_max_c: float = Field(
        default=1400.0,
        gt=0.0,
        description="Upper clamp for operator temperature setpoint (deg C).",
    )

    deadband_c: float = Field(
        default=5.0,
        gt=0.0,
        description="on/off hysteresis half-band around setpoint",
    )
    hh_limit_c: float = Field(
        default=1400.0,
        gt=0.0,
        description="high-high cutoff; heater is latched off at/above this",
    )

    heater_label: str = Field(
        default="Heater",
        min_length=1,
        max_length=40,
        description="HMI name for the heater on/off relay output.",
    )

    @field_validator("heater_label")
    @classmethod
    def _strip_label(cls, value: str) -> str:
        """Trim labels and reject whitespace-only names (DbC)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("signal label must not be blank")
        return trimmed

    @model_validator(mode="after")
    def _check_invariants(self) -> TemperatureConfig:
        if self.setpoint_min_c >= self.setpoint_max_c:
            raise ValueError(
                f"setpoint_min_c ({self.setpoint_min_c}) must be less than "
                f"setpoint_max_c ({self.setpoint_max_c})"
            )
        if self.setpoint_max_c > self.temp_full_scale_c:
            raise ValueError(
                f"setpoint_max_c ({self.setpoint_max_c}) must not exceed "
                f"temp_full_scale_c ({self.temp_full_scale_c})"
            )
        if self.hh_limit_c > self.temp_full_scale_c:
            raise ValueError(
                f"hh_limit_c ({self.hh_limit_c}) must not exceed "
                f"temp_full_scale_c ({self.temp_full_scale_c})"
            )
        return self


class TemperatureStatus(BaseModel):
    """Snapshot of controller state for the UI / WebSocket stream."""

    state: TemperatureState
    permissive: bool

    setpoint_c: float
    measured_temp_c: float

    relay_on: bool
    trips: list[str]

    hh_limit_c: float
    deadband_c: float

    estopped: bool = False
