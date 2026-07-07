"""Pydantic data models + enums for the temperature subsystem.

Split out of `temperature_controller.py` so the controller module stays clean
and these models stay plain data: no business logic lives here, so this module
is safely importable from anywhere (tests, FastAPI router, controller) without
circular-import risk. Mirrors the structure of `power_supply_models.py`.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from shared.python.compatibility import StrEnum


class TemperatureState(StrEnum):
    """Controller state machine state."""

    IDLE = "idle"
    ARMED = "armed"
    RUNNING = "running"
    TRIPPED = "tripped"


class TcType(StrEnum):
    """Thermocouple type the heater controller reads from.

    The bench has a type-K and a type-R thermocouple wired to separate channels;
    the operator picks which one drives the heater for a given experiment.
    """

    TYPE_K = "K"
    TYPE_R = "R"


def _stripped_label(value: str) -> str:
    """Trim a label and reject whitespace-only names (DbC helper, DRY)."""
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("label must not be blank")
    return trimmed


class ThermocoupleChannel(BaseModel):
    """One thermocouple input: the tag carrying it, its full scale, and a label.

    The firmware scales every TC channel 0-100 % over the *same* full scale
    (default 1400 C), and the P1-04THM does the per-type linearization on-module,
    so K and R differ only by which tag they land on — not by how the backend
    converts percent to deg C.
    """

    tag: str = Field(
        default="TAG_0",
        description="Modbus tag carrying this thermocouple's scaled reading.",
    )
    full_scale_c: float = Field(
        default=1400.0,
        gt=0.0,
        description="deg C at 100 % of the tag (must match the firmware scaling).",
    )
    label: str = Field(
        default="Thermocouple",
        min_length=1,
        max_length=40,
        description="HMI name for this thermocouple.",
    )

    @field_validator("label")
    @classmethod
    def _strip_label(cls, value: str) -> str:
        return _stripped_label(value)


def _default_type_k() -> ThermocoupleChannel:
    # P1-04THM channel 1 -> TAG_0.
    return ThermocoupleChannel(tag="TAG_0", full_scale_c=1400.0, label="Type K (Ch 1)")


def _default_type_r() -> ThermocoupleChannel:
    # P1-04THM channel 2 -> TAG_1.
    return ThermocoupleChannel(tag="TAG_1", full_scale_c=1400.0, label="Type R (Ch 2)")


class TemperatureConfig(BaseModel):
    """Operator-configurable parameters for the temperature controller.

    Two thermocouple channels (type K and type R) are configured; `active_tc_type`
    selects which one drives the controller. `temp_tag` / `temp_full_scale_c` are
    derived (read-only) from the active channel, so the rest of the system reads
    the controlled thermocouple through one stable accessor regardless of which
    type is selected (DRY / LOD).

    Cross-field invariants (checked against the *active* channel's full scale):
        setpoint_min_c < setpoint_max_c
        setpoint_max_c <= temp_full_scale_c
        hh_limit_c <= temp_full_scale_c
    """

    type_k: ThermocoupleChannel = Field(
        default_factory=_default_type_k,
        description="Type-K thermocouple channel.",
    )
    type_r: ThermocoupleChannel = Field(
        default_factory=_default_type_r,
        description="Type-R thermocouple channel.",
    )
    active_tc_type: TcType = Field(
        default=TcType.TYPE_K,
        description="Which thermocouple the controller currently reads.",
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
        description=(
            "on/off hysteresis half-band around setpoint (deg C); the relay "
            "switches ON at setpoint-deadband and OFF at setpoint+deadband, so "
            "the full control band is 2x this. Smaller = tighter regulation."
        ),
    )
    min_on_time_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "minimum seconds the relay must stay ON before it may switch OFF "
            "(anti-short-cycle; 0 disables). Caps how often the heater cycles."
        ),
    )
    min_off_time_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "minimum seconds the relay must stay OFF before it may switch ON "
            "(anti-short-cycle; 0 disables). Caps how often the heater cycles."
        ),
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
    def _strip_heater_label(cls, value: str) -> str:
        return _stripped_label(value)

    @property
    def active_channel(self) -> ThermocoupleChannel:
        """The thermocouple channel the controller currently reads."""
        return self.type_r if self.active_tc_type == TcType.TYPE_R else self.type_k

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_tag(self) -> str:
        """Tag of the active thermocouple (what the controller reads)."""
        return self.active_channel.tag

    @computed_field  # type: ignore[prop-decorator]
    @property
    def temp_full_scale_c(self) -> float:
        """Full scale (deg C) of the active thermocouple."""
        return self.active_channel.full_scale_c

    @computed_field  # type: ignore[prop-decorator]
    @property
    def active_tc_label(self) -> str:
        """HMI label of the active thermocouple."""
        return self.active_channel.label

    @model_validator(mode="after")
    def _check_invariants(self) -> TemperatureConfig:
        if self.setpoint_min_c >= self.setpoint_max_c:
            raise ValueError(
                f"setpoint_min_c ({self.setpoint_min_c}) must be less than "
                f"setpoint_max_c ({self.setpoint_max_c})"
            )
        full_scale = self.temp_full_scale_c
        if self.setpoint_max_c > full_scale:
            raise ValueError(
                f"setpoint_max_c ({self.setpoint_max_c}) must not exceed the "
                f"active thermocouple full scale ({full_scale})"
            )
        if self.hh_limit_c > full_scale:
            raise ValueError(
                f"hh_limit_c ({self.hh_limit_c}) must not exceed the active "
                f"thermocouple full scale ({full_scale})"
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
    min_on_time_s: float = 0.0
    min_off_time_s: float = 0.0

    active_tc_type: TcType = Field(default=TcType.TYPE_K)
    active_tc_label: str = "Type K"

    type_k_temp_c: float | None = Field(
        default=None,
        description=(
            "Latest type-K thermocouple reading (deg C), regardless of which TC "
            "is controlling, so the HMI can display and plot both channels at "
            "once. None when no reading is available yet."
        ),
    )
    type_r_temp_c: float | None = Field(
        default=None,
        description=(
            "Latest type-R thermocouple reading (deg C), regardless of which TC "
            "is controlling, so the HMI can display and plot both channels at "
            "once. None when no reading is available yet."
        ),
    )
    control_sensor_holding: bool = Field(
        default=False,
        description=(
            "True when the controlling thermocouple's reading is currently being "
            "held by the deglitch filter (a live dropout is being ridden out). "
            "Lets the HMI warn that the control sensor is intermittently faulting; "
            "a sustained fault escalates to a latched TC_FAULT trip."
        ),
    )

    estopped: bool = False

    last_setpoint_c: float | None = Field(
        default=None,
        description=(
            "Last operator setpoint (deg C) recalled from persisted settings, "
            "used by the HMI to pre-fill the target after a restart. None when "
            "nothing has been recalled. Recalling it never arms/energizes the "
            "heater — the controller stays IDLE until the operator presses Start."
        ),
    )
