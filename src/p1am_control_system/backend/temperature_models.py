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


class TcPath(StrEnum):
    """Acquisition path a thermocouple's reading arrives through.

    Each physical thermocouple (K and R) can be read two ways:

    - ``TC_CARD``: wired straight into the P1-04THM thermocouple module, which
      linearizes the junction on-module (the original bench path).
    - ``ANALOG``: routed through an external signal conditioner that emits a
      4-20 mA loop into the P1-4ADL2DAL analog input card. Added to reject the
      electrical noise the bare Type-R sleeve picks up at high temperature.

    Path is orthogonal to :class:`TcType`, so the operator selects one of four
    sources (K/R x card/analog) exactly the way K vs R was selected before.
    ``TC_CARD`` is the default so existing configs keep their behavior.
    """

    TC_CARD = "thm"
    ANALOG = "analog"


def _stripped_label(value: str) -> str:
    """Trim a label and reject whitespace-only names (DbC helper, DRY)."""
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("label must not be blank")
    return trimmed


class ThermocoupleChannel(BaseModel):
    """One thermocouple input: the tag carrying it, its range, and a label.

    The firmware publishes every channel as 0-100 % of its electrical span, and
    the backend maps that percent AFFINELY onto ``[range_min_c, full_scale_c]``:
    ``deg C = range_min_c + pct/100 * (full_scale_c - range_min_c)``.

    - **TC-card channels** (P1-04THM) are linearized on-module and reported as a
      simple 0-100 % of a full scale, so ``range_min_c`` stays 0 (0 % -> 0 C).
    - **Analog (conditioner) channels** map the conditioner's 4-20 mA loop, whose
      4 mA and 20 mA endpoints are NOT 0 and full scale. E.g. an FC-T1 Type-K
      spans -150..1372 C over 4-20 mA, so ``range_min_c = -150`` and
      ``full_scale_c = 1372``; 4 mA (0 %) reads -150 C, 20 mA (100 %) reads 1372 C.
    """

    tag: str = Field(
        default="TAG_0",
        description="Modbus tag carrying this thermocouple's scaled reading.",
    )
    range_min_c: float = Field(
        default=0.0,
        description="deg C at 0 % of the tag (4 mA for a 4-20 mA conditioner).",
    )
    full_scale_c: float = Field(
        default=1400.0,
        gt=0.0,
        description="deg C at 100 % of the tag (20 mA for a 4-20 mA conditioner).",
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

    @model_validator(mode="after")
    def _check_span(self) -> ThermocoupleChannel:
        if self.full_scale_c <= self.range_min_c:
            raise ValueError(
                f"full_scale_c ({self.full_scale_c}) must exceed range_min_c "
                f"({self.range_min_c}) — the 0->100 % span must be positive"
            )
        return self

    def scale_percent(self, pct: float) -> float:
        """Map a 0-100 % tag reading to deg C over this channel's range (DRY).

        Single source of truth for percent->deg C, so every reader (control law,
        HMI display, cross-check) converts identically regardless of channel.
        """
        return self.range_min_c + (pct / 100.0) * (self.full_scale_c - self.range_min_c)


def _default_type_k() -> ThermocoupleChannel:
    # P1-04THM channel 1 -> TAG_0.
    return ThermocoupleChannel(tag="TAG_0", full_scale_c=1400.0, label="Type K (Ch 1)")


def _default_type_r() -> ThermocoupleChannel:
    # P1-04THM channel 2 -> TAG_1.
    return ThermocoupleChannel(tag="TAG_1", full_scale_c=1400.0, label="Type R (Ch 2)")


def _default_analog_k() -> ThermocoupleChannel:
    # Signal-conditioned type-K on P1-4ADL2DAL AI2 (4-20 mA) -> TAG_14.
    # AutomationDirect FC-T1 Type-K factory range: -150..1372 C over 4-20 mA.
    return ThermocoupleChannel(
        tag="TAG_14", range_min_c=-150.0, full_scale_c=1372.0, label="Type K (Analog)"
    )


def _default_analog_r() -> ThermocoupleChannel:
    # Signal-conditioned type-R on P1-4ADL2DAL AI3 (4-20 mA) -> TAG_15.
    # AutomationDirect FC-T1 Type-R factory range: 65..1768 C over 4-20 mA.
    return ThermocoupleChannel(
        tag="TAG_15", range_min_c=65.0, full_scale_c=1768.0, label="Type R (Analog)"
    )


class TemperatureConfig(BaseModel):
    """Operator-configurable parameters for the temperature controller.

    Four thermocouple channels are configured — type K and type R, each on two
    acquisition paths (the P1-04THM card and an analog signal conditioner). The
    pair (`active_tc_type`, `active_tc_path`) selects which one drives the
    controller. `temp_tag` / `temp_full_scale_c` are derived (read-only) from the
    active channel, so the rest of the system reads the controlled thermocouple
    through one stable accessor regardless of which source is selected (DRY/LOD).

    Cross-field invariants (checked against the *active* channel's full scale):
        setpoint_min_c < setpoint_max_c
        setpoint_max_c <= temp_full_scale_c
        hh_limit_c <= temp_full_scale_c
    """

    type_k: ThermocoupleChannel = Field(
        default_factory=_default_type_k,
        description="Type-K thermocouple on the P1-04THM card (TC-card path).",
    )
    type_r: ThermocoupleChannel = Field(
        default_factory=_default_type_r,
        description="Type-R thermocouple on the P1-04THM card (TC-card path).",
    )
    analog_k: ThermocoupleChannel = Field(
        default_factory=_default_analog_k,
        description="Signal-conditioned type-K on an analog input (analog path).",
    )
    analog_r: ThermocoupleChannel = Field(
        default_factory=_default_analog_r,
        description="Signal-conditioned type-R on an analog input (analog path).",
    )
    active_tc_type: TcType = Field(
        default=TcType.TYPE_K,
        description="Which thermocouple type (K or R) the controller reads.",
    )
    active_tc_path: TcPath = Field(
        default=TcPath.TC_CARD,
        description="Which acquisition path (TC card or analog conditioner).",
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

    def channel_for(self, tc_type: TcType, tc_path: TcPath) -> ThermocoupleChannel:
        """Return the channel for a (type, path) pair — single source of truth.

        Every "which tag/full-scale/label for this source" decision in the whole
        system routes through here (DRY), so the 2x2 source matrix has exactly
        one mapping. Used by ``active_channel`` and by the integration layer to
        find both the controlling channel and its same-path cross-check partner.
        """
        if tc_path == TcPath.ANALOG:
            return self.analog_r if tc_type == TcType.TYPE_R else self.analog_k
        return self.type_r if tc_type == TcType.TYPE_R else self.type_k

    @property
    def active_channel(self) -> ThermocoupleChannel:
        """The thermocouple channel the controller currently reads."""
        return self.channel_for(self.active_tc_type, self.active_tc_path)

    @property
    def cross_check_channel(self) -> ThermocoupleChannel:
        """The OTHER physical sensor on the active path (HH / cross-check partner).

        The complement of :attr:`active_channel`: same acquisition path, opposite
        thermocouple type. This is the reading the controller uses as its
        HH-on-either backstop and stuck-sensor cross-check, so both checks compare
        the two physical sensors seen through the SAME path the operator selected.
        Written out (rather than ``channel_for`` of a computed "other" type) so no
        thermocouple-type literal is passed as an argument — keeps it clean under
        the CI type checker's skipped-import mode.
        """
        if self.active_tc_path == TcPath.ANALOG:
            return (
                self.analog_k if self.active_tc_type == TcType.TYPE_R else self.analog_r
            )
        return self.type_k if self.active_tc_type == TcType.TYPE_R else self.type_r

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
    active_tc_path: TcPath = Field(
        default=TcPath.TC_CARD,
        description="Acquisition path (TC card or analog conditioner) in use.",
    )
    active_tc_label: str = "Type K"

    type_k_temp_c: float | None = Field(
        default=None,
        description=(
            "Latest type-K reading (deg C) on the ACTIVE acquisition path, "
            "regardless of which TC is controlling, so the HMI can display and "
            "plot both channels at once. On the TC-card path this is the P1-04THM "
            "K junction; on the analog path it is the conditioned K loop. None "
            "when no reading is available yet."
        ),
    )
    type_r_temp_c: float | None = Field(
        default=None,
        description=(
            "Latest type-R reading (deg C) on the ACTIVE acquisition path, "
            "regardless of which TC is controlling, so the HMI can display and "
            "plot both channels at once. None when no reading is available yet."
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
    burnout_high_side: bool = Field(
        default=True,
        description=(
            "P1-04THM open-circuit (burnout) fail direction. True = HIGH-side (an "
            "open thermocouple reads full scale, so the heater shuts off — "
            "fail-safe); False = LOW-side (an open reads 0 C, which looks cold). "
            "Operator-selectable from the HMI."
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
