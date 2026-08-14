"""Pydantic data models + enums for the power-supply subsystem.

Split out of `power_supply.py` so the controller module stays under the
per-file LOC budget. Kept as plain data: no business logic lives here so
this module is safely importable from anywhere (tests, FastAPI router,
controller) without circular-import risk.
"""

from __future__ import annotations

import hardware
from pydantic import BaseModel, Field, field_validator, model_validator
from signal_stats import NoiseMetric, NoiseStats, compute_noise

from shared.python.compatibility import StrEnum


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
    temp_full_scale_c: float = Field(
        default=hardware.THERMOCOUPLE_FULL_SCALE_C,
        gt=0.0,
        description=(
            "deg C at 100 % of temp_tag. The firmware publishes thermocouples "
            "as percent of full scale, so this is what converts the tag to the "
            "degC domain temp_alarm_max_c is expressed in. Defaults to the "
            "firmware contract value; must match it."
        ),
    )

    # Operator-facing signal names (single source of truth for the HMI labels —
    # trend legend, telemetry readouts, and wiring guide all read these). Kept
    # 1..40 chars so a blank or runaway label can't slip through.
    command_label: str = Field(
        default="Current command",
        min_length=1,
        max_length=40,
        description="HMI name for the current-command AO (AO0).",
    )
    aux_command_label: str = Field(
        default="Aux command",
        min_length=1,
        max_length=40,
        description="HMI name for the spare AO (AO1).",
    )
    current_feedback_label: str = Field(
        default="Current",
        min_length=1,
        max_length=40,
        description="HMI name for the current-feedback AI (AI0).",
    )
    voltage_feedback_label: str = Field(
        default="Voltage",
        min_length=1,
        max_length=40,
        description="HMI name for the voltage-feedback AI (AI1).",
    )
    temp_label: str = Field(
        default="Temperature",
        min_length=1,
        max_length=40,
        description="HMI name for the temperature thermocouple (TC0).",
    )

    current_full_scale_a: float = Field(
        default=200.0,
        gt=0.0,
        description=(
            "Calibration: amps that correspond to a full-scale signal "
            "(100 % = 20 mA = 5 V) on the current command AND current-monitor "
            "legs. Set this to what the supply's meter reads at full output. "
            "Default 200 A (this bench supply's max); adjustable in Calibration."
        ),
    )
    voltage_full_scale_v: float = Field(
        default=300.0,
        gt=0.0,
        description=(
            "Calibration: volts that correspond to a full-scale signal "
            "(100 % = 20 mA = 5 V) on the voltage-monitor AI. Set this to what "
            "the supply's voltmeter reads at full scale. Default 300 V (this "
            "bench supply's max); adjustable in Calibration."
        ),
    )

    current_setpoint_min_a: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower clamp for operator current setpoint (A).",
    )
    current_setpoint_max_a: float = Field(
        default=200.0,
        gt=0.0,
        description=(
            "Upper clamp for operator current setpoint (A). Default 200 A to "
            "match the supply's full scale; the output clamp (% of full) is the "
            "live safety guard. Adjustable."
        ),
    )

    power_alarm_max_w: float = Field(
        default=60000.0,
        gt=0.0,
        description=(
            "Power trip threshold in watts (HH_POWER). Default 60 kW = 200 A x "
            "300 V (the supply's max) so it guards against a genuine over-power "
            "fault without nuisance-tripping. Adjustable."
        ),
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
    output_clamp_percent: float = Field(
        default=20.0,
        gt=0.0,
        le=100.0,
        description=(
            "Hard upper clamp on the commanded AO output, as a percent of full "
            "output (0-100]. The controller never commands above this even when "
            "the current setpoint would scale higher — the command is capped, "
            "not the setpoint. This is the operator's safety limit for live-"
            "current testing; default 20 %. Decreasing it takes effect on the "
            "next tick (the slew limiter passes decreases through instantly)."
        ),
    )

    # ---- Arc / signal-noise detection (DC arc shows as AC noise on the feedback) -
    noise_window: int = Field(
        default=100,
        ge=2,
        le=10_000,
        description=(
            "Number of most-recent feedback samples used to quantify signal "
            "noise. At a 10 Hz scan, 100 samples ≈ a 10 s window. Bigger = "
            "smoother/slower; smaller = twitchier/faster arc response."
        ),
    )
    noise_metric: NoiseMetric = Field(
        default=NoiseMetric.STD,
        description=(
            "Which noise metric the arc thresholds are compared against: 'std' "
            "(sample std-dev, engineering units), 'peak_to_peak', 'rms' (AC RMS "
            "about the mean), or 'cv' (dimensionless std/|mean| ratio)."
        ),
    )
    current_arc_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Arc-detect threshold for the CURRENT feedback noise, in the units "
            "of the selected metric (A for std/p2p/rms, ratio for cv). Arcing is "
            "flagged when the metric exceeds this. None disables current-arc "
            "detection. Tune by watching the live noise readout on a steady arc."
        ),
    )
    voltage_arc_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Arc-detect threshold for the VOLTAGE feedback noise (units as per "
            "the selected metric). None disables voltage-arc detection."
        ),
    )

    @field_validator(
        "command_label",
        "aux_command_label",
        "current_feedback_label",
        "voltage_feedback_label",
        "temp_label",
    )
    @classmethod
    def _strip_label(cls, value: str) -> str:
        """Trim labels and reject whitespace-only names (DbC, DRY across fields)."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("signal label must not be blank")
        return trimmed

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


class PowerSupplyLastSetpoint(BaseModel):
    """The operator's last commanded setpoint, persisted for HMI pre-fill.

    Stored durably on every operator setpoint change and recalled on boot so the
    HMI can show the value the operator last dialed in. It is a *settings* record
    only: recalling it never arms or energizes the supply (the controller stays
    IDLE after restore).
    """

    mode: PowerSupplyMode
    value_a: float | None = Field(
        default=None,
        description="Last current setpoint in amps (set when mode == CURRENT).",
    )
    value_w: float | None = Field(
        default=None,
        description="Last power setpoint in watts (set when mode == POWER).",
    )


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

    output_clamp_percent: float = Field(
        default=20.0,
        description="Active hard upper clamp on commanded output (% of full).",
    )
    output_clamped: bool = Field(
        default=False,
        description=(
            "True when the output clamp is actively limiting the command — i.e. "
            "the current setpoint would otherwise drive the AO above "
            "output_clamp_percent. Lets the UI flag that the operator's limit "
            "is in effect."
        ),
    )
    effective_max_current_a: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "The most current the supply can actually deliver right now given "
            "the output limit: output_clamp_percent/100 * current_full_scale_a. "
            "The setpoint band may go higher, but the clamp caps real output "
            "here — the UI shows this so the two limits aren't confusing."
        ),
    )

    current_noise: NoiseStats = Field(
        default_factory=lambda: compute_noise([]),
        description="Rolling noise/variability stats for the current feedback.",
    )
    voltage_noise: NoiseStats = Field(
        default_factory=lambda: compute_noise([]),
        description="Rolling noise/variability stats for the voltage feedback.",
    )
    arcing: bool = Field(
        default=False,
        description=(
            "True when either feedback channel's noise metric exceeds its arc "
            "threshold — the operator-facing 'system may be arcing' indicator."
        ),
    )

    last_setpoint: PowerSupplyLastSetpoint | None = Field(
        default=None,
        description=(
            "Operator's last commanded setpoint recalled from durable storage on "
            "boot. Populated by the service so the HMI can pre-fill the setpoint "
            "field after a restart; it is purely informational and does NOT arm "
            "or energize the output — the controller stays IDLE until the "
            "operator re-enables permissive. None when nothing was persisted."
        ),
    )
