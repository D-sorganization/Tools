"""Typed water saturation-pressure contract shared by REST and MCP."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.python.sidekick.process_calculators.constants import (
    IAPWS_MAX_TEMPERATURE_C,
    IAPWS_MIN_TEMPERATURE_C,
)

# Empirical fits are only characterised on liquid water up to 100 degC.
EMPIRICAL_MAX_TEMPERATURE_C = 100.0


class WaterVaporPressureRequest(BaseModel):
    """Saturation-pressure inputs bounded by the IAPWS-IF97 saturation line.

    The temperature envelope is the triple point (0.01 degC) through the
    critical point (373.946 degC). 0.0 degC and 374.0 degC are outside that
    domain. Buck and Magnus remain limited to 100 degC inside the envelope.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature_c: float = Field(
        ge=IAPWS_MIN_TEMPERATURE_C,
        le=IAPWS_MAX_TEMPERATURE_C,
        description="Water temperature [degC]",
    )
    method: Literal["iapws", "buck", "magnus"] = Field(
        default="iapws",
        description=(
            "Correlation: iapws (0.01-373.946 degC), buck or magnus "
            "(0.01-100 degC; magnus drifts ~2.6% high at 100 degC)"
        ),
    )

    @model_validator(mode="after")
    def _empirical_fits_stay_in_range(self) -> "WaterVaporPressureRequest":
        if self.method != "iapws" and self.temperature_c > EMPIRICAL_MAX_TEMPERATURE_C:
            raise ValueError(
                f"{self.method} is valid only up to "
                f"{EMPIRICAL_MAX_TEMPERATURE_C:g} degC; use iapws"
            )
        return self


class WaterVaporPressureResponse(BaseModel):
    """Finite non-negative saturation pressure, not a design approval."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pressure_pa: float = Field(ge=0, description="Saturation pressure [Pa]")
    method_used: str = Field(description="Correlation that produced the value")
