"""Typed water saturation-pressure contract shared by REST and MCP."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Empirical fits are only characterised on liquid water between 0 and 100 degC.
EMPIRICAL_MAX_TEMPERATURE_C = 100.0


class WaterVaporPressureRequest(BaseModel):
    """Saturation-pressure inputs; IAPWS-IF97 covers the full liquid range."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature_c: float = Field(ge=0, le=374, description="Water temperature [degC]")
    method: Literal["iapws", "buck", "magnus"] = Field(
        default="iapws",
        description=(
            "Correlation: iapws (0-374 degC), buck or magnus (0-100 degC; "
            "magnus drifts ~2.6% high at 100 degC)"
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
