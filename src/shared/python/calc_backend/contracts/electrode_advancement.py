"""Typed electrode consumption contract shared by REST and MCP."""

from pydantic import BaseModel, ConfigDict, Field


class ElectrodeAdvancementRequest(BaseModel):
    """Constant-current consumption inputs; rate must be calibrated by the caller."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    consumption_rate_in_per_kah: float = Field(
        gt=0, description="Calibrated linear consumption rate [in/(kA*h)]"
    )
    current_ka: float = Field(ge=0, description="Constant furnace current [kA]")
    time_hrs: float = Field(ge=0, description="Elapsed operating time [h]")


class ElectrodeAdvancementResponse(BaseModel):
    """Finite non-negative consumption estimate, not a furnace safety approval."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    consumption_in: float = Field(
        ge=0, description="Estimated electrode consumption [in]"
    )
