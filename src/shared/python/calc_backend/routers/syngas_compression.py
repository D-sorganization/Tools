"""REST adapter and typed contract for one-stage syngas compression."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.python.sidekick.process_calculators.syngas_compression_engine import (
    CompressionStage,
    SyngasCompressionEngine,
)

CompressionType = Literal["isentropic", "polytropic", "isothermal"]


class SyngasCompressionRequest(BaseModel):
    """One compressor stage. Positive fractions are normalized by the engine."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    composition: dict[str, float] = Field(
        description="Species mole fractions or percentages"
    )
    inlet_pressure_bar: float = Field(gt=0, description="Inlet pressure [bar]")
    outlet_pressure_bar: float = Field(gt=0, description="Outlet pressure [bar]")
    inlet_temperature_k: float = Field(gt=0, description="Inlet temperature [K]")
    isentropic_efficiency: float = Field(
        gt=0, le=1, description="Isentropic efficiency (0, 1]"
    )
    flow_rate_kmol_h: float = Field(gt=0, description="Molar flow [kmol/h]")
    compression_type: CompressionType = Field(
        default="isentropic",
        description="isentropic, polytropic, or isothermal",
    )

    @model_validator(mode="after")
    def _compression_stage_is_physical(self) -> SyngasCompressionRequest:
        """Require a non-empty composition and a pressure increase."""
        if not self.composition:
            raise ValueError("composition must be non-empty")
        if self.outlet_pressure_bar <= self.inlet_pressure_bar:
            raise ValueError("outlet_pressure_bar must exceed inlet_pressure_bar")
        total = 0.0
        for species, fraction in self.composition.items():
            if fraction < 0.0:
                raise ValueError(f"{species} mole fraction must be >= 0")
            total += fraction
        if total <= 0.0:
            raise ValueError("composition must sum to a positive value")
        return self


class SyngasCompressionResponse(BaseModel):
    """Ideal-gas stage result. Not a design-manual approval."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pressure_ratio: float = Field(gt=0, description="Outlet/inlet pressure ratio")
    heat_capacity_ratio: float = Field(gt=1, description="Mixture heat-capacity ratio")
    work_actual_j_per_mol: float = Field(gt=0, description="Shaft work [J/mol]")
    outlet_temperature_k: float = Field(gt=0, description="Discharge temperature [K]")
    heat_rise_k: float = Field(ge=0, description="Discharge minus inlet [K]")
    power_hp: float = Field(gt=0, description="Shaft power [hp]")
    compression_type: CompressionType


router = APIRouter(prefix="/api/calc/syngas-compression", tags=["syngas-compression"])

_PRESSURE_RATIO_TOLERANCE = 1e-9


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _reject_unknown_species(
    engine: SyngasCompressionEngine, composition: dict[str, float]
) -> None:
    """Preconditions: every species exists in the engine database."""
    database = engine.species_db
    missing = sorted(name for name in composition if database.get_species(name) is None)
    if missing:
        raise ValueError("unknown species: " + ", ".join(missing))


def _response_from_engine(
    request: SyngasCompressionRequest, raw: dict[str, Any], gamma: float
) -> SyngasCompressionResponse:
    """Postcondition: positive finite power and a physical discharge temperature."""
    pressure_ratio = float(raw["pressure_ratio"])
    work = float(raw["work_actual"])
    outlet = float(raw["temp_out_actual"])
    power = float(raw["power_hp"])
    heat_rise = float(raw["heat_rise"])
    expected = request.outlet_pressure_bar / request.inlet_pressure_bar
    values = (pressure_ratio, work, outlet, power, heat_rise, gamma)
    if not all(_is_finite(value) for value in values):
        raise RuntimeError("compression result is not finite")
    tolerance = _PRESSURE_RATIO_TOLERANCE * max(1.0, abs(expected))
    if abs(pressure_ratio - expected) > tolerance:
        raise RuntimeError("pressure ratio does not match the request")
    if work <= 0.0 or power <= 0.0 or outlet <= 0.0 or gamma <= 1.0:
        raise RuntimeError("work, power, temperature, and gamma must be positive")
    isothermal = request.compression_type == "isothermal"
    if isothermal and outlet != request.inlet_temperature_k:
        raise RuntimeError("isothermal discharge must equal the inlet temperature")
    if not isothermal and outlet <= request.inlet_temperature_k:
        raise RuntimeError("discharge temperature must exceed the inlet")
    if heat_rise < 0.0:
        raise RuntimeError("heat rise must be non-negative")
    return SyngasCompressionResponse(
        pressure_ratio=pressure_ratio,
        heat_capacity_ratio=gamma,
        work_actual_j_per_mol=work,
        outlet_temperature_k=outlet,
        heat_rise_k=heat_rise,
        power_hp=power,
        compression_type=request.compression_type,
    )


def _evaluate(request: SyngasCompressionRequest) -> SyngasCompressionResponse:
    engine = SyngasCompressionEngine()
    _reject_unknown_species(engine, request.composition)
    props = engine.calculate_mixture_properties(request.composition)
    stage = CompressionStage(
        inlet_pressure=request.inlet_pressure_bar,
        outlet_pressure=request.outlet_pressure_bar,
        inlet_temperature=request.inlet_temperature_k,
        efficiency=request.isentropic_efficiency,
        compression_type=request.compression_type,
    )
    raw = engine.calculate_compression_work(stage, request.flow_rate_kmol_h, props)
    return _response_from_engine(request, raw, float(props["heat_capacity_ratio"]))


@router.post("", response_model=SyngasCompressionResponse)
def calculate_syngas_compression(
    request: SyngasCompressionRequest,
) -> SyngasCompressionResponse:
    """Evaluate one ideal-gas compression stage.

    Preconditions: known species, outlet pressure above inlet, and positive
    temperature, flow, and efficiency in (0, 1].
    Postcondition: positive finite shaft power, or HTTP 422 when the engine
    rejects the stage.
    """
    try:
        return _evaluate(request)
    except (ValueError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
