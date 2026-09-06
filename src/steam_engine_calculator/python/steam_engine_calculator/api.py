"""FastAPI backend for Steam Engine Calculator.

Wraps the validated SteamCalculationEngine (CoolProp / Cantera / simplified)
so the React frontend gets physically accurate results instead of hardcoded
ideal-gas constants.

See issue #605.

Usage:
    uvicorn steam_engine_calculator.api:app --reload --port 8002
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from utils.compatibility import StrEnum

from shared.python.cors import add_cors_middleware
from shared.python.sidekick.calculators.thermo.steam_engine import (
    SteamCalculationEngine,
    SteamProperties,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Steam Engine Calculator API",
    description="Thermodynamic property calculations for water/steam",
    version="1.0.0",
)
add_cors_middleware(app)

# Singleton engine (initialised once, reused across requests)
_engine = SteamCalculationEngine()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CalculationMode(StrEnum):
    """Supported calculation modes."""

    TP = "tp"
    SAT_T = "sat_t"
    SAT_P = "sat_p"


# Inputs each mode must carry; enforced by SteamRequest's model validator so
# saturation modes stop demanding invented placeholder values (issue #3980).
_MODES_REQUIRING_TEMPERATURE = frozenset({CalculationMode.TP, CalculationMode.SAT_T})
_MODES_REQUIRING_PRESSURE = frozenset({CalculationMode.TP, CalculationMode.SAT_P})


class SteamRequest(BaseModel):
    """Input for a steam property calculation.

    Only the inputs the requested mode consumes are required (issue #3980):
    ``tp`` needs temperature and pressure, ``sat_t`` is driven by temperature
    alone and ``sat_p`` by pressure alone.  Sending the mode-foreign field is
    still accepted so clients that always send both keep working.
    """

    mode: CalculationMode = Field(description="Calculation mode")
    temperature: float | None = Field(
        default=None,
        gt=0,
        description="Temperature in Kelvin (required for 'tp' and 'sat_t')",
    )
    pressure: float | None = Field(
        default=None,
        gt=0,
        description="Pressure in Pascals (required for 'tp' and 'sat_p')",
    )
    engine: str = Field(
        default="auto",
        description="Calculation engine: 'coolprop', 'cantera', 'simplified', 'auto'",
    )

    @model_validator(mode="after")
    def _require_mode_inputs(self) -> SteamRequest:
        """DbC: reject requests missing the inputs their mode consumes."""
        missing: list[str] = []
        if self.temperature is None and self.mode in _MODES_REQUIRING_TEMPERATURE:
            missing.append("temperature")
        if self.pressure is None and self.mode in _MODES_REQUIRING_PRESSURE:
            missing.append("pressure")
        if missing:
            raise ValueError(f"mode '{self.mode.value}' requires: {', '.join(missing)}")
        return self


class SteamResponse(BaseModel):
    """Full set of thermodynamic properties returned to the client."""

    temperature: float
    pressure: float
    density: float
    specificVolume: float
    enthalpy: float
    entropy: float
    internalEnergy: float
    cp: float
    cv: float
    speedOfSound: float
    thermalConductivity: float
    dynamicViscosity: float
    kinematicViscosity: float
    quality: float
    phase: str
    compressibilityFactor: float
    prandtlNumber: float
    specificHeatRatio: float
    engine: str = Field(description="Which backend engine was used")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _props_to_response(props: SteamProperties, engine_used: str) -> SteamResponse:
    """Convert a SteamProperties dataclass to a SteamResponse model."""
    return SteamResponse(
        temperature=props.temperature,
        pressure=props.pressure,
        density=props.density,
        specificVolume=props.specific_volume,
        enthalpy=props.enthalpy,
        entropy=props.entropy,
        internalEnergy=props.internal_energy,
        cp=props.cp,
        cv=props.cv,
        speedOfSound=props.speed_of_sound,
        thermalConductivity=props.thermal_conductivity,
        dynamicViscosity=props.dynamic_viscosity,
        kinematicViscosity=props.kinematic_viscosity,
        quality=props.quality,
        phase=props.phase,
        compressibilityFactor=props.compressibility_factor or 0.0,
        prandtlNumber=props.prandtl_number or 0.0,
        specificHeatRatio=props.specific_heat_ratio or 0.0,
        engine=engine_used,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/steam/calculate", response_model=SteamResponse)
def calculate_steam(request: SteamRequest) -> SteamResponse:
    """Calculate steam properties using the validated Python engine.

    Replaces the hardcoded ideal-gas approximations that were previously
    baked into the React frontend.  See issue #605.
    """
    try:
        # The engine the caller asked for (or that auto-selection intended).
        requested_engine = _engine.select_best_engine(request.engine)

        if request.mode == CalculationMode.TP:
            if request.temperature is None or request.pressure is None:
                # The model validator rejects this over HTTP; direct callers
                # of calculate_steam hit the endpoint's own DbC precondition.
                raise HTTPException(
                    status_code=400,
                    detail="mode 'tp' requires temperature and pressure",
                )
            props = _engine.calculate_properties(
                request.temperature, request.pressure, engine=request.engine
            )
        elif request.mode == CalculationMode.SAT_T:
            if request.temperature is None:
                raise HTTPException(
                    status_code=400, detail="mode 'sat_t' requires temperature"
                )
            props = _engine.calculate_saturated_properties_from_temperature(
                request.temperature, engine=request.engine
            )
        elif request.mode == CalculationMode.SAT_P:
            if request.pressure is None:
                raise HTTPException(
                    status_code=400, detail="mode 'sat_p' requires pressure"
                )
            props = _engine.calculate_saturated_properties_from_pressure(
                request.pressure, engine=request.engine
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

        # Report the engine that ACTUALLY produced the numbers. After an internal
        # fallback the accurate backend may have been replaced by the simplified
        # correlations; reporting the requested engine would mislabel the result
        # (issue #3318). ``engine_used`` is populated by calculate_properties; if
        # absent (older paths) fall back to the requested engine.
        engine_name = props.engine_used or requested_engine

        return _props_to_response(props, engine_name)

    except HTTPException:
        raise
    except ValueError as exc:
        logger.info("Invalid steam calculation request: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (TypeError, RuntimeError, ArithmeticError) as exc:
        logger.exception("Steam calculation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
