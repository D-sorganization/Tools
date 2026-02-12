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
import os
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from upstream_drift_tools.calculators.thermo.steam_engine import (
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

# Restrict CORS to known local development origins.
# Override with CORS_ORIGINS env var (comma-separated) if needed.
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
_env_origins = os.environ.get("CORS_ORIGINS")
_cors_origins = _env_origins.split(",") if _env_origins else _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton engine (initialised once, reused across requests)
_engine = SteamCalculationEngine()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CalculationMode(str, Enum):
    """Supported calculation modes."""

    TP = "tp"
    SAT_T = "sat_t"
    SAT_P = "sat_p"


class SteamRequest(BaseModel):
    """Input for a steam property calculation."""

    mode: CalculationMode = Field(description="Calculation mode")
    temperature: float = Field(description="Temperature in Kelvin")
    pressure: float = Field(description="Pressure in Pascals")
    engine: str = Field(
        default="auto",
        description="Calculation engine: 'coolprop', 'cantera', 'simplified', 'auto'",
    )


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
        engine_name = _engine._select_best_engine(request.engine)

        if request.mode == CalculationMode.TP:
            props = _engine.calculate_properties(
                request.temperature, request.pressure, engine=request.engine
            )
        elif request.mode == CalculationMode.SAT_T:
            props = _engine.calculate_saturated_properties_from_temperature(
                request.temperature, engine=request.engine
            )
        elif request.mode == CalculationMode.SAT_P:
            props = _engine.calculate_saturated_properties_from_pressure(
                request.pressure, engine=request.engine
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

        return _props_to_response(props, engine_name)

    except HTTPException:
        raise
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        logger.exception("Steam calculation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
