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
from pydantic import BaseModel, Field
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


class SteamRequest(BaseModel):
    """Input for a steam property calculation."""

    mode: CalculationMode = Field(description="Calculation mode")
    temperature: float = Field(gt=0, description="Temperature in Kelvin")
    pressure: float = Field(gt=0, description="Pressure in Pascals")
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
        # The engine the caller asked for (or that auto-selection intended).
        requested_engine = _engine._select_best_engine(request.engine)

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
