"""Pressure drop calculator router.  See issue #613."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
    PressureDropCalculator,
    PressureDropResult,
)

from ..contracts.pressure_drop import PressureDropRequest, PressureDropResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])

_calculator = PressureDropCalculator()


@router.post("", response_model=PressureDropResponse)
def calculate_pressure_drop(request: PressureDropRequest) -> PressureDropResponse:
    """Calculate pressure drop using Darcy-Weisbach equation.

    Delegates to PressureDropCalculator from upstream_drift_tools to avoid
    duplicating the Darcy-Weisbach implementation inline. See GH1705.
    """
    try:
        result: PressureDropResult = _calculator.calculate_pressure_drop(
            pipe_diameter_m=request.pipe_diameter_m,
            pipe_length_m=request.pipe_length_m,
            roughness_m=request.roughness_m,
            flow_rate_kg_s=request.flow_rate_kg_s,
            temperature_k=request.temperature_k,
            pressure_pa=request.pressure_pa,
            molecular_weight_kg_mol=request.molecular_weight_kg_mol,
        )
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PressureDropResponse(
        pressure_drop_pa=result.pressure_drop_pa,
        reynolds_number=result.reynolds_number,
        friction_factor=result.friction_factor,
        velocity_m_s=result.velocity,
        flow_regime=result.flow_regime,
        density_kg_m3=result.density,
        viscosity_pa_s=result.viscosity,
    )
