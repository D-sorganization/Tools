"""Pressure drop calculator router.  See issue #613."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.pressure_drop import PressureDropRequest, PressureDropResponse

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])


@router.post("", response_model=PressureDropResponse)
def calculate_pressure_drop(request: PressureDropRequest) -> PressureDropResponse:
    """Calculate pressure drop using Darcy-Weisbach equation.

    Delegates to the canonical ``PressureDropCalculator`` from
    ``upstream_drift_tools`` so that physics constants and the
    Swamee-Jain / Sutherland implementations live in exactly one place.
    See issue #1705.
    """
    from upstream_drift_tools.process_calculators import PressureDropCalculator

    calc = PressureDropCalculator()

    try:
        result = calc.calculate_pressure_drop(
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
