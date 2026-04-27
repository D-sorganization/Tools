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
            viscosity_pa_s=request.viscosity_pa_s,
        )
<<<<<<< HEAD
=======

        if density <= 0:
            raise HTTPException(422, "degenerate input: density must be positive")

        if request.dynamic_viscosity_pa_s is not None:
            viscosity = request.dynamic_viscosity_pa_s
        else:
            sutherland_map = {
                "air": (291.15, 1.827e-5, 120.0),
                "steam": (373.15, 1.22e-5, 961.0),
                "ch4": (293.15, 1.09e-5, 164.0),
                "h2": (293.15, 8.76e-6, 72.0),
                "n2": (300.55, 1.781e-5, 111.0),
                "co2": (293.15, 1.47e-5, 240.0),
            }
            gas_key = request.gas_name.lower().strip() if request.gas_name else ""
            if gas_key in sutherland_map:
                t_ref, mu_ref, s_const = sutherland_map[gas_key]
                viscosity = (
                    mu_ref
                    * ((t_ref + s_const) / (request.temperature_k + s_const))
                    * (request.temperature_k / t_ref) ** 1.5
                )
            else:
                # Kinetic theory fallback scaling from air based on sqrt(M_gas / M_air)
                # Air molecular weight is ~0.02897 kg/mol
                m_air = 0.02897
                t_ref, mu_ref, s_const = sutherland_map["air"]
                mu_air = (
                    mu_ref
                    * ((t_ref + s_const) / (request.temperature_k + s_const))
                    * (request.temperature_k / t_ref) ** 1.5
                )
                viscosity = mu_air * math.sqrt(request.molecular_weight_kg_mol / m_air)

        if viscosity <= 0:
            raise HTTPException(422, "degenerate input: viscosity must be positive")

        # Velocity
        vol_flow = request.flow_rate_kg_s / density
        area = math.pi * (request.pipe_diameter_m / 2) ** 2
        if area <= 0:
            raise HTTPException(422, "degenerate input: area must be positive")
        velocity = vol_flow / area

        # Reynolds number
        re = (density * velocity * request.pipe_diameter_m) / viscosity

        # Friction factor
        rel_roughness = request.roughness_m / request.pipe_diameter_m

        if re > 4000:
            a_val = rel_roughness / 3.7
            b_val = 5.74 / (re**0.9)
            try:
                friction_factor = 0.25 / (math.log10(a_val + b_val) ** 2)
            except ValueError:
                friction_factor = 0.02
        elif re > 2300:
            friction_factor = 0.03
        else:
            friction_factor = 64 / re if re > 0 else 0.05

        # Darcy-Weisbach pressure drop
        pressure_drop_pa = (
            friction_factor
            * (request.pipe_length_m / request.pipe_diameter_m)
            * (density * velocity**2 / 2)
        )

        # Flow regime
        if re < 2300:
            flow_regime = "Laminar"
        elif re < 4000:
            flow_regime = "Transitional"
        else:
            flow_regime = "Turbulent"

>>>>>>> origin/main
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
