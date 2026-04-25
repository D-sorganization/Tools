"""Pressure drop calculator router.  See issue #613."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

from ..contracts.pressure_drop import PressureDropRequest, PressureDropResponse

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])

# Universal gas constant [J/(mol*K)]
_R_UNIVERSAL = 8.314462618


@router.post("", response_model=PressureDropResponse)
def calculate_pressure_drop(request: PressureDropRequest) -> PressureDropResponse:
    """Calculate pressure drop using Darcy-Weisbach equation.

    Uses an inline implementation matching the legacy PressureDropCalculator
    to avoid import issues with the advanced pressure_drop_calculator package.
    """
    try:
        # Gas density (ideal gas)
        density = (request.pressure_pa * request.molecular_weight_kg_mol) / (
            _R_UNIVERSAL * request.temperature_k
        )

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

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PressureDropResponse(
        pressure_drop_pa=pressure_drop_pa,
        reynolds_number=re,
        friction_factor=friction_factor,
        velocity_m_s=velocity,
        flow_regime=flow_regime,
        density_kg_m3=density,
        viscosity_pa_s=viscosity,
    )
