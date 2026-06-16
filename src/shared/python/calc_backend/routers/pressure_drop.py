"""Pressure drop calculator router with standardized API responses.

Provides endpoints for calculating pressure drop using the Darcy-Weisbach equation.
All responses follow the StandardResponse format with consistent error handling.

See issue #613 (calc backend) and #2411 (API standardization).
"""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import APIRouter, HTTPException

from shared.python.sidekick.api import (
    ErrorCode,
    StandardResponseBuilder,
)
from shared.python.sidekick.process_calculators.pressure_drop_calculator import (
    PressureDropCalculator,
    PressureDropResult,
)

from ..models.pressure_drop import PressureDropRequest, PressureDropResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calc/pressure-drop", tags=["pressure-drop"])

_calculator = PressureDropCalculator()


class _PressureDropApiResponse(dict[str, Any]):
    """Standard-response dict with direct access to pressure-drop data fields."""

    def __getattr__(self, name: str) -> Any:
        data = self.get("data")
        if isinstance(data, dict) and name in data:
            return data[name]
        raise AttributeError(name)


@router.post("")
def calculate_pressure_drop(
    request: PressureDropRequest,
) -> dict[str, Any]:
    """Calculate pressure drop using Darcy-Weisbach equation.

    Delegates to the shared Sidekick PressureDropCalculator to avoid duplicating
    the Darcy-Weisbach implementation inline. See GH1705.

    Uses standardized response format (issue #2411).

    Args:
        request: PressureDropRequest with pipe and flow parameters.

    Returns:
        StandardResponse with status="success" and calculated values, or
        status="error" with error details and HTTP 422 status code.

    Example request:
        ```json
        {
            "pipe_diameter_m": 0.1,
            "pipe_length_m": 100.0,
            "roughness_m": 0.000045,
            "flow_rate_kg_s": 5.0,
            "temperature_k": 300.0,
            "pressure_pa": 101325.0,
            "molecular_weight_kg_mol": 28.97
        }
        ```

    Example success response:
        ```json
        {
            "status": "success",
            "data": {
                "pressure_drop_pa": 1023.4,
                "reynolds_number": 50000.0,
                "friction_factor": 0.025,
                "velocity_m_s": 45.2,
                "flow_regime": "Turbulent",
                "density_kg_m3": 1.177,
                "viscosity_pa_s": 1.86e-5
            },
            "error": null,
            "metadata": {
                "request_id": "...",
                "processing_time_ms": 125.0,
                "api_version": "1.0.0"
            }
        }
        ```
    """
    builder = StandardResponseBuilder()

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
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        response = builder.error(
            code=ErrorCode.INVALID_INPUT,
            message=f"Pressure drop calculation failed: {str(exc)}",
            details={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
        raise HTTPException(status_code=422, detail=response.to_dict())  # noqa: B904

    response_data = PressureDropResponse(
        pressure_drop_pa=result.pressure_drop_pa,
        reynolds_number=result.reynolds_number,
        friction_factor=result.friction_factor,
        velocity_m_s=result.velocity,
        flow_regime=result.flow_regime,
        density_kg_m3=result.density,
        viscosity_pa_s=result.viscosity,
    ).model_dump()

    response = builder.success(
        data=response_data,
    )
    return cast(dict[str, Any], _PressureDropApiResponse(response.to_dict()))
