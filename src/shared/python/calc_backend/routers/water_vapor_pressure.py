"""REST adapter for the shared water saturation-pressure correlations."""

from fastapi import APIRouter, HTTPException

from shared.python.sidekick.process_calculators.syngas_water_calculator import (
    SyngasWaterCalculator,
)

from ..contracts.water_vapor_pressure import (
    WaterVaporPressureRequest,
    WaterVaporPressureResponse,
)

router = APIRouter(
    prefix="/api/calc/water-vapor-pressure", tags=["water-vapor-pressure"]
)


@router.post("", response_model=WaterVaporPressureResponse)
def calculate_water_vapor_pressure(
    request: WaterVaporPressureRequest,
) -> WaterVaporPressureResponse:
    """Evaluate saturation pressure of liquid water at a given temperature.

    Preconditions: finite temperature within the selected method's range.
    Postcondition: finite non-negative pressure in Pa, or HTTP 422 if the
    underlying correlation rejects the temperature.
    """
    try:
        pressure_pa, method_used = SyngasWaterCalculator().calculate_vapor_pressure(
            request.temperature_c, request.method
        )
    except (ValueError, ArithmeticError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return WaterVaporPressureResponse(pressure_pa=pressure_pa, method_used=method_used)
