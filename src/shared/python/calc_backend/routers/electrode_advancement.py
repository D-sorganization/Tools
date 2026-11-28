"""REST adapter for the existing linear electrode consumption engine."""

import math

from fastapi import APIRouter, HTTPException

from shared.python.sidekick.process_calculators import (
    electrode_advancement_calculator,
)

from ..contracts.electrode_advancement import (
    ElectrodeAdvancementRequest,
    ElectrodeAdvancementResponse,
)

router = APIRouter(
    prefix="/api/calc/electrode-advancement", tags=["electrode-advancement"]
)


@router.post("", response_model=ElectrodeAdvancementResponse)
def calculate_electrode_advancement(
    request: ElectrodeAdvancementRequest,
) -> ElectrodeAdvancementResponse:
    """Estimate consumption at constant current using a caller-calibrated rate.

    Preconditions: finite positive rate and finite non-negative current/time.
    Postcondition: finite non-negative consumption in inches, or HTTP 422 if
    the engine's product exceeds the floating-point range.
    """
    engine = electrode_advancement_calculator.ElectrodeAdvancementCalculator(
        request.consumption_rate_in_per_kah
    )
    consumption = engine.calculate_consumption(request.current_ka, request.time_hrs)
    if not math.isfinite(consumption):
        raise HTTPException(
            status_code=422, detail="Consumption exceeds finite numeric range"
        )
    return ElectrodeAdvancementResponse(consumption_in=consumption)
