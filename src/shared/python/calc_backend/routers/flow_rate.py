"""Flow rate converter router.  See issue #608."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..contracts.flow_rate import FlowRateConvertRequest, FlowRateConvertResponse

router = APIRouter(prefix="/api/calc/flow-rate", tags=["flow-rate"])

# ---------------------------------------------------------------------------
# Conversion factors -- all relative to SI base (kg/s, mol/s, m3/s)
# ---------------------------------------------------------------------------

MASS_TO_KG_S: dict[str, float] = {
    "kg/s": 1.0,
    "kg/h": 1.0 / 3600.0,
    "kg/min": 1.0 / 60.0,
    "g/s": 1e-3,
    "g/h": 1e-3 / 3600.0,
    "lb/s": 0.45359237,
    "lb/h": 0.45359237 / 3600.0,
    "lb/min": 0.45359237 / 60.0,
    "ton/h": 1000.0 / 3600.0,
}

MOLAR_TO_MOL_S: dict[str, float] = {
    "mol/s": 1.0,
    "mol/h": 1.0 / 3600.0,
    "mol/min": 1.0 / 60.0,
    "kmol/s": 1e3,
    "kmol/h": 1e3 / 3600.0,
    "kmol/min": 1e3 / 60.0,
    "lbmol/s": 453.59237,
    "lbmol/h": 453.59237 / 3600.0,
    "lbmol/min": 453.59237 / 60.0,
}

VOLUMETRIC_TO_M3_S: dict[str, float] = {
    "m3/s": 1.0,
    "m3/h": 1.0 / 3600.0,
    "m3/min": 1.0 / 60.0,
    "L/s": 1e-3,
    "L/min": 1e-3 / 60.0,
    "L/h": 1e-3 / 3600.0,
    "ft3/s": 0.028316846592,
    "ft3/min": 0.028316846592 / 60.0,
    "ft3/h": 0.028316846592 / 3600.0,
    "CFM": 0.028316846592 / 60.0,
    "GPM": 6.30902e-5,
}

_TABLES = {
    "mass": MASS_TO_KG_S,
    "molar": MOLAR_TO_MOL_S,
    "volumetric": VOLUMETRIC_TO_M3_S,
}


@router.post("", response_model=FlowRateConvertResponse)
def convert_flow_rate(request: FlowRateConvertRequest) -> FlowRateConvertResponse:
    """Convert a flow rate value between compatible units."""
    table = _TABLES.get(request.category)
    if table is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{request.category}'. "
            f"Must be one of: {', '.join(_TABLES)}",
        )

    if request.from_unit not in table:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown from_unit '{request.from_unit}' for category "
            f"'{request.category}'. Valid units: {', '.join(table)}",
        )
    if request.to_unit not in table:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown to_unit '{request.to_unit}' for category "
            f"'{request.category}'. Valid units: {', '.join(table)}",
        )

    # Convert: source -> base SI -> target
    base = request.value * table[request.from_unit]
    result = base / table[request.to_unit]

    return FlowRateConvertResponse(
        result=result,
        from_unit=request.from_unit,
        to_unit=request.to_unit,
        category=request.category,
    )
