"""Typed pressure-drop-v2 calculator metadata (Refs #6019)."""

from typing import Any

from ..contracts.pressure_drop_v2 import PressureDropRequestV2, PressureDropResponseV2
from .spec import WorkedExample

PRESSURE_DROP_V2: dict[str, Any] = {
    "id": "pressure-drop-v2",
    "title": "Pressure drop (typed v2 response)",
    "category": "process",
    "request_model": PressureDropRequestV2,
    "response_model": PressureDropResponseV2,
    "maturity": "beta",
    "formula_ids": (),
    "units": {
        "pipe_diameter_m": "m",
        "pipe_length_m": "m",
        "roughness_m": "m",
        "flow_rate_kg_s": "kg/s",
        "temperature_k": "K",
        "pressure_pa": "Pa",
        "molecular_weight_kg_mol": "kg/mol",
        "viscosity_pa_s": "Pa*s",
    },
    "worked_example": WorkedExample(
        request={
            "pipe_diameter_m": 0.1,
            "pipe_length_m": 10.0,
            "flow_rate_kg_s": 0.001,
            "temperature_k": 300.0,
            "pressure_pa": 101325.0,
            "molecular_weight_kg_mol": 0.029,
            "viscosity_pa_s": 1.8e-5,
        },
        response_summary={"status": "success", "data.pressure_drop_pa": "> 0"},
    ),
    "route_subpath": "",
}

PRESSURE_DROP_V2_VALIDATE: dict[str, Any] = {
    **PRESSURE_DROP_V2,
    "id": "pressure-drop-v2-validate",
    "title": "Validate pressure drop v2 inputs",
    "response_model": None,
    "route_subpath": "validate",
    "worked_example": WorkedExample(
        request=PRESSURE_DROP_V2["worked_example"].request,
        response_summary={"data.valid": True},
    ),
}
