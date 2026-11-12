"""Attach catalog specs to imported router handlers (issue #6020)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from ..routers.acid_gas_dewpoint import calculate_acid_gas_dewpoint
from ..routers.acid_gas_dewpoint import router as acid_router
from ..routers.baghouse import calculate_baghouse
from ..routers.baghouse import router as baghouse_router
from ..routers.financial import calculate_financial
from ..routers.financial import router as financial_router
from ..routers.flare import calculate_flare
from ..routers.flare import router as flare_router
from ..routers.flow_rate import convert_flow_rate
from ..routers.flow_rate import router as flow_rate_router
from ..routers.ode_solver import router as ode_router
from ..routers.ode_solver import solve_ode
from ..routers.pressure_drop import calculate_pressure_drop
from ..routers.pressure_drop import router as pressure_router
from ..routers.rotation_converter import (
    compute_reference_frame_conversion,
    compute_rotation,
)
from ..routers.rotation_converter import (
    router as rotation_router,
)
from ..routers.scrubber import calculate_scrubber
from ..routers.scrubber import router as scrubber_router
from ..routers.symbolic_solver import (
    compute_derivative,
    simplify_expression,
    solve_equation,
)
from ..routers.symbolic_solver import (
    router as symbolic_router,
)
from ..routers.syngas_water import calculate_syngas_water
from ..routers.syngas_water import router as syngas_router
from ..routers.thermal_profile import predict_thermal_profile
from ..routers.thermal_profile import router as thermal_router
from ..routers.wgs_reactor import calculate_wgs
from ..routers.wgs_reactor import router as wgs_router
from .store import register_from_catalog


def wire_calculator_registry() -> None:
    """Register every POST calculator exposed by ``CALCULATOR_ROUTERS``."""
    pairs: tuple[tuple[str, APIRouter, Callable[..., object]], ...] = (
        ("acid-gas-dewpoint", acid_router, calculate_acid_gas_dewpoint),
        ("baghouse", baghouse_router, calculate_baghouse),
        ("financial", financial_router, calculate_financial),
        ("flare", flare_router, calculate_flare),
        ("flow-rate", flow_rate_router, convert_flow_rate),
        ("ode-solver", ode_router, solve_ode),
        ("pressure-drop", pressure_router, calculate_pressure_drop),
        ("rotation-converter", rotation_router, compute_rotation),
        (
            "rotation-reference-frame",
            rotation_router,
            compute_reference_frame_conversion,
        ),
        ("scrubber", scrubber_router, calculate_scrubber),
        ("symbolic-solve", symbolic_router, solve_equation),
        ("symbolic-derivative", symbolic_router, compute_derivative),
        ("symbolic-simplify", symbolic_router, simplify_expression),
        ("syngas-water", syngas_router, calculate_syngas_water),
        ("thermal-profile", thermal_router, predict_thermal_profile),
        ("wgs-reactor", wgs_router, calculate_wgs),
    )
    for catalog_id, router, handler in pairs:
        register_from_catalog(catalog_id, router=router, fn=handler)
