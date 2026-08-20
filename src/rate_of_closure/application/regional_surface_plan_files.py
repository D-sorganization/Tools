"""Strict native file adapter for canonical regional surface-plan requests."""

from __future__ import annotations

from pathlib import Path

from shared.python.swing_sim.ground.regional_plan_records import (
    MAX_REGIONAL_PLAN_WIRE_BYTES,
    GroundRegionalMaterialPlanRequest,
)
from shared.python.swing_sim.ground.regional_plan_wire import (
    regional_material_plan_request_from_json,
)

from .atomic_text_files import write_utf8_text_atomic
from .bounded_text_files import read_bounded_utf8


def read_regional_surface_plan_request(
    source: str | Path,
) -> GroundRegionalMaterialPlanRequest:
    """Read and completely validate one canonical request before returning it."""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"regional surface plan does not exist: {path}")
    return regional_material_plan_request_from_json(
        read_bounded_utf8(
            path,
            MAX_REGIONAL_PLAN_WIRE_BYTES,
            "regional material plan",
        )
    )


def write_regional_surface_plan_request_atomic(
    request: GroundRegionalMaterialPlanRequest,
    destination: str | Path | None,
) -> bool:
    """Write exact canonical request bytes, or return false after cancellation."""
    if destination is None:
        return False
    if type(request) is not GroundRegionalMaterialPlanRequest:
        raise TypeError("request must be an exact GroundRegionalMaterialPlanRequest")
    # The protected delta gate skips imported modules, so annotate this local
    # boundary explicitly without a cast that becomes redundant when the
    # helper is included in the same MyPy root set.
    write_succeeded: bool = write_utf8_text_atomic(
        request.to_json(), destination, document_name="regional surface plan"
    )
    return write_succeeded


__all__ = [
    "read_regional_surface_plan_request",
    "write_regional_surface_plan_request_atomic",
]
