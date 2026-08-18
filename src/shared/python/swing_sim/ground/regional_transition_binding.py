"""Validate regional transition identities against an embedded plan."""

from __future__ import annotations

from ._vector_math import dot, subtract
from .regional_plan_records import GroundRegionalMaterialPlanRequest
from .regional_surface_types import SurfaceRegionTransition

_BOUNDARY_TOLERANCE_M = 1e-9


def _selection(
    plan: GroundRegionalMaterialPlanRequest,
    coordinate_m: float,
) -> tuple[str | None, str]:
    matches = tuple(
        region
        for region in plan.regions
        if region.lower_coordinate_m < coordinate_m < region.upper_coordinate_m
    )
    if not matches:
        return None, plan.base_surface.surface_id
    selected = max(matches, key=lambda region: region.precedence)
    return selected.region_id, selected.surface.surface_id


def _boundary_sides(
    plan: GroundRegionalMaterialPlanRequest,
    coordinate_m: float,
) -> tuple[tuple[str | None, str], tuple[str | None, str]]:
    region_bounds = tuple(
        bound
        for region in plan.regions
        for bound in (region.lower_coordinate_m, region.upper_coordinate_m)
    )
    matching = tuple(
        bound
        for bound in region_bounds
        if abs(bound - coordinate_m) <= _BOUNDARY_TOLERANCE_M
    )
    if not matching:
        raise ValueError("transition crossing must lie on a regional plan boundary")
    boundary = min(matching, key=lambda bound: abs(bound - coordinate_m))
    all_bounds = tuple(
        sorted({plan.lower_coordinate_m, plan.upper_coordinate_m, *region_bounds})
    )
    lower = max(bound for bound in all_bounds if bound < boundary)
    upper = min(bound for bound in all_bounds if bound > boundary)
    left = _selection(plan, lower + (boundary - lower) / 2.0)
    right = _selection(plan, boundary + (upper - boundary) / 2.0)
    return left, right


def validate_transition_against_plan(
    plan: GroundRegionalMaterialPlanRequest,
    transition: SurfaceRegionTransition,
    axis_velocity_m_s: float,
) -> None:
    """Require one ledger row to match a real plan boundary identity change."""
    coordinate = dot(
        subtract(transition.position_m, plan.axis_origin_m),
        plan.axis_unit,
    )
    left, right = _boundary_sides(plan, coordinate)
    declared = (
        (transition.from_region_id, transition.from_surface_id),
        (transition.to_region_id, transition.to_surface_id),
    )
    if left == right:
        raise ValueError("transition identities must match the regional plan crossing")
    if axis_velocity_m_s > 0.0:
        expected = (left, right)
    elif axis_velocity_m_s < 0.0:
        expected = (right, left)
    else:
        raise ValueError("transition direction must be nonzero at a plan crossing")
    if declared != expected:
        raise ValueError("transition identities must match the regional plan crossing")


__all__ = ["validate_transition_against_plan"]
